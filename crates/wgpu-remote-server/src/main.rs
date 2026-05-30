//! `wgpu-remote-server` — accepts QUIC connections from a `wgpu-remote-client`
//! and replays GPU actions against the local wgpu device.
//!
//! Supports two modes:
//!   - **CA mode** (default): loads or generates a private CA, issues a server
//!     cert on startup. Clients pin the CA cert.
//!   - **Self-signed mode** (`--self-signed`): generates a throwaway self-signed
//!     cert on every startup, like v0. Useful for quick testing.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use wgpu_remote_server::{Engine, run_connection};
use wgpu_remote_transport::pki::CertAuthority;
use wgpu_remote_transport::quic::QuicEndpoint;

const USAGE: &str = "\
wgpu-remote-server — proxy wgpu calls to this machine's GPU over QUIC

USAGE:
    wgpu-remote-server [OPTIONS]
    wgpu-remote-server init-ca [OPTIONS]

SUBCOMMANDS:
    init-ca             Generate a new CA keypair and write it to disk.
                        Does not start the server.

SERVER OPTIONS:
    --bind <ADDR>       Address to listen on  [default: 0.0.0.0:4433]
    --ca-cert <PATH>    CA certificate (DER) [default: ./ca-cert.der]
    --ca-key <PATH>     CA private key (DER) [default: ./ca-key.der]
    --san <NAME>        Extra SAN for the server cert. Can be repeated.
                        `localhost` and the bind IP are always included.
    --cert-out <PATH>   Write the CA cert (DER) here so clients can grab it.
                        Only used with --self-signed to write the self-signed
                        cert. In CA mode the CA cert is already at --ca-cert.
    --self-signed       Use a throwaway self-signed cert instead of a CA.
                        Equivalent to the old v0 behavior.
    --port-file <PATH>  Write the actual bound port (ASCII) to this file.
    -h, --help          Print this help

INIT-CA OPTIONS:
    --ca-cert <PATH>    Where to write the CA cert [default: ./ca-cert.der]
    --ca-key <PATH>     Where to write the CA key  [default: ./ca-key.der]
    -h, --help          Print this help
";

#[derive(Debug)]
enum Cmd {
    Serve(ServeArgs),
    InitCa(CaArgs),
}

#[derive(Debug)]
struct ServeArgs {
    bind: SocketAddr,
    ca_cert: PathBuf,
    ca_key: PathBuf,
    extra_sans: Vec<String>,
    cert_out: Option<PathBuf>,
    self_signed: bool,
    port_file: Option<PathBuf>,
}

#[derive(Debug)]
struct CaArgs {
    ca_cert: PathBuf,
    ca_key: PathBuf,
}

fn parse_args() -> Result<Cmd, String> {
    let mut argv = std::env::args().skip(1).peekable();

    // Check for subcommand
    let is_init_ca = argv.peek().map(|s| s.as_str()) == Some("init-ca");
    if is_init_ca {
        argv.next();
    }

    let mut bind: SocketAddr = "0.0.0.0:4433".parse().unwrap();
    let mut ca_cert = PathBuf::from("./ca-cert.der");
    let mut ca_key = PathBuf::from("./ca-key.der");
    let mut extra_sans: Vec<String> = Vec::new();
    let mut cert_out: Option<PathBuf> = None;
    let mut self_signed = false;
    let mut port_file: Option<PathBuf> = None;

    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--bind" => {
                let v = argv.next().ok_or("--bind requires a value")?;
                bind = v.parse().map_err(|e| format!("invalid --bind: {e}"))?;
            }
            "--ca-cert" => {
                let v = argv.next().ok_or("--ca-cert requires a value")?;
                ca_cert = PathBuf::from(v);
            }
            "--ca-key" => {
                let v = argv.next().ok_or("--ca-key requires a value")?;
                ca_key = PathBuf::from(v);
            }
            "--san" => {
                let v = argv.next().ok_or("--san requires a value")?;
                extra_sans.push(v);
            }
            "--cert-out" => {
                let v = argv.next().ok_or("--cert-out requires a value")?;
                cert_out = Some(PathBuf::from(v));
            }
            "--self-signed" => {
                self_signed = true;
            }
            "--port-file" => {
                let v = argv.next().ok_or("--port-file requires a value")?;
                port_file = Some(PathBuf::from(v));
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    if is_init_ca {
        Ok(Cmd::InitCa(CaArgs { ca_cert, ca_key }))
    } else {
        Ok(Cmd::Serve(ServeArgs {
            bind,
            ca_cert,
            ca_key,
            extra_sans,
            cert_out,
            self_signed,
            port_file,
        }))
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    if let Err(e) = real_main().await {
        eprintln!("error: {e}");
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

async fn real_main() -> anyhow::Result<()> {
    let cmd = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}\n");
            eprint!("{USAGE}");
            std::process::exit(2);
        }
    };

    let _ = rustls::crypto::ring::default_provider().install_default();

    match cmd {
        Cmd::InitCa(args) => init_ca(args),
        Cmd::Serve(args) => serve(args).await,
    }
}

fn init_ca(args: CaArgs) -> anyhow::Result<()> {
    let ca = CertAuthority::generate()?;
    ca.save(&args.ca_cert, &args.ca_key)?;
    eprintln!("CA cert written to {}", args.ca_cert.display());
    eprintln!("CA key  written to {}", args.ca_key.display());
    eprintln!("Give the CA cert to clients. Keep the CA key secret.");
    Ok(())
}

async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    let endpoint = if args.self_signed {
        // Legacy self-signed mode.
        let (ep, cert) = QuicEndpoint::server(args.bind)?;
        let cert_out = args.cert_out.unwrap_or_else(|| PathBuf::from("./server-cert.der"));
        std::fs::write(&cert_out, cert.as_ref())?;
        eprintln!("self-signed cert (DER) written to {}", cert_out.display());
        ep
    } else {
        // CA mode: load or auto-generate a CA, then issue a server cert.
        let ca = if args.ca_cert.exists() && args.ca_key.exists() {
            eprintln!("loading CA from {} + {}", args.ca_cert.display(), args.ca_key.display());
            CertAuthority::load(&args.ca_cert, &args.ca_key)?
        } else {
            eprintln!("no CA found — generating a new one");
            let ca = CertAuthority::generate()?;
            ca.save(&args.ca_cert, &args.ca_key)?;
            eprintln!("CA cert written to {}", args.ca_cert.display());
            eprintln!("CA key  written to {}", args.ca_key.display());
            ca
        };

        // Build SANs: always include localhost + the bind IP, plus any extras.
        let bind_ip = args.bind.ip().to_string();
        let mut sans: Vec<String> = vec!["localhost".to_string()];
        if bind_ip != "0.0.0.0" && bind_ip != "::" {
            sans.push(bind_ip);
        }
        sans.extend(args.extra_sans);
        let san_refs: Vec<&str> = sans.iter().map(|s| s.as_str()).collect();

        let server_cert = ca.issue_server_cert(&san_refs)?;
        let ep = QuicEndpoint::server_with_cert(args.bind, server_cert)?;
        eprintln!("server cert SANs: {}", sans.join(", "));
        ep
    };

    let local = endpoint.local_addr()?;
    if let Some(pf) = &args.port_file {
        std::fs::write(pf, local.port().to_string())?;
    }
    eprintln!("wgpu-remote-server listening on {local}");

    let engine = Arc::new(Engine::new().await?);
    eprintln!("wgpu engine ready");

    loop {
        match endpoint.accept().await {
            Ok(conn) => {
                eprintln!("accepted connection from {}", conn.remote_address());
                let engine = engine.clone();
                tokio::spawn(async move {
                    if let Err(e) = run_connection(engine, conn).await {
                        eprintln!("connection ended with error: {e}");
                    }
                });
            }
            Err(e) => {
                eprintln!("accept failed: {e}");
            }
        }
    }
}
