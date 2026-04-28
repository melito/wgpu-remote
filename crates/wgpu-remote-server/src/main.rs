//! `wgpu-remote-server` — accepts QUIC connections from a `wgpu-remote-client`
//! and replays GPU actions against the local wgpu device.
//!
//! v1 scope:
//!   - Single shared `Engine` (and therefore single shared GPU device).
//!   - Self-signed cert, written to disk so the client can pin it.
//!   - Sequential or concurrent connections — each gets the same engine.
//!     Resource IDs are not session-scoped yet, so multiple concurrent
//!     clients will collide. Document & defer to v1.1.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use wgpu_remote_server::{Engine, run_connection};
use wgpu_remote_transport::quic::QuicEndpoint;

const USAGE: &str = "\
wgpu-remote-server — proxy wgpu calls to this machine's GPU over QUIC

USAGE:
    wgpu-remote-server [OPTIONS]

OPTIONS:
    --bind <ADDR>       Address to listen on  [default: 0.0.0.0:4433]
    --cert-out <PATH>   Write the self-signed server cert (DER) to this file
                        so the client can pin it.  [default: ./server-cert.der]
    --port-file <PATH>  Write the actual bound port (as ASCII text, no trailing
                        newline) to this file. Useful when --bind specifies
                        port 0 and the OS picks one.
    -h, --help          Print this help

The cert is regenerated on every startup. The client must use the freshly
written cert; old certs are invalid.
";

#[derive(Debug)]
struct Args {
    bind: SocketAddr,
    cert_out: PathBuf,
    port_file: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut bind: SocketAddr = "0.0.0.0:4433".parse().unwrap();
        let mut cert_out = PathBuf::from("./server-cert.der");
        let mut port_file: Option<PathBuf> = None;
        let mut argv = std::env::args().skip(1);
        while let Some(arg) = argv.next() {
            match arg.as_str() {
                "--bind" => {
                    let v = argv.next().ok_or("--bind requires a value")?;
                    bind = v.parse().map_err(|e| format!("invalid --bind: {e}"))?;
                }
                "--cert-out" => {
                    let v = argv.next().ok_or("--cert-out requires a value")?;
                    cert_out = PathBuf::from(v);
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
        Ok(Self {
            bind,
            cert_out,
            port_file,
        })
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
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n");
            eprint!("{USAGE}");
            std::process::exit(2);
        }
    };

    // rustls needs a default crypto provider before any TLS work.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let (endpoint, cert) = QuicEndpoint::server(args.bind)?;
    let local = endpoint.local_addr()?;

    std::fs::write(&args.cert_out, cert.as_ref())?;
    if let Some(pf) = &args.port_file {
        std::fs::write(pf, local.port().to_string())?;
    }
    eprintln!("wgpu-remote-server listening on {local}");
    eprintln!("server cert (DER) written to {}", args.cert_out.display());

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
                // Don't exit on a single bad connection — keep listening.
            }
        }
    }
}
