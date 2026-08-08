//! Lightweight private CA for wgpu-remote.
//!
//! [`CertAuthority`] can:
//! - Generate a fresh CA keypair + self-signed root certificate.
//! - Load an existing CA from DER files on disk.
//! - Issue server certificates signed by the CA, with caller-supplied SANs.
//!
//! The CA cert is what clients pin (via `QuicEndpoint::client`). Server certs
//! can rotate freely without redistributing trust anchors.

use std::net::IpAddr;
use std::path::Path;

use rcgen::{
    BasicConstraints, CertificateParams, DnType, IsCa, KeyPair, KeyUsagePurpose, SanType,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

/// A private certificate authority that signs server certificates.
pub struct CertAuthority {
    ca_key: KeyPair,
    /// The rcgen Certificate used for signing leaf certs.
    ca_cert: rcgen::Certificate,
    /// The original DER bytes of the CA cert (stable across loads).
    /// This is what goes into cert chains and what clients pin.
    ca_cert_der: CertificateDer<'static>,
}

/// A server certificate + private key, ready for use with rustls / quinn.
pub struct ServerCert {
    pub cert_chain: Vec<CertificateDer<'static>>,
    pub private_key: PrivateKeyDer<'static>,
}

fn ca_params() -> CertificateParams {
    let mut params = CertificateParams::default();
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![
        KeyUsagePurpose::KeyCertSign,
        KeyUsagePurpose::CrlSign,
    ];
    params
        .distinguished_name
        .push(DnType::CommonName, "wgpu-remote CA");
    params
}

impl CertAuthority {
    /// Generate a new CA keypair and self-signed root certificate.
    pub fn generate() -> Result<Self, rcgen::Error> {
        let ca_key = KeyPair::generate()?;
        let params = ca_params();
        let ca_cert = params.self_signed(&ca_key)?;
        let ca_cert_der = ca_cert.der().clone();
        Ok(Self { ca_key, ca_cert, ca_cert_der })
    }

    /// Load an existing CA from DER-encoded cert and PKCS#8 DER-encoded key.
    ///
    /// The original cert DER is preserved for cert chains. The CA cert is
    /// re-derived from the key (same DN) for rcgen's `signed_by` API — the
    /// signature on issued certs is made with the same key, so clients that
    /// trust the original CA cert will accept them.
    pub fn from_der(cert_der: Vec<u8>, key_der: &[u8]) -> Result<Self, rcgen::Error> {
        let key = PrivatePkcs8KeyDer::from(key_der.to_vec());
        let ca_key = KeyPair::from_der_and_sign_algo(
            &PrivateKeyDer::Pkcs8(key),
            &rcgen::PKCS_ECDSA_P256_SHA256,
        )?;
        let params = ca_params();
        let ca_cert = params.self_signed(&ca_key)?;
        let ca_cert_der = CertificateDer::from(cert_der);
        Ok(Self { ca_key, ca_cert, ca_cert_der })
    }

    /// Load a CA from DER files on disk.
    pub fn load(cert_path: &Path, key_path: &Path) -> Result<Self, PkiError> {
        let cert_der = std::fs::read(cert_path)
            .map_err(|e| PkiError::Io(format!("read {}: {e}", cert_path.display())))?;
        let key_der = std::fs::read(key_path)
            .map_err(|e| PkiError::Io(format!("read {}: {e}", key_path.display())))?;
        Self::from_der(cert_der, &key_der).map_err(PkiError::Rcgen)
    }

    /// Write the CA cert and key to DER files on disk.
    pub fn save(&self, cert_path: &Path, key_path: &Path) -> Result<(), PkiError> {
        std::fs::write(cert_path, self.ca_cert_der.as_ref())
            .map_err(|e| PkiError::Io(format!("write {}: {e}", cert_path.display())))?;
        std::fs::write(key_path, self.ca_key.serialize_der())
            .map_err(|e| PkiError::Io(format!("write {}: {e}", key_path.display())))?;
        Ok(())
    }

    /// The DER-encoded CA certificate. Hand this to clients so they can trust
    /// any server cert signed by this CA.
    pub fn ca_cert_der(&self) -> CertificateDer<'static> {
        self.ca_cert_der.clone()
    }

    /// Issue a server certificate with the given SANs, signed by this CA.
    ///
    /// `sans` should include every hostname and IP the server will be reachable
    /// at (e.g. `["localhost", "192.168.1.42"]`). DNS names and IP addresses
    /// are auto-detected from the string format.
    pub fn issue_server_cert(&self, sans: &[&str]) -> Result<ServerCert, rcgen::Error> {
        let server_key = KeyPair::generate()?;
        let san_types: Vec<SanType> = sans
            .iter()
            .map(|s| {
                if let Ok(ip) = s.parse::<IpAddr>() {
                    SanType::IpAddress(ip)
                } else {
                    SanType::DnsName((*s).try_into().unwrap())
                }
            })
            .collect();
        let mut params = CertificateParams::new(Vec::<String>::new())?;
        params.subject_alt_names = san_types;
        params
            .distinguished_name
            .push(DnType::CommonName, "wgpu-remote server");
        let server_cert = params.signed_by(&server_key, &self.ca_cert, &self.ca_key)?;

        Ok(ServerCert {
            cert_chain: vec![server_cert.der().clone(), self.ca_cert_der.clone()],
            private_key: PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
                server_key.serialize_der(),
            )),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PkiError {
    #[error("rcgen: {0}")]
    Rcgen(#[from] rcgen::Error),
    #[error("io: {0}")]
    Io(String),
}
