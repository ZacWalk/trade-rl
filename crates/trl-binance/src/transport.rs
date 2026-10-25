//! Where archive bytes come from: HTTPS, or a fixture directory for tests
//! and offline reproduction.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::{Error, Result, invalid};

/// A completed response. Bodies of non-200 responses are not read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

/// GET one URL, reading at most `max_bytes` of body.
pub trait Fetch {
    fn get(&self, url: &str, max_bytes: u64) -> Result<Response>;
}

/// HTTPS over rustls with webpki roots: no redirects, no cookies, no
/// retries, a ten-minute ceiling per request.
pub struct HttpFetch {
    agent: ureq::Agent,
}

impl Default for HttpFetch {
    fn default() -> HttpFetch {
        HttpFetch::new()
    }
}

impl HttpFetch {
    pub fn new() -> HttpFetch {
        let config = ureq::Agent::config_builder()
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(600)))
            .user_agent(concat!("trade-rl/", env!("CARGO_PKG_VERSION")))
            .build();
        HttpFetch {
            agent: ureq::Agent::new_with_config(config),
        }
    }
}

impl Fetch for HttpFetch {
    fn get(&self, url: &str, max_bytes: u64) -> Result<Response> {
        let mut response = self
            .agent
            .get(url)
            .call()
            .map_err(|e| Error::Transport(format!("{url}: {e}")))?;
        let status = response.status().as_u16();
        if status != 200 {
            return Ok(Response {
                status,
                body: Vec::new(),
            });
        }
        let body = response
            .body_mut()
            .with_config()
            .limit(max_bytes)
            .read_to_vec()
            .map_err(|e| match e {
                ureq::Error::BodyExceedsLimit(_) => Error::Limit(format!("{url} exceeds {max_bytes} bytes")),
                other => Error::Transport(format!("{url}: {other}")),
            })?;
        Ok(Response { status, body })
    }
}

/// Serves each URL's last path segment from a directory; absent files are 404.
pub struct FixtureFetch {
    root: PathBuf,
}

impl FixtureFetch {
    pub fn new(root: impl AsRef<Path>) -> Result<FixtureFetch> {
        let root = root.as_ref();
        if root.as_os_str().is_empty() || !root.is_dir() {
            return Err(invalid(format!(
                "--fixtures requires a non-empty directory, not '{}'",
                root.display()
            )));
        }
        Ok(FixtureFetch { root: root.into() })
    }
}

impl Fetch for FixtureFetch {
    fn get(&self, url: &str, max_bytes: u64) -> Result<Response> {
        let name = url.rsplit('/').next().unwrap_or_default();
        if name.is_empty() || name.contains('\\') || name.contains("..") {
            return Err(invalid(format!("unsafe fixture filename in {url}")));
        }
        let path = self.root.join(name);
        let length = match std::fs::metadata(&path) {
            Ok(m) => m.len(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Response {
                    status: 404,
                    body: Vec::new(),
                });
            }
            Err(e) => return Err(crate::io(&path)(e)),
        };
        if length > max_bytes {
            return Err(Error::Limit(format!("{url} exceeds {max_bytes} bytes")));
        }
        let body = std::fs::read(&path).map_err(crate::io(&path))?;
        Ok(Response { status: 200, body })
    }
}
