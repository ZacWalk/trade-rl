use std::path::{Path, PathBuf};

/// Everything that stops a command. All of these are bad input or an
/// environment problem (exit code 2); a failed control or audit is a result,
/// not an error, and is reported through the command's own outcome.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid arguments, settings or data.
    #[error("{0}")]
    Invalid(String),
    /// A file could not be read or written.
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The bar database failed.
    #[error("bar store: {0}")]
    Store(#[from] tradedb::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Shorthand for [`Error::Invalid`].
pub fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// Attach a path to an I/O error.
pub fn io(path: impl AsRef<Path>) -> impl FnOnce(std::io::Error) -> Error {
    let path = path.as_ref().to_path_buf();
    move |source| Error::Io { path, source }
}
