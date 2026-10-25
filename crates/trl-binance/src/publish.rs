//! Publishing a downloaded file under its final name.
//!
//! The bytes are written and flushed to `<name>.part`, created exclusively,
//! then hard-linked into place; a hard link never replaces an existing name,
//! so a concurrent writer or a file that appeared meanwhile is refused rather
//! than overwritten. With `force` the old file is first moved aside to
//! `<name>.previous`, which is itself never overwritten.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{Result, invalid, io};

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

/// Move without ever replacing the destination.
fn move_no_replace(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::hard_link(from, to)?;
    fs::remove_file(from)
}

pub fn publish(destination: &Path, bytes: &[u8], force: bool) -> Result<()> {
    if destination.file_name().is_none() {
        return Err(invalid(format!("{} is not a file path", destination.display())));
    }
    let stage = sibling(destination, ".part");
    let previous = sibling(destination, ".previous");
    let existing = fs::symlink_metadata(destination).ok();
    if let Some(meta) = &existing {
        if !force {
            return Err(invalid(format!(
                "{} appeared during download; refusing to replace it",
                destination.display()
            )));
        }
        if !meta.file_type().is_file() {
            return Err(invalid(format!(
                "refusing to replace {}, which is not a regular file",
                destination.display()
            )));
        }
    }
    if fs::symlink_metadata(&previous).is_ok() {
        return Err(invalid(format!(
            "preserving existing {}; inspect it before retrying",
            previous.display()
        )));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stage)
        .map_err(|e| {
            invalid(format!(
                "cannot create {} exclusively ({e}); another download may be running, or remove a stale file",
                stage.display()
            ))
        })?;
    let written = file.write_all(bytes).and_then(|_| file.sync_all());
    drop(file);
    if let Err(e) = written {
        let _ = fs::remove_file(&stage);
        return Err(io(&stage)(e));
    }
    if existing.is_some() {
        move_no_replace(destination, &previous).map_err(|e| {
            invalid(format!(
                "cannot move {} aside ({e}); staged data retained at {}",
                destination.display(),
                stage.display()
            ))
        })?;
    }
    move_no_replace(&stage, destination).map_err(|e| {
        let kept = if existing.is_some() {
            format!("; original retained at {}", previous.display())
        } else {
            String::new()
        };
        invalid(format!(
            "cannot publish {} ({e}); staged data retained at {}{kept}",
            destination.display(),
            stage.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_keeps_one_previous() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.csv");
        publish(&path, b"one", false).unwrap();
        assert!(publish(&path, b"two", false).is_err());
        publish(&path, b"two", true).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"two");
        assert_eq!(fs::read(sibling(&path, ".previous")).unwrap(), b"one");
        // An existing backup is never overwritten.
        assert!(publish(&path, b"three", true).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"two");
        assert!(!sibling(&path, ".part").exists());
    }

    #[test]
    fn stale_stage_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.csv");
        fs::write(sibling(&path, ".part"), b"stale").unwrap();
        assert!(publish(&path, b"new", false).is_err());
        assert!(!path.exists());
    }
}
