//! A strict reader for the single-member ZIP archives Binance publishes.
//!
//! The archive must hold exactly one regular, unencrypted member with the
//! expected name, stored or deflated, whose local header starts the file and
//! whose central-directory entry fills the directory. Local and central
//! records must agree, and the decompressed data must match the declared
//! size and CRC-32. Nothing is ever written to disk from here.

use crate::{Error, Result};

const LOCAL: &[u8; 4] = b"PK\x03\x04";
const CENTRAL: &[u8; 4] = b"PK\x01\x02";
const END: &[u8; 4] = b"PK\x05\x06";
const END64_LOCATOR: &[u8; 4] = b"PK\x06\x07";
const END64: &[u8; 4] = b"PK\x06\x06";
const DESCRIPTOR: &[u8; 4] = b"PK\x07\x08";

fn bad(message: &str) -> Error {
    Error::Parse(format!("Binance ZIP: {message}"))
}

struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn slice(&self, at: usize, n: usize) -> Result<&[u8]> {
        at.checked_add(n)
            .and_then(|end| self.0.get(at..end))
            .ok_or_else(|| bad("truncated record"))
    }

    fn u16(&self, at: usize) -> Result<u64> {
        Ok(u64::from(u16::from_le_bytes(self.slice(at, 2)?.try_into().unwrap())))
    }

    fn u32(&self, at: usize) -> Result<u64> {
        Ok(u64::from(u32::from_le_bytes(self.slice(at, 4)?.try_into().unwrap())))
    }

    fn u64(&self, at: usize) -> Result<u64> {
        Ok(u64::from_le_bytes(self.slice(at, 8)?.try_into().unwrap()))
    }
}

fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, entry) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *entry = c;
        }
        t
    });
    !data
        .iter()
        .fold(!0u32, |c, &b| table[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8))
}

/// The ZIP64 extended-information field: values replace the 32-bit fields
/// that are saturated, in the order uncompressed, compressed.
fn zip64_sizes(extra: &[u8], mut uncompressed: u64, mut compressed: u64) -> Result<(u64, u64)> {
    let e = Bytes(extra);
    let mut at = 0;
    while at + 4 <= extra.len() {
        let (id, size) = (e.u16(at)?, e.u16(at + 2)? as usize);
        if id == 1 {
            let mut field = at + 4;
            if uncompressed == 0xFFFF_FFFF {
                uncompressed = e.u64(field)?;
                field += 8;
            }
            if compressed == 0xFFFF_FFFF {
                compressed = e.u64(field)?;
            }
        }
        at += 4 + size;
    }
    if uncompressed == 0xFFFF_FFFF || compressed == 0xFFFF_FFFF {
        return Err(bad("saturated sizes without a ZIP64 field"));
    }
    Ok((uncompressed, compressed))
}

/// Extract the only member of `zip`, which must be named `name`, refusing
/// more than `limit` decompressed bytes.
pub fn extract_single(zip: &[u8], name: &str, limit: u64) -> Result<Vec<u8>> {
    let z = Bytes(zip);
    if zip.len() < 22 || &zip[..4] != LOCAL {
        return Err(bad("incomplete or unsupported envelope"));
    }
    // The end record's comment must run exactly to the end of the file.
    let lowest = zip.len().saturating_sub(65_557);
    let end = (lowest..=zip.len() - 22)
        .rev()
        .find(|&i| &zip[i..i + 4] == END && z.u16(i + 20).is_ok_and(|c| zip.len() - i == 22 + c as usize))
        .ok_or_else(|| bad("no end-of-directory record"))?;
    let entries = z.u16(end + 10)?;
    if z.u16(end + 4)? != 0 || z.u16(end + 6)? != 0 || z.u16(end + 8)? != entries {
        return Err(bad("multi-disk archives are not supported"));
    }
    let (mut directory_size, mut directory_offset) = (z.u32(end + 12)?, z.u32(end + 16)?);
    let mut directory_end = end as u64;
    if end >= 20 && &zip[end - 20..end - 16] == END64_LOCATOR {
        let locator = end - 20;
        let position = z.u64(locator + 8)?;
        if z.u32(locator + 4)? != 0
            || z.u32(locator + 16)? != 1
            || position > locator as u64
            || locator as u64 - position < 56
        {
            return Err(bad("bad ZIP64 locator"));
        }
        let p = position as usize;
        if z.slice(p, 4)? != END64
            || z.u64(p + 4)? != locator as u64 - position - 12
            || z.u32(p + 16)? != 0
            || z.u32(p + 20)? != 0
            || z.u64(p + 24)? != 1
            || z.u64(p + 32)? != 1
            || (entries != 1 && entries != 0xFFFF)
        {
            return Err(bad("bad ZIP64 end record"));
        }
        let (size64, offset64) = (z.u64(p + 40)?, z.u64(p + 48)?);
        if (directory_offset != 0xFFFF_FFFF && directory_offset != offset64)
            || (directory_size != 0xFFFF_FFFF && directory_size != size64)
        {
            return Err(bad("ZIP64 and 32-bit directory records disagree"));
        }
        (directory_size, directory_offset, directory_end) = (size64, offset64, position);
    } else if entries != 1 {
        return Err(bad("the archive must hold exactly one member"));
    }
    if directory_offset > directory_end || directory_size != directory_end - directory_offset {
        return Err(bad("the central directory does not fill its space"));
    }

    let c = directory_offset as usize;
    let name_len = name.len() as u64;
    if directory_size < 46 || c < 30 || z.slice(c, 4)? != CENTRAL {
        return Err(bad("bad central directory"));
    }
    let (central_name, central_extra, central_comment) = (z.u16(c + 28)?, z.u16(c + 30)?, z.u16(c + 32)?);
    if central_name != name_len
        || directory_size != 46 + central_name + central_extra + central_comment
        || z.slice(c + 46, name.len())? != name.as_bytes()
        || z.u16(c + 34)? != 0
        || z.u32(c + 42)? != 0
    {
        return Err(bad("unexpected, unsafe or unsupported member"));
    }
    let flags = z.u16(c + 8)?;
    let method = z.u16(c + 10)?;
    // Encrypted, strongly encrypted or with a masked local header.
    if flags & 0x2041 != 0 || (method != 0 && method != 8) {
        return Err(bad("encrypted or unsupported compression"));
    }
    let host = zip[c + 5];
    let external = z.u32(c + 38)?;
    let mode = (external >> 16) & 0o170_000;
    if ((host == 3 || host == 19) && mode != 0 && mode != 0o100_000) || external & 0x10 != 0 {
        return Err(bad("the member is not a regular file"));
    }
    let crc = z.u32(c + 16)?;
    let extra = z.slice(c + 46 + central_name as usize, central_extra as usize)?;
    let (uncompressed, compressed) = zip64_sizes(extra, z.u32(c + 24)?, z.u32(c + 20)?)?;

    // The local header must agree with the central directory.
    let local_name = z.u16(26)?;
    let local_extra = z.u16(28)?;
    if local_name != name_len
        || z.slice(30, name.len())? != name.as_bytes()
        || z.u16(6)? != flags
        || z.u16(8)? != method
        || directory_offset < 30 + local_name + local_extra
    {
        return Err(bad("local and central records disagree"));
    }
    let start = 30 + local_name + local_extra;
    let data_end = start
        .checked_add(compressed)
        .filter(|&e| e <= directory_offset)
        .ok_or_else(|| bad("member data overruns the directory"))?;
    if flags & 8 == 0 {
        let local_extra_bytes = z.slice(30 + local_name as usize, local_extra as usize)?;
        let local = zip64_sizes(local_extra_bytes, z.u32(22)?, z.u32(18)?)?;
        if z.u32(14)? != crc || local != (uncompressed, compressed) || data_end != directory_offset {
            return Err(bad("local and central sizes or CRC disagree"));
        }
    } else {
        // A data descriptor follows the data: optional signature, CRC, then
        // 32- or 64-bit sizes. Exactly one form must reach the directory.
        let d = data_end as usize;
        let signed = z.slice(d, 4).is_ok_and(|s| s == DESCRIPTOR);
        let at = if signed { d + 4 } else { d };
        let matches = |wide: bool| -> bool {
            let sizes = if wide {
                (z.u64(at + 4), z.u64(at + 12))
            } else {
                (z.u32(at + 4), z.u32(at + 8))
            };
            let end = at + if wide { 20 } else { 12 };
            matches!(sizes, (Ok(cs), Ok(us)) if cs == compressed && us == uncompressed)
                && z.u32(at).is_ok_and(|v| v == crc)
                && end as u64 == directory_offset
        };
        if !matches(false) && !matches(true) {
            return Err(bad("the data descriptor disagrees with the directory"));
        }
    }

    if uncompressed > limit {
        return Err(Error::Limit(
            "Binance ZIP member exceeds the decompression limit".into(),
        ));
    }
    let data = &zip[start as usize..data_end as usize];
    let out = if method == 0 {
        if compressed != uncompressed {
            return Err(bad("stored sizes disagree"));
        }
        data.to_vec()
    } else {
        miniz_oxide::inflate::decompress_to_vec_with_limit(data, uncompressed as usize)
            .map_err(|_| bad("corrupt deflate data or size"))?
    };
    if out.len() as u64 != uncompressed || crc32(&out) != crc as u32 {
        return Err(bad("corrupt data, size or CRC"));
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A minimal ZIP writer for tests: one member, stored or deflated.
    pub fn build(name: &str, data: &[u8], deflate: bool) -> Vec<u8> {
        let crc = crc32(data);
        let body = if deflate {
            miniz_oxide::deflate::compress_to_vec(data, 6)
        } else {
            data.to_vec()
        };
        let method: u16 = if deflate { 8 } else { 0 };
        let mut out = Vec::new();
        out.extend_from_slice(LOCAL);
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0x21, 0]);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&body);
        let central = out.len();
        out.extend_from_slice(CENTRAL);
        out.extend_from_slice(&[20, 3]);
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0x21, 0]);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
        out.extend_from_slice(&(0o100_644u32 << 16).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        let size = out.len() - central;
        out.extend_from_slice(END);
        out.extend_from_slice(&[0, 0, 0, 0, 1, 0, 1, 0]);
        out.extend_from_slice(&(size as u32).to_le_bytes());
        out.extend_from_slice(&(central as u32).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn crc_reference() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn stored_and_deflated_round_trip() {
        let data = b"id,price,qty,quote_qty,time,is_buyer_maker\n1,2,3,4,5,true\n".repeat(50);
        for deflate in [false, true] {
            let zip = build("a.csv", &data, deflate);
            assert_eq!(extract_single(&zip, "a.csv", 1 << 20).unwrap(), data);
            assert!(extract_single(&zip, "b.csv", 1 << 20).is_err());
            assert!(matches!(extract_single(&zip, "a.csv", 10), Err(Error::Limit(_))));
        }
    }

    #[test]
    fn tampering_is_detected() {
        let data = b"1,2,3,4,5,true\n".repeat(20);
        let zip = build("a.csv", &data, false);
        let mut flipped = zip.clone();
        flipped[40] ^= 1;
        assert!(extract_single(&flipped, "a.csv", 1 << 20).is_err());
        let mut encrypted = zip.clone();
        encrypted[6] |= 1;
        assert!(extract_single(&encrypted, "a.csv", 1 << 20).is_err());
        let mut truncated = zip.clone();
        truncated.truncate(zip.len() - 1);
        assert!(extract_single(&truncated, "a.csv", 1 << 20).is_err());
        let traversal = build("../escape.csv", &data, false);
        assert!(extract_single(&traversal, "escape.csv", 1 << 20).is_err());
        assert!(extract_single(b"not a zip at all, not a zip", "a.csv", 1 << 20).is_err());
    }
}
