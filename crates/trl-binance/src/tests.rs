//! End-to-end acquisition against fixture archives: no network.

use super::*;
use crate::zip::tests::build;
use sha2::{Digest, Sha256};

fn sidecar(zip: &[u8], name: &str) -> String {
    let digest: String = Sha256::digest(zip).iter().map(|b| format!("{b:02x}")).collect();
    format!("{digest}  {name}\n")
}

/// Write `csv` as the fixture archive (and sidecar) for `spec`.
fn fixture(dir: &Path, spec: &ArchiveSpec, csv: &str) -> Names {
    let names = spec.names().unwrap();
    let zip = build(&names.csv_name, csv.as_bytes(), true);
    std::fs::write(dir.join(&names.zip_name), &zip).unwrap();
    std::fs::write(
        dir.join(format!("{}.CHECKSUM", names.zip_name)),
        sidecar(&zip, &names.zip_name),
    )
    .unwrap();
    names
}

fn trades(date: &str) -> ArchiveSpec {
    ArchiveSpec {
        market: Market::Um,
        kind: Kind::Trades,
        period: Period::Daily,
        symbol: "ETHUSDT".into(),
        interval: String::new(),
        date: date.into(),
    }
}

const TRADES: &str = "id,price,qty,quote_qty,time,is_buyer_maker\n\
    1,3500.5,0.2,700.1,1721606400123,true\n\
    2,3501.0,0.1,350.1,1721606401123,false\n";

fn config(out: &Path) -> DownloadConfig {
    DownloadConfig {
        symbol: "ETHUSDT".into(),
        market: Market::Um,
        kind: Kind::Trades,
        period: None,
        interval: "1m".into(),
        from: "2024-07-22".into(),
        to: "2024-07-22".into(),
        out: out.into(),
        force: false,
        skip_missing: false,
        limits: Limits::default(),
    }
}

#[test]
fn names_follow_binance_layout() {
    let n = trades("2024-07-22").names().unwrap();
    assert_eq!(
        n.zip_url,
        "https://data.binance.vision/data/futures/um/daily/trades/ETHUSDT/ETHUSDT-trades-2024-07-22.zip"
    );
    assert_eq!(n.checksum_url, format!("{}.CHECKSUM", n.zip_url));
    let klines = ArchiveSpec {
        market: Market::Spot,
        kind: Kind::Klines,
        period: Period::Monthly,
        symbol: "ETHUSDT".into(),
        interval: "1m".into(),
        date: "2024-07".into(),
    };
    assert_eq!(
        klines.names().unwrap().zip_url,
        "https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-2024-07.zip"
    );
    for bad in [
        ArchiveSpec {
            symbol: "ethusdt".into(),
            ..trades("2024-07-22")
        },
        ArchiveSpec {
            symbol: "_ETH".into(),
            ..trades("2024-07-22")
        },
        ArchiveSpec {
            interval: "1m".into(),
            ..trades("2024-07-22")
        },
        ArchiveSpec {
            market: Market::Um,
            interval: "1s".into(),
            ..klines.clone()
        },
        trades("2024-02-30"),
        trades("2024-07"),
    ] {
        assert!(bad.names().is_err(), "{bad:?}");
    }
}

#[test]
fn checksum_format_is_strict() {
    let zip = b"zip bytes";
    let good = sidecar(zip, "a.zip");
    verify_checksum(zip, good.as_bytes(), "a.zip").unwrap();
    verify_checksum(zip, good.replace("  ", " *").as_bytes(), "a.zip").unwrap();
    verify_checksum(zip, good.replace('\n', "\r\n").as_bytes(), "a.zip").unwrap();
    assert!(verify_checksum(zip, good.as_bytes(), "b.zip").is_err());
    assert!(verify_checksum(b"other", good.as_bytes(), "a.zip").is_err());
    assert!(verify_checksum(zip, good.to_uppercase().as_bytes(), "a.zip").is_err());
}

#[test]
fn window_dates() {
    let mut c = config(Path::new("x"));
    c.from = "2024-01-30".into();
    c.to = "2024-03-02".into();
    assert_eq!(c.dates().unwrap().len(), 33);
    c.kind = Kind::Klines;
    assert_eq!(c.dates().unwrap(), vec!["2024-01", "2024-02", "2024-03"]);
    c.to = "2023-12-31".into();
    assert!(c.dates().is_err());
    c.kind = Kind::Trades;
    c.from = "2000-01-01".into();
    c.to = "2020-01-01".into();
    assert!(matches!(c.dates(), Err(Error::Limit(_))));
    assert_eq!(config(Path::new("")).output(), Path::new("data").join("raw"));
}

#[test]
fn download_publish_skip_and_force() {
    let fixtures = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let names = fixture(fixtures.path(), &trades("2024-07-22"), TRADES);
    let client = FixtureFetch::new(fixtures.path()).unwrap();
    let mut log = Vec::new();
    let summary = download_window(&config(out.path()), &client, &mut log).unwrap();
    assert_eq!(summary.downloaded, 1);
    let published = out.path().join(&names.csv_name);
    assert_eq!(std::fs::read_to_string(&published).unwrap(), TRADES);
    assert!(String::from_utf8(log).unwrap().contains("2 rows, SHA256 verified"));

    // Existing data is kept, not re-fetched.
    let again = download_window(&config(out.path()), &client, &mut Vec::new()).unwrap();
    assert_eq!((again.downloaded, again.skipped), (0, 1));

    // --force replaces and keeps the old file as .previous.
    let changed = TRADES.replace("3501.0", "3502.0");
    fixture(fixtures.path(), &trades("2024-07-22"), &changed);
    let mut forced = config(out.path());
    forced.force = true;
    download_window(&forced, &client, &mut Vec::new()).unwrap();
    assert_eq!(std::fs::read_to_string(&published).unwrap(), changed);
    let previous = out.path().join(format!("{}.previous", names.csv_name));
    assert_eq!(std::fs::read_to_string(&previous).unwrap(), TRADES);
    assert!(download_window(&forced, &client, &mut Vec::new()).is_err());
    assert_eq!(std::fs::read_to_string(&previous).unwrap(), TRADES);
}

#[test]
fn verification_failures_never_publish() {
    let fixtures = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let names = fixture(fixtures.path(), &trades("2024-07-22"), TRADES);
    let client = FixtureFetch::new(fixtures.path()).unwrap();
    let destination = out.path().join(&names.csv_name);

    // A wrong digest.
    let checksum = fixtures.path().join(format!("{}.CHECKSUM", names.zip_name));
    let good = std::fs::read_to_string(&checksum).unwrap();
    std::fs::write(
        &checksum,
        good.replacen(&good[..1], if &good[..1] == "0" { "1" } else { "0" }, 1),
    )
    .unwrap();
    assert!(matches!(
        download_window(&config(out.path()), &client, &mut Vec::new()),
        Err(Error::Checksum(_))
    ));
    assert!(!destination.exists());

    // A missing sidecar is a checksum failure even when skipping missing archives.
    std::fs::remove_file(&checksum).unwrap();
    let mut skip = config(out.path());
    skip.skip_missing = true;
    assert!(matches!(
        download_window(&skip, &client, &mut Vec::new()),
        Err(Error::Checksum(_))
    ));
    assert!(!destination.exists());

    // A missing archive can be skipped, but then nothing was fetched.
    std::fs::remove_file(fixtures.path().join(&names.zip_name)).unwrap();
    let mut log = Vec::new();
    let summary = download_window(&skip, &client, &mut log).unwrap();
    assert_eq!(summary.missing, 1);
    assert!(String::from_utf8(log).unwrap().contains("unavailable (404)"));
    assert!(matches!(
        download_window(&config(out.path()), &client, &mut Vec::new()),
        Err(Error::NotFound(_))
    ));

    // Too many rows fails before publication.
    fixture(fixtures.path(), &trades("2024-07-22"), TRADES);
    let mut tight = config(out.path());
    tight.limits.max_rows = 1;
    assert!(matches!(
        download_window(&tight, &client, &mut Vec::new()),
        Err(Error::Limit(_))
    ));
    assert!(!destination.exists());
}

#[test]
fn raw_kinds_are_stored_unchanged() {
    let fixtures = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let spec = ArchiveSpec {
        market: Market::Um,
        kind: Kind::FundingRate,
        period: Period::Monthly,
        symbol: "ETHUSDT".into(),
        interval: String::new(),
        date: "2024-07".into(),
    };
    let raw = "calc_time,funding_interval_hours,last_funding_rate\n1719792000000,8,0.0001\n";
    let names = fixture(fixtures.path(), &spec, raw);
    let mut c = config(out.path());
    c.kind = Kind::FundingRate;
    let mut log = Vec::new();
    download_window(&c, &FixtureFetch::new(fixtures.path()).unwrap(), &mut log).unwrap();
    assert_eq!(std::fs::read_to_string(out.path().join(names.csv_name)).unwrap(), raw);
    assert!(String::from_utf8(log).unwrap().contains("raw CSV bytes"));
}

#[test]
fn fixture_names_are_contained() {
    let fixtures = tempfile::tempdir().unwrap();
    let client = FixtureFetch::new(fixtures.path()).unwrap();
    assert!(client.get("https://x/..", 10).is_err());
    assert!(client.get("https://x/a\\..\\secret", 10).is_err());
    assert!(client.get("https://x/", 10).is_err());
    assert_eq!(client.get("https://x/absent.zip", 10).unwrap().status, 404);
    assert!(FixtureFetch::new("").is_err());
}
