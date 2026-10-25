//! End-to-end checks of the `trl` binary on synthetic data and fixture
//! archives. No test touches the network.

use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

struct Out {
    code: i32,
    text: String,
}

fn trl(args: &[&str]) -> Out {
    let output = Command::new(env!("CARGO_BIN_EXE_trl"))
        .args(args)
        .output()
        .expect("trl runs");
    Out {
        code: output.status.code().unwrap_or(-1),
        text: format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    }
}

fn expect(args: &[&str], codes: &[i32]) -> String {
    let out = trl(args);
    assert!(
        codes.contains(&out.code),
        "trl {}: exit {}\n{}",
        args.join(" "),
        out.code,
        out.text
    );
    out.text
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn only_run(root: &Path) -> String {
    let mut runs: Vec<String> = std::fs::read_dir(root)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with('.') && n != "verify")
        .collect();
    runs.sort();
    assert_eq!(runs.len(), 1, "{runs:?}");
    runs.remove(0)
}

#[test]
fn usage_and_exit_codes() {
    assert_eq!(trl(&[]).code, 2);
    assert_eq!(trl(&["--help"]).code, 0);
    assert_eq!(trl(&["walk", "--agnet", "breakout"]).code, 2);
    assert_eq!(trl(&["walk", "--seeds", "-1"]).code, 2);
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("none.tradedb");
    let out = trl(&["audit", "--db", s(&db)]);
    assert_eq!(out.code, 2);
    assert!(out.text.contains("no bar database"), "{}", out.text);
}

#[test]
fn binance_walk_reproduces_and_detects_tampering() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("bars.tradedb");
    let runs = dir.path().join("runs");
    expect(&["synth", "--db", s(&db), "--kind", "trend", "--bars", "16000"], &[0]);
    let audit = expect(&["audit", "--db", s(&db)], &[0]);
    assert!(
        audit.contains("20 features") && audit.contains("shift audit passed"),
        "{audit}"
    );

    let walk = expect(
        &[
            "walk",
            "--db",
            s(&db),
            "--agent",
            "breakout",
            "--seeds",
            "2",
            "--train",
            "5000",
            "--valid",
            "500",
            "--gap",
            "300",
            "--test",
            "2000",
            "--episode",
            "1000",
            "--epochs",
            "1",
            "--seed",
            "7",
            "--out",
            s(&runs),
        ],
        &[0, 1],
    );
    assert!(
        walk.contains("control:rotated") && walk.contains("Validation window"),
        "{walk}"
    );
    let id = only_run(&runs);

    let verify = expect(&["verify", "--run", &id, "--root", s(&runs)], &[0, 1]);
    assert!(
        verify.contains("byte-identical") && !verify.contains("DIFFERS"),
        "{verify}"
    );

    let report = expect(&["report", "--run", &id, "--root", s(&runs)], &[0]);
    assert!(
        report.contains("pooled results") && report.contains("control:ensemble-rotated"),
        "{report}"
    );
    let list = expect(&["list", "--root", s(&runs)], &[0]);
    assert!(list.contains(&id) && list.contains("agent=breakout"), "{list}");

    // A same-size edit of a recorded number must be caught.
    let metrics = runs.join(&id).join("metrics.csv");
    let original = std::fs::read(&metrics).unwrap();
    let mut altered = original.clone();
    let digit = altered.iter().rposition(u8::is_ascii_digit).unwrap();
    altered[digit] = if altered[digit] == b'0' { b'1' } else { b'0' };
    std::fs::write(&metrics, &altered).unwrap();
    let differs = expect(&["verify", "--run", &id, "--root", s(&runs)], &[1]);
    assert!(
        differs.contains("metrics.csv  DIFFERS") && differs.contains("NOT REPRODUCED"),
        "{differs}"
    );
    std::fs::write(&metrics, &original).unwrap();

    // A different feature contract is refused, not re-run.
    let manifest = runs.join(&id).join("manifest.txt");
    let text = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(
        &manifest,
        text.replace("feature_profile=binance-v1", "feature_profile=unknown"),
    )
    .unwrap();
    let mismatch = expect(&["verify", "--run", &id, "--root", s(&runs)], &[2]);
    assert!(mismatch.contains("mismatch"), "{mismatch}");
}

#[test]
fn ohlcv_import_audit_patterns_and_guards() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("bars.tradedb");
    let csv = dir.path().join("prices.csv");
    std::fs::write(&csv, trl_core::synthetic::session_csv(7000)).unwrap();
    let import = [
        "bars",
        "--db",
        s(&db),
        "--format",
        "ohlcv",
        "--in",
        s(&csv),
        "--symbol",
        "TEST",
        "--interval",
        "15m",
        "--calendar",
        "fixture-sessions",
        "--currency",
        "USD",
        "--source-name",
        "synthetic",
    ];
    let text = expect(&import, &[0]);
    assert!(text.contains("7000 bars") && text.contains("ohlcv-bars-v1"), "{text}");
    let again = expect(&import, &[2]);
    assert!(again.contains("refusing to overwrite"), "{again}");

    let audit = expect(&["audit", "--db", s(&db)], &[0]);
    assert!(
        audit.contains("11 features") && audit.contains("shift audit passed"),
        "{audit}"
    );
    let patterns = expect(&["patterns", "--db", s(&db), "--shuffles", "20", "--block", "26"], &[0]);
    assert!(
        patterns.contains("UNAVAILABLE") && patterns.contains("previous session"),
        "{patterns}"
    );
    let unblocked = expect(&["patterns", "--db", s(&db)], &[2]);
    assert!(unblocked.contains("--block"), "{unblocked}");

    let runs = dir.path().join("runs");
    let refused = expect(
        &[
            "walk",
            "--db",
            s(&db),
            "--agent",
            "flat",
            "--cost-activity",
            "1",
            "--out",
            s(&runs),
        ],
        &[2],
    );
    assert!(refused.contains("trade count") && !runs.exists(), "{refused}");

    // Exported OHLCV re-imports to the same bars.
    let exported = dir.path().join("export.csv");
    expect(
        &["series", "export", "--db", s(&db), "TEST/15m", "--out", s(&exported)],
        &[0],
    );
    let again = [
        "bars",
        "--db",
        s(&db),
        "--format",
        "ohlcv",
        "--in",
        s(&exported),
        "--symbol",
        "TEST",
        "--interval",
        "15m",
        "--calendar",
        "fixture-sessions",
        "--currency",
        "USD",
        "--source-name",
        "synthetic",
        "--series",
        "TEST/copy",
    ];
    expect(&again, &[0]);
    let list = expect(&["series", "list", "--db", s(&db)], &[0]);
    let hashes: Vec<&str> = list.lines().filter_map(|l| l.split("hash ").nth(1)).collect();
    assert_eq!(hashes.len(), 2, "{list}");
    assert_eq!(hashes[0], hashes[1], "{list}");

    let bad = dir.path().join("bad.csv");
    for rows in [
        "time,open,high,low,close,session\n1577975400000,100,101,99,NaN,1\n",
        "time,open,high,low,close,session\n1577975400000,100,99,101,100,1\n",
        "time,open,high,low,close\n1577975400000,100,101,99,100\n",
    ] {
        std::fs::write(&bad, rows).unwrap();
        let out = trl(&[
            "bars",
            "--db",
            s(&db),
            "--format",
            "ohlcv",
            "--in",
            s(&bad),
            "--symbol",
            "BAD",
            "--interval",
            "15m",
            "--calendar",
            "fixture-sessions",
            "--currency",
            "USD",
        ]);
        assert_eq!(out.code, 2, "{rows}\n{}", out.text);
    }
    let list = expect(&["series", "list", "--db", s(&db)], &[0]);
    assert!(!list.contains("BAD"), "{list}");
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

/// A one-member stored ZIP, as Binance publishes them (minus compression).
fn stored_zip(name: &str, data: &[u8]) -> Vec<u8> {
    let crc = crc32(data).to_le_bytes();
    let (n, len) = ((name.len() as u16).to_le_bytes(), (data.len() as u32).to_le_bytes());
    let mut z = Vec::new();
    z.extend_from_slice(b"PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x21\x00");
    z.extend_from_slice(&crc);
    z.extend_from_slice(&len);
    z.extend_from_slice(&len);
    z.extend_from_slice(&n);
    z.extend_from_slice(&[0, 0]);
    z.extend_from_slice(name.as_bytes());
    z.extend_from_slice(data);
    let central = z.len() as u32;
    z.extend_from_slice(b"PK\x01\x02\x14\x03\x14\x00\x00\x00\x00\x00\x00\x00\x21\x00");
    z.extend_from_slice(&crc);
    z.extend_from_slice(&len);
    z.extend_from_slice(&len);
    z.extend_from_slice(&n);
    z.extend_from_slice(&[0; 8]);
    z.extend_from_slice(&(0o100_644u32 << 16).to_le_bytes());
    z.extend_from_slice(&[0; 4]);
    z.extend_from_slice(name.as_bytes());
    let size = z.len() as u32 - central;
    z.extend_from_slice(b"PK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x00");
    z.extend_from_slice(&size.to_le_bytes());
    z.extend_from_slice(&central.to_le_bytes());
    z.extend_from_slice(&[0, 0]);
    z
}

fn publish_fixture(dir: &Path, stem: &str, csv: &str) {
    let zip = stored_zip(&format!("{stem}.csv"), csv.as_bytes());
    let digest: String = Sha256::digest(&zip).iter().map(|b| format!("{b:02x}")).collect();
    std::fs::write(dir.join(format!("{stem}.zip")), &zip).unwrap();
    std::fs::write(
        dir.join(format!("{stem}.zip.CHECKSUM")),
        format!("{digest}  {stem}.zip\n"),
    )
    .unwrap();
}

#[test]
fn fixture_download_then_import() {
    let dir = tempfile::tempdir().unwrap();
    let fixtures = dir.path().join("fixtures");
    std::fs::create_dir(&fixtures).unwrap();
    let mut csv = String::from("id,price,qty,quote_qty,time,is_buyer_maker\n");
    for i in 0..30u64 {
        let price = 3500.0 + (i % 7) as f64;
        csv += &format!(
            "{},{price},0.5,{},{},{}\n",
            i + 1,
            price * 0.5,
            1_721_606_400_000 + i * 1700,
            i % 3 == 0
        );
    }
    publish_fixture(&fixtures, "ETHUSDT-trades-2024-07-22", &csv);
    let raw: PathBuf = dir.path().join("raw");
    let args = [
        "download",
        "--symbol",
        "ETHUSDT",
        "--kind",
        "trades",
        "--from",
        "2024-07-22",
        "--to",
        "2024-07-22",
        "--fixtures",
        s(&fixtures),
        "--out",
        s(&raw),
    ];
    let text = expect(&args, &[0]);
    assert!(text.contains("30 rows, SHA256 verified"), "{text}");
    assert_eq!(
        std::fs::read_to_string(raw.join("ETHUSDT-trades-2024-07-22.csv")).unwrap(),
        csv
    );
    let skipped = expect(&args, &[0]);
    assert!(skipped.contains("already present"), "{skipped}");

    let missing = trl(&[
        "download",
        "--from",
        "2024-07-23",
        "--to",
        "2024-07-23",
        "--fixtures",
        s(&fixtures),
        "--out",
        s(&raw),
    ]);
    assert_eq!(missing.code, 2, "{}", missing.text);
    let empty = trl(&[
        "download",
        "--from",
        "2024-07-22",
        "--to",
        "2024-07-22",
        "--fixtures",
        "",
    ]);
    assert_eq!(empty.code, 2);
    assert!(empty.text.contains("--fixtures"), "{}", empty.text);

    let db = dir.path().join("bars.tradedb");
    let bars = expect(&["bars", "--db", s(&db), "--in", s(&raw), "--interval", "10s"], &[0]);
    assert!(
        bars.contains("30 trades -> ") && bars.contains("stored ETHUSDT/10s"),
        "{bars}"
    );
    let refused = expect(&["bars", "--db", s(&db), "--in", s(&raw), "--interval", "10x"], &[2]);
    assert!(refused.contains("bad --interval '10x'"), "{refused}");
}

#[test]
fn series_show_and_remove() {
    let dir = tempfile::tempdir().unwrap();
    let series = trl_core::synthetic::trend_series(500);
    let db = dir.path().join("bars.tradedb");
    let text = expect(&["synth", "--db", s(&db), "--kind", "trend", "--bars", "500"], &[0]);
    assert!(text.contains("stored synthetic/trend"), "{text}");
    assert!(text.contains(&format!("hash {:016x}", series.hash())), "{text}");
    let show = expect(&["series", "show", "--db", s(&db), "synthetic/trend"], &[0]);
    assert!(show.contains("binance") && show.contains("500"), "{show}");
    expect(&["series", "remove", "--db", s(&db), "synthetic/trend"], &[0]);
    expect(&["series", "remove", "--db", s(&db), "synthetic/trend"], &[2]);
}
