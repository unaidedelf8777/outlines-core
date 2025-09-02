//! Minimal pprof-based profiling CLI for Index::new.
//!
//! Build:  cargo build --release
//! Run:    cargo run --release --bin oc-index-prof -- \
//!           --iters 300 \
//!           --regex 'A: [\\w \\.*\\-=\\+,\\?/]{10,50}\\. The answer is [1-9][0-9]{0,9}\\.' \
//!           --out ./oc_index_flame.svg --freq 1000
//!
//! Tip: For crisper stacks, also set: RUSTFLAGS="-Ctarget-cpu=native"

use std::env;
use std::fs::File;
use std::hint::black_box;
use std::time::{Duration, Instant};

use anyhow::Result;
use pprof::{flamegraph, ProfilerGuardBuilder};

// ----- import your library types -----
// Adjust this line to your crate name/module paths if different:
use outlines_core::{index::Index, vocabulary::Vocabulary};

fn parse_arg(flag: &str, default: &str) -> String {
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        if a == flag {
            return it.next().unwrap_or_else(|| default.to_string());
        }
    }
    default.to_string()
}
fn parse_u(flag: &str, default: usize) -> usize {
    parse_arg(flag, &default.to_string()).parse().unwrap_or(default)
}

fn main() -> Result<()> {
    // --------- args ----------
    let regex = parse_arg(
        "--regex",
        r#"A: [\w \.\*\-=\+,\?/]{10,50}\. The answer is [1-9][0-9]{0,9}\."#,
    );
    let iters  = parse_u("--iters", 300);
    let warmup = parse_u("--warmup", 0);
    let freq   = parse_u("--freq", 10000); // Hz
    let out    = parse_arg("--out", "./oc_index_flame.svg");

    // --------- vocabulary ----------
    // Adjust to your actual API if needed.
    let vocab = Vocabulary::from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct", None).unwrap_or_else(|e| {
        panic!("from_pretrained(\"gpt2\") failed: {e}");
    });

    // --------- start profiler ----------
    // pprof uses timer-based sampling; no perf required.
    let guard = ProfilerGuardBuilder::default()
        .frequency(freq as i32)
        // optional: trim noise from the graph
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()?;

    // --------- warmup ----------
    for _ in 0..warmup {
        let idx = Index::new(&regex, &vocab).expect("index build");
        black_box(idx);
    }

    // --------- timed loop ----------
    let mut total = Duration::ZERO;
    let mut states_sum = 0usize;

    for _ in 0..iters {
        let t0 = Instant::now();
        let idx = Index::new(&regex, &vocab).expect("index build");
        total += t0.elapsed();
        println!("{:?}", idx.allowed_tokens(&0));
        black_box(&idx);
    }

    let avg = total.as_secs_f64() * 1e3 / (iters as f64);
    eprintln!(
        "avg build time: {:.3} ms over {iters} iters (sum states: {})",
        avg, states_sum
    );

    // --------- write flamegraph ----------
    if let Ok(report) = guard.report().build() {
        let mut file = File::create(&out)?;
        report.flamegraph(&mut file)?;
        eprintln!("flamegraph written to {out}");
    } else {
        eprintln!("pprof: no samples collected (try increasing --iters or --freq).");
    }

    Ok(())
}
