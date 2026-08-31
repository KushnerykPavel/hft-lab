use std::hint::black_box;
use std::sync::Barrier;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
const CACHE_LINE_SIZE: usize = 128;
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
const CACHE_LINE_SIZE: usize = 64;
const DEFAULT_ITERATIONS: u64 = 50_000_000;
const DEFAULT_RUNS: usize = 7;

struct SharedCounters {
    producer: AtomicU64,
    consumer: AtomicU64,
}

#[repr(align(64))]
struct CachePadded<T>(T);

struct PaddedCounters {
    producer: CachePadded<AtomicU64>,
    consumer: CachePadded<AtomicU64>,
}

trait CounterPair: Sync {
    fn producer(&self) -> &AtomicU64;
    fn consumer(&self) -> &AtomicU64;
}

impl CounterPair for SharedCounters {
    fn producer(&self) -> &AtomicU64 {
        &self.producer
    }

    fn consumer(&self) -> &AtomicU64 {
        &self.consumer
    }
}

impl CounterPair for PaddedCounters {
    fn producer(&self) -> &AtomicU64 {
        &self.producer.0
    }

    fn consumer(&self) -> &AtomicU64 {
        &self.consumer.0
    }
}

fn execute<C: CounterPair>(counters: &C, iterations: u64) -> (Duration, u64) {
    let barrier = Barrier::new(3);

    let duration = thread::scope(|scope| {
        let producer = counters.producer();
        let worker_barrier = &barrier;
        scope.spawn(move || {
            worker_barrier.wait(); // Ready.
            worker_barrier.wait(); // Start.

            for _ in 0..iterations {
                producer.fetch_add(1, Ordering::Relaxed);
            }

            worker_barrier.wait(); // Finished.
        });

        let consumer = counters.consumer();
        let worker_barrier = &barrier;
        scope.spawn(move || {
            worker_barrier.wait();
            worker_barrier.wait();

            for _ in 0..iterations {
                consumer.fetch_add(1, Ordering::Relaxed);
            }

            worker_barrier.wait();
        });

        barrier.wait(); // Both workers are ready.
        let started = Instant::now();
        barrier.wait(); // Release both workers together.
        barrier.wait(); // Wait until both loops finish.
        started.elapsed()
    });

    let checksum = counters
        .producer()
        .load(Ordering::Relaxed)
        .wrapping_add(counters.consumer().load(Ordering::Relaxed));
    black_box(counters);
    black_box(checksum);
    (duration, checksum)
}

fn run_shared(iterations: u64) -> (Duration, u64) {
    execute(
        &SharedCounters {
            producer: AtomicU64::new(0),
            consumer: AtomicU64::new(0),
        },
        iterations,
    )
}

fn run_padded(iterations: u64) -> (Duration, u64) {
    execute(
        &PaddedCounters {
            producer: CachePadded(AtomicU64::new(0)),
            consumer: CachePadded(AtomicU64::new(0)),
        },
        iterations,
    )
}

#[derive(Clone, Copy, PartialEq)]
enum Selection {
    Both,
    Shared,
    Padded,
}

struct Config {
    iterations: u64,
    runs: usize,
    selection: Selection,
    warmup: bool,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut config = Self {
            iterations: DEFAULT_ITERATIONS,
            runs: DEFAULT_RUNS,
            selection: Selection::Both,
            warmup: true,
        };
        let mut args = std::env::args().skip(1);

        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--iterations" => {
                    let value = args
                        .next()
                        .ok_or("--iterations requires a positive integer")?;
                    config.iterations = value
                        .parse()
                        .map_err(|_| format!("invalid --iterations value: {value}"))?;
                    if config.iterations == 0 {
                        return Err("--iterations must be greater than zero".into());
                    }
                }
                "--runs" => {
                    let value = args.next().ok_or("--runs requires a positive integer")?;
                    config.runs = value
                        .parse()
                        .map_err(|_| format!("invalid --runs value: {value}"))?;
                    if config.runs == 0 {
                        return Err("--runs must be greater than zero".into());
                    }
                }
                "--layout" => {
                    let value = args
                        .next()
                        .ok_or("--layout requires both, shared, or padded")?;
                    config.selection = match value.as_str() {
                        "both" => Selection::Both,
                        "shared" => Selection::Shared,
                        "padded" => Selection::Padded,
                        _ => return Err(format!("invalid --layout value: {value}")),
                    };
                }
                "--no-warmup" => config.warmup = false,
                "--help" | "-h" => {
                    println!(
                        "Usage: session02_false_sharing [--iterations N] [--runs N] \
                         [--layout both|shared|padded] [--no-warmup]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }

        Ok(config)
    }
}

fn print_run(name: &str, run: usize, duration: Duration, checksum: u64, iterations: u64) {
    let updates = 2.0 * iterations as f64;
    let ns_per_update = duration.as_nanos() as f64 / updates;
    let million_updates_per_second = updates / duration.as_secs_f64() / 1_000_000.0;
    println!(
        "{name:<8} run {run:>2}: total={duration:>10.3?}, {ns_per_update:>8.3} ns/update, \
         {million_updates_per_second:>8.2} M updates/s, checksum={checksum}"
    );
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn print_summary(name: &str, durations: &[Duration], iterations: u64) -> f64 {
    let updates = 2.0 * iterations as f64;
    let samples: Vec<f64> = durations
        .iter()
        .map(|duration| duration.as_nanos() as f64 / updates)
        .collect();
    let minimum = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let sample_median = median(&samples);

    println!(
        "{name:<8} summary: min={minimum:.3}, median={sample_median:.3}, \
         mean={mean:.3}, max={maximum:.3} ns/update"
    );
    sample_median
}

fn main() {
    let config = Config::parse().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!("use --help for usage");
        std::process::exit(2);
    });

    assert_eq!(std::mem::align_of::<SharedCounters>(), CACHE_LINE_SIZE);
    assert_eq!(std::mem::size_of::<SharedCounters>(), CACHE_LINE_SIZE);
    assert_eq!(std::mem::align_of::<PaddedCounters>(), CACHE_LINE_SIZE);
    assert_eq!(std::mem::size_of::<PaddedCounters>(), 2 * CACHE_LINE_SIZE);

    println!(
        "workload: two threads, {} increments per thread, {} total atomic updates",
        config.iterations,
        config.iterations.saturating_mul(2)
    );
    println!(
        "layout: shared={} bytes, padded={} bytes, assumed cache line={} bytes",
        std::mem::size_of::<SharedCounters>(),
        std::mem::size_of::<PaddedCounters>(),
        CACHE_LINE_SIZE
    );

    if config.warmup {
        match config.selection {
            Selection::Both => {
                black_box(run_shared(config.iterations));
                black_box(run_padded(config.iterations));
            }
            Selection::Shared => {
                black_box(run_shared(config.iterations));
            }
            Selection::Padded => {
                black_box(run_padded(config.iterations));
            }
        }
        println!("warm-up complete\n");
    }

    let mut shared_durations = Vec::with_capacity(config.runs);
    let mut padded_durations = Vec::with_capacity(config.runs);
    let expected_checksum = config.iterations.wrapping_mul(2);

    for run in 1..=config.runs {
        let record = |name: &str, result: (Duration, u64), durations: &mut Vec<Duration>| {
            let (duration, checksum) = result;
            assert_eq!(checksum, expected_checksum, "worker missed atomic updates");
            durations.push(duration);
            print_run(name, run, duration, checksum, config.iterations);
        };

        match config.selection {
            Selection::Both if run % 2 == 1 => {
                record(
                    "Shared",
                    run_shared(config.iterations),
                    &mut shared_durations,
                );
                record(
                    "Padded",
                    run_padded(config.iterations),
                    &mut padded_durations,
                );
            }
            Selection::Both => {
                record(
                    "Padded",
                    run_padded(config.iterations),
                    &mut padded_durations,
                );
                record(
                    "Shared",
                    run_shared(config.iterations),
                    &mut shared_durations,
                );
            }
            Selection::Shared => record(
                "Shared",
                run_shared(config.iterations),
                &mut shared_durations,
            ),
            Selection::Padded => record(
                "Padded",
                run_padded(config.iterations),
                &mut padded_durations,
            ),
        }
    }

    println!();
    let shared_median = (!shared_durations.is_empty())
        .then(|| print_summary("Shared", &shared_durations, config.iterations));
    let padded_median = (!padded_durations.is_empty())
        .then(|| print_summary("Padded", &padded_durations, config.iterations));

    if let (Some(shared), Some(padded)) = (shared_median, padded_median) {
        println!(
            "Padded median speedup vs shared cache line: {:.2}x",
            shared / padded
        );
    }
}
