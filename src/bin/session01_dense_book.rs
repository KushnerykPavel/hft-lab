use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

const MIN_PRICE: u64 = 95_000;
const MAX_PRICE: u64 = 105_000;
const PRICE_COUNT: u64 = MAX_PRICE - MIN_PRICE;
const OPERATION_COUNT: usize = 1_000_000;
const DEFAULT_RUNS: usize = 7;

#[derive(Debug, Clone, Default)]
struct PriceLevel {
    // A zero quantity represents an absent level in DenseBook.
    qty: u64,
}

#[derive(Default)]
struct TreeBook {
    levels: BTreeMap<u64, PriceLevel>,
}

impl TreeBook {
    #[inline]
    fn update(&mut self, price: u64, qty: u64) {
        if qty == 0 {
            self.levels.remove(&price);
        } else {
            self.levels.insert(price, PriceLevel { qty });
        }
    }

    #[inline]
    fn get(&self, price: u64) -> Option<&PriceLevel> {
        self.levels.get(&price)
    }
}

struct DenseBook {
    base: u64,
    levels: Vec<PriceLevel>,
}

impl DenseBook {
    fn new(base: u64) -> Self {
        Self {
            base,
            levels: Vec::new(),
        }
    }

    #[inline]
    fn update(&mut self, price: u64, qty: u64) {
        let Some(index) = price
            .checked_sub(self.base)
            .and_then(|offset| usize::try_from(offset).ok())
        else {
            return;
        };

        if index >= self.levels.len() {
            self.levels.resize(index + 1, PriceLevel::default());
        }
        self.levels[index].qty = qty;
    }

    #[inline]
    fn get(&self, price: u64) -> Option<&PriceLevel> {
        let index = usize::try_from(price.checked_sub(self.base)?).ok()?;
        self.levels.get(index).filter(|level| level.qty != 0)
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Lookup { price: u64 },
    Update { price: u64, qty: u64 },
}

/// Small deterministic generator so the workload has no external dependency.
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }
}

fn generate_workload() -> Vec<Operation> {
    let mut rng = XorShift64(0x4d59_5df4_d0f3_3173);
    let mut operations = Vec::with_capacity(OPERATION_COUNT);

    for index in 0..OPERATION_COUNT {
        let price = MIN_PRICE + rng.next() % PRICE_COUNT;

        // Exactly seven lookups and three updates in every ten operations.
        if index % 10 < 7 {
            operations.push(Operation::Lookup { price });
        } else {
            let qty = 1 + rng.next() % 10_000;
            operations.push(Operation::Update { price, qty });
        }
    }

    operations
}

trait Book {
    fn update(&mut self, price: u64, qty: u64);
    fn quantity(&self, price: u64) -> Option<u64>;
}

impl Book for TreeBook {
    #[inline]
    fn update(&mut self, price: u64, qty: u64) {
        TreeBook::update(self, price, qty);
    }

    #[inline]
    fn quantity(&self, price: u64) -> Option<u64> {
        self.get(price).map(|level| level.qty)
    }
}

impl Book for DenseBook {
    #[inline]
    fn update(&mut self, price: u64, qty: u64) {
        DenseBook::update(self, price, qty);
    }

    #[inline]
    fn quantity(&self, price: u64) -> Option<u64> {
        self.get(price).map(|level| level.qty)
    }
}

fn seed_book<B: Book>(mut book: B) -> B {
    for price in MIN_PRICE..MAX_PRICE {
        book.update(price, 1 + (price - MIN_PRICE) % 1_000);
    }
    book
}

fn execute<B: Book>(mut book: B, operations: &[Operation]) -> (Duration, u64) {
    let mut checksum = 0_u64;
    let started = Instant::now();

    for operation in operations {
        match *operation {
            Operation::Lookup { price } => {
                checksum = checksum.wrapping_add(book.quantity(price).unwrap_or(0));
            }
            Operation::Update { price, qty } => book.update(price, qty),
        }
    }

    black_box(&book);
    black_box(checksum);
    (started.elapsed(), checksum)
}

fn run_tree(operations: &[Operation]) -> (Duration, u64) {
    execute(seed_book(TreeBook::default()), operations)
}

fn run_dense(operations: &[Operation]) -> (Duration, u64) {
    execute(seed_book(DenseBook::new(MIN_PRICE)), operations)
}

#[derive(Clone, Copy, PartialEq)]
enum Selection {
    Both,
    Tree,
    Dense,
}

struct Config {
    runs: usize,
    selection: Selection,
    warmup: bool,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut config = Self {
            runs: DEFAULT_RUNS,
            selection: Selection::Both,
            warmup: true,
        };
        let mut args = std::env::args().skip(1);

        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--runs" => {
                    let value = args.next().ok_or("--runs requires a positive integer")?;
                    config.runs = value
                        .parse()
                        .map_err(|_| format!("invalid --runs value: {value}"))?;
                    if config.runs == 0 {
                        return Err("--runs must be greater than zero".into());
                    }
                }
                "--book" => {
                    let value = args.next().ok_or("--book requires both, tree, or dense")?;
                    config.selection = match value.as_str() {
                        "both" => Selection::Both,
                        "tree" => Selection::Tree,
                        "dense" => Selection::Dense,
                        _ => return Err(format!("invalid --book value: {value}")),
                    };
                }
                "--no-warmup" => config.warmup = false,
                "--help" | "-h" => {
                    println!(
                        "Usage: session01_dense_book [--runs N] [--book both|tree|dense] [--no-warmup]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }

        Ok(config)
    }
}

fn print_run(name: &str, run: usize, duration: Duration, checksum: u64) {
    let ns_per_operation = duration.as_nanos() as f64 / OPERATION_COUNT as f64;
    println!(
        "{name:<9} run {run:>2}: total={duration:>10.3?}, {ns_per_operation:>8.3} ns/op, checksum={checksum}"
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

fn print_summary(name: &str, durations: &[Duration]) -> f64 {
    let samples: Vec<f64> = durations
        .iter()
        .map(|duration| duration.as_nanos() as f64 / OPERATION_COUNT as f64)
        .collect();
    let minimum = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let sample_median = median(&samples);

    println!(
        "{name:<9} summary: min={minimum:.3}, median={sample_median:.3}, mean={mean:.3}, max={maximum:.3} ns/op"
    );
    sample_median
}

fn main() {
    let config = Config::parse().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!("use --help for usage");
        std::process::exit(2);
    });
    let operations = generate_workload();

    println!(
        "workload: {} deterministic operations, prices [{MIN_PRICE}, {MAX_PRICE}), 70% lookup / 30% update",
        operations.len()
    );

    if config.warmup {
        match config.selection {
            Selection::Both => {
                black_box(run_tree(&operations));
                black_box(run_dense(&operations));
            }
            Selection::Tree => {
                black_box(run_tree(&operations));
            }
            Selection::Dense => {
                black_box(run_dense(&operations));
            }
        }
        println!("warm-up complete\n");
    }

    let mut tree_durations = Vec::with_capacity(config.runs);
    let mut dense_durations = Vec::with_capacity(config.runs);
    let mut expected_checksum = None;

    for run in 1..=config.runs {
        let mut record = |name: &str, result: (Duration, u64), durations: &mut Vec<Duration>| {
            let (duration, checksum) = result;
            if let Some(expected) = expected_checksum {
                assert_eq!(
                    checksum, expected,
                    "implementations produced different results"
                );
            } else {
                expected_checksum = Some(checksum);
            }
            durations.push(duration);
            print_run(name, run, duration, checksum);
        };

        match config.selection {
            Selection::Both if run % 2 == 1 => {
                record("TreeBook", run_tree(&operations), &mut tree_durations);
                record("DenseBook", run_dense(&operations), &mut dense_durations);
            }
            Selection::Both => {
                record("DenseBook", run_dense(&operations), &mut dense_durations);
                record("TreeBook", run_tree(&operations), &mut tree_durations);
            }
            Selection::Tree => record("TreeBook", run_tree(&operations), &mut tree_durations),
            Selection::Dense => record("DenseBook", run_dense(&operations), &mut dense_durations),
        }
    }

    println!();
    let tree_median =
        (!tree_durations.is_empty()).then(|| print_summary("TreeBook", &tree_durations));
    let dense_median =
        (!dense_durations.is_empty()).then(|| print_summary("DenseBook", &dense_durations));

    if let (Some(tree), Some(dense)) = (tree_median, dense_median) {
        let change = (dense / tree - 1.0) * 100.0;
        println!("DenseBook median latency change vs TreeBook: {change:+.2}%");
    }
}
