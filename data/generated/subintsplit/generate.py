#!/usr/bin/env python3
"""Reproducible generator for the SubIntSplit encoding datasets.

SubIntSplit targets integer columns whose bits carry several *distinct semantic
fields* (a slow-moving high field, a low-cardinality middle field, a churning
low field).  Splitting such a value into its bit ranges and compressing each
range independently beats compressing the whole integer.  The datasets produced
here all have that property; uniform random integers deliberately do not.

Datasets produced (all single column, headerless CSV + schema.json):

  ../encodings/subintsplit_i64/   FLS_I64  Twitter snowflake IDs (correctness)
  snowflake_i64/                  FLS_I64  Twitter snowflake IDs (benchmark)
  tpch_partkey_i32/               FLS_I32  TPC-H lineitem L_PARTKEY
  ipv4_i32/                       FLS_I32  IPv4 addresses packed into 32 bits

Usage
-----
    python3 generate.py                 # 65536 rows each (what is committed)
    python3 generate.py --rows 1048576  # bigger, for local benchmarking

Everything is driven by fixed seeds, so re-running with the same --rows
reproduces byte-identical CSVs.

The IPv4 dataset is built from *real* address space: `ipv4_prefixes.csv` holds
4096 real allocated ranges sampled from the DB-IP city-level IPv4 database.
That pool is committed so this script is self-contained; it can be rebuilt with

    python3 generate.py --rebuild-prefixes --dbip /path/to/dbip-city-ipv4.csv
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENCODINGS_DIR = HERE.parent / "encodings"
PREFIX_POOL = HERE / "ipv4_prefixes.csv"

DEFAULT_ROWS = 65536  # 64 vectors of 1024 = one full FastLanes rowgroup

# ── Snowflake layout (Twitter): 1 unused sign bit | 41 ts | 10 machine | 12 seq
SNOWFLAKE_TIMESTAMP_BITS = 41
SNOWFLAKE_MACHINE_BITS = 10
SNOWFLAKE_SEQUENCE_BITS = 12
SNOWFLAKE_MACHINE_SHIFT = SNOWFLAKE_SEQUENCE_BITS
SNOWFLAKE_TIMESTAMP_SHIFT = SNOWFLAKE_SEQUENCE_BITS + SNOWFLAKE_MACHINE_BITS
SNOWFLAKE_SEQUENCE_MASK = (1 << SNOWFLAKE_SEQUENCE_BITS) - 1

# ms between the Twitter snowflake epoch (2010-11-04T01:42:54.657Z) and the
# simulated start of the stream (2023-06-01T00:00:00Z).  Real snowflake IDs
# encode exactly this quantity, which is why their top timestamp bits are
# near-constant within any one capture.
SNOWFLAKE_EPOCH_OFFSET_MS = 397_182_945_343

# Number of ID-issuing machines actually present in the stream.  Twitter-scale
# deployments use a few dozen out of the 1024 addressable ids.
SNOWFLAKE_NUM_MACHINES = 40

# TPC-H scale factor 1: L_PARTKEY in [1, 200000].
TPCH_MAX_PARTKEY = 200_000


# ──────────────────────────────────────────────────────────────────────────
# dataset writing
# ──────────────────────────────────────────────────────────────────────────
def write_dataset(directory: Path, values: list[int], fls_type: str, readme: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    schema = {"columns": [{"name": "COLUMN_0", "type": fls_type}]}
    (directory / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    (directory / "generated.csv").write_text("".join(f"{v}\n" for v in values))
    (directory / "README.MD").write_text(readme)


# ──────────────────────────────────────────────────────────────────────────
# 1/2. Twitter snowflake IDs
# ──────────────────────────────────────────────────────────────────────────
def gen_snowflake(rows: int, seed: int) -> list[int]:
    """Simulate a snowflake ID stream captured from a live firehose.

    Each of the three fields behaves differently, which is exactly what
    SubIntSplit is meant to exploit:

    * timestamp advances monotonically over tens of seconds, so its high bits
      are constant and only its low ~16 bits move within one dataset;
    * machine id comes from a small fixed fleet (a few dozen of the 1024
      addressable ids), so its top bits are constant and the rest is a
      low-cardinality, unordered field;
    * sequence counts the IDs one machine issues inside one millisecond.
      Requests arrive in bursts that a single machine serves back to back, so
      the field churns fast over a small range and resets constantly.
    """
    rng = random.Random(seed)
    machines = list(range(SNOWFLAKE_NUM_MACHINES))
    # Machines are not equally busy in practice: weight them mildly.
    weights = [1.0 + (i % 7) * 0.25 for i in machines]

    timestamp = SNOWFLAKE_EPOCH_OFFSET_MS
    seq_counters: dict[tuple[int, int], int] = {}
    out: list[int] = []
    while len(out) < rows:
        machine = rng.choices(machines, weights)[0]
        # A burst of requests served back to back by one machine in one tick.
        burst = rng.randint(1, 24)
        for _ in range(burst):
            key = (timestamp, machine)
            seq = seq_counters.get(key, 0)
            if seq > SNOWFLAKE_SEQUENCE_MASK:  # exhausted -> force clock forward
                timestamp += 1
                key = (timestamp, machine)
                seq = seq_counters.get(key, 0)
            seq_counters[key] = seq + 1
            out.append(
                (timestamp << SNOWFLAKE_TIMESTAMP_SHIFT)
                | (machine << SNOWFLAKE_MACHINE_SHIFT)
                | seq
            )
        # Fleet-wide the stream runs at roughly one tick per burst, with the
        # occasional quiet gap.
        timestamp += rng.randint(0, 2)
        if rng.random() < 0.02:
            timestamp += rng.randint(1, 200)
    return out[:rows]


# ──────────────────────────────────────────────────────────────────────────
# 3. TPC-H lineitem L_PARTKEY
# ──────────────────────────────────────────────────────────────────────────
def gen_tpch_partkey(rows: int, seed: int) -> list[int]:
    """TPC-H SF=1 L_PARTKEY-like column.

    Keys live in [1, 200000] so the top 14 bits of the int32 are always zero.
    On top of the TPC-H uniform base we add the two skews that real order lines
    exhibit: a popular-part (Zipfian) head, and locality runs where consecutive
    lineitems reference nearby part keys from the same catalogue region.  Both
    make the *upper* bits of the key change far more slowly than the lower ones.
    """
    rng = random.Random(seed)

    # Zipfian head over the 2048 most popular parts.
    hot_n = 2048
    hot_weights = [1.0 / (i ** 0.9) for i in range(1, hot_n + 1)]
    hot_parts = [rng.randint(1, TPCH_MAX_PARTKEY) for _ in range(hot_n)]

    out: list[int] = []
    while len(out) < rows:
        r = rng.random()
        if r < 0.45:
            # popular part, single lineitem
            out.append(rng.choices(hot_parts, hot_weights)[0])
        elif r < 0.85:
            # a run of lineitems referencing nearby parts from one catalogue
            # region (a supplier's shipment) -> shared high bits
            base = rng.randint(1, TPCH_MAX_PARTKEY)
            for _ in range(rng.randint(2, 12)):
                key = base + rng.randint(0, 255)
                out.append(min(key, TPCH_MAX_PARTKEY))
        else:
            out.append(rng.randint(1, TPCH_MAX_PARTKEY))
    return out[:rows]


# ──────────────────────────────────────────────────────────────────────────
# 4. IPv4 addresses packed into 32 bits
# ──────────────────────────────────────────────────────────────────────────
def build_prefix_pool(dbip_csv: Path, count: int = 4096) -> list[tuple[int, int]]:
    """Sample `count` real allocated IPv4 ranges from the DB-IP city database.

    The DB-IP csv is ~3.2M rows of `start_ip,end_ip,country,...`; we take a
    deterministic evenly-spaced stride so the sample keeps the real global
    distribution of allocation sizes and first octets.
    """
    ranges: list[tuple[int, int]] = []
    with dbip_csv.open("r", encoding="utf-8", errors="replace") as fh:
        total = sum(1 for _ in fh)
    stride = max(1, total // count)
    with dbip_csv.open("r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh):
            if idx % stride:
                continue
            start_s, _, rest = line.partition(",")
            end_s, _, _ = rest.partition(",")
            if ":" in start_s:  # skip any ipv6 row
                continue
            try:
                start = ipv4_to_int(start_s)
                end = ipv4_to_int(end_s)
            except ValueError:
                continue
            if end < start:
                continue
            ranges.append((start, end))
            if len(ranges) == count:
                break
    return ranges


def ipv4_to_int(text: str) -> int:
    parts = text.strip().split(".")
    if len(parts) != 4:
        raise ValueError(text)
    value = 0
    for p in parts:
        octet = int(p)
        if not 0 <= octet <= 255:
            raise ValueError(text)
        value = (value << 8) | octet
    return value


def load_prefix_pool() -> list[tuple[int, int]]:
    if not PREFIX_POOL.exists():
        raise SystemExit(
            f"{PREFIX_POOL} is missing; rebuild it with "
            "--rebuild-prefixes --dbip /path/to/dbip-city-ipv4.csv"
        )
    pool = []
    for line in PREFIX_POOL.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        start_s, end_s = line.split(",")
        pool.append((int(start_s), int(end_s)))
    return pool


def gen_ipv4(rows: int, seed: int) -> list[int]:
    """Client IPv4 addresses as they appear in a web access log.

    Addresses are drawn from real allocated ranges (see `ipv4_prefixes.csv`),
    with a Zipfian popularity over ranges and short bursts from one range, so
    the network part (high 16-24 bits) is heavily repeated while the host part
    (low 8-11 bits) churns.  Values are the raw 32 address bits reinterpreted
    as a signed int32, so addresses at or above 128.0.0.0 come out negative.
    """
    rng = random.Random(seed)
    pool = load_prefix_pool()
    weights = [1.0 / (i ** 1.1) for i in range(1, len(pool) + 1)]
    order = list(range(len(pool)))
    rng.shuffle(order)

    out: list[int] = []
    while len(out) < rows:
        idx = order[rng.choices(range(len(pool)), weights)[0]]
        start, end = pool[idx]
        span = end - start
        burst = rng.randint(1, 16)  # a session: several hits from one network
        for _ in range(burst):
            addr = start if span == 0 else start + rng.randint(0, span)
            out.append(to_signed_i32(addr))
    return out[:rows]


def to_signed_i32(value: int) -> int:
    return value - (1 << 32) if value >= (1 << 31) else value


# ──────────────────────────────────────────────────────────────────────────
# stats / main
# ──────────────────────────────────────────────────────────────────────────
def report(name: str, values: list[int]) -> None:
    lo, hi = min(values), max(values)
    card = len(set(values))
    print(f"{name:<22} rows={len(values)} min={lo} max={hi} distinct={card}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS, help=f"rows per dataset (default {DEFAULT_ROWS})")
    ap.add_argument("--rebuild-prefixes", action="store_true", help="rebuild ipv4_prefixes.csv from the DB-IP csv")
    ap.add_argument("--dbip", type=Path, help="path to dbip-city-ipv4.csv (with --rebuild-prefixes)")
    args = ap.parse_args()

    if args.rebuild_prefixes:
        if not args.dbip:
            raise SystemExit("--rebuild-prefixes needs --dbip")
        pool = build_prefix_pool(args.dbip)
        PREFIX_POOL.write_text(
            "# real allocated IPv4 ranges sampled from the DB-IP city-level database\n"
            "# start_ip_int,end_ip_int\n"
            + "".join(f"{s},{e}\n" for s, e in pool)
        )
        print(f"wrote {PREFIX_POOL} ({len(pool)} ranges)")

    rows = args.rows

    correctness = gen_snowflake(rows, seed=42)
    write_dataset(
        ENCODINGS_DIR / "subintsplit_i64",
        correctness,
        "FLS_I64",
        "- to test subintsplit on int64\n"
        "- Twitter-format snowflake IDs: [0 | 41-bit ms timestamp | 10-bit machine id | 12-bit sequence]\n"
        "- generated by data/generated/subintsplit/generate.py\n",
    )
    report("subintsplit_i64", correctness)

    # A second, independent capture so the benchmark input is not a byte copy
    # of the correctness input.
    snowflake = gen_snowflake(rows, seed=43)
    write_dataset(
        HERE / "snowflake_i64",
        snowflake,
        "FLS_I64",
        "- subintsplit benchmark: Twitter snowflake IDs\n"
        "- [0 | 41-bit ms timestamp | 10-bit machine id | 12-bit sequence]\n"
        "- generated by data/generated/subintsplit/generate.py\n",
    )
    report("snowflake_i64", snowflake)

    partkey = gen_tpch_partkey(rows, seed=1337)
    write_dataset(
        HERE / "tpch_partkey_i32",
        partkey,
        "FLS_I32",
        "- subintsplit benchmark: TPC-H SF=1 lineitem L_PARTKEY\n"
        "- keys in [1, 200000]: top 14 bits always zero, high bits move slowly\n"
        "- generated by data/generated/subintsplit/generate.py\n",
    )
    report("tpch_partkey_i32", partkey)

    ipv4 = gen_ipv4(rows, seed=2024)
    write_dataset(
        HERE / "ipv4_i32",
        ipv4,
        "FLS_I32",
        "- subintsplit benchmark: IPv4 addresses packed into a signed int32\n"
        "- sampled from real allocated ranges (DB-IP), clustered by network prefix\n"
        "- generated by data/generated/subintsplit/generate.py\n",
    )
    report("ipv4_i32", ipv4)


if __name__ == "__main__":
    main()
