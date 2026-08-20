# SubIntSplit datasets

Benchmark inputs for the **SubIntSplit** integer encoding.

SubIntSplit targets integer columns whose bits carry several *distinct semantic
fields* — typically a slow-moving high field, a low-cardinality middle field and
a fast-churning low field. Compressing each bit range separately beats
compressing the integer as a whole. Every dataset here has that property by
construction; uniform random integers deliberately do not, and would make the
encoding look useless.

All *committed* datasets are single-column, 65536 rows (= 64 vectors of 1024 =
one full FastLanes rowgroup), stored as a headerless `generated.csv` plus a
`schema.json`. Larger datasets for benchmarking are generated on demand into a
scratch directory and are **not committed** — see [Regenerating](#regenerating).

| Dataset | Type | Content |
|---|---|---|
| `../encodings/subintsplit_i64/` | `FLS_I64` | Twitter snowflake IDs — correctness dataset |
| `snowflake_i64/` | `FLS_I64` | Twitter snowflake IDs — benchmark (independent capture, different seed) |
| `tpch_partkey_i32/` | `FLS_I32` | TPC-H SF=1 lineitem `L_PARTKEY` |
| `ipv4_i32/` | `FLS_I32` | IPv4 addresses packed into 32 bits |

C++ path constants for all four live in `data/include/data/generated.hpp`
(`SUBINTSPLIT_I64_EXPR`, `SUBINTSPLIT_SNOWFLAKE_I64`,
`SUBINTSPLIT_TPCH_PARTKEY_I32`, `SUBINTSPLIT_IPV4_I32`).

## Bit-field structure

### Snowflake IDs (`FLS_I64`)

Twitter layout, MSB → LSB:

```
 63     62                      22 21        12 11         0
┌──────┬──────────────────────────┬────────────┬────────────┐
│ 0    │ timestamp (41b)          │ machine    │ sequence   │
│ sign │ ms since Twitter epoch   │ id (10b)   │ (12b)      │
└──────┴──────────────────────────┴────────────┴────────────┘
```

The stream simulates a firehose capture starting at 2023-06-01 (timestamp field
= ms since the Twitter epoch 2010-11-04T01:42:54.657Z, hence the large constant
offset). Observed in the committed 65536 rows:

* **timestamp** — monotonically non-decreasing, spans ~16.8 s, 3561 distinct
  ticks. Needs 39 bits absolutely, but only ~15 bits *move*: bits 39–63 of the
  ID are constant.
* **machine id** — 40 distinct values out of the addressable 1024, unordered,
  mildly skewed. Bits 18–21 of the ID are therefore constant zero.
* **sequence** — per (millisecond, machine) counter, range 0–44, resets
  constantly. Requests arrive in bursts a single machine serves back to back,
  so the field churns fast over a small range.

All IDs are positive and unique.

### TPC-H `L_PARTKEY` (`FLS_I32`)

Keys live in `[1, 200000]` per the TPC-H spec at scale factor 1, so the **top
14 bits of the int32 are always zero**. On top of the uniform TPC-H base the
generator adds the two skews real order lines exhibit:

* a Zipfian popular-part head (2048 hot parts), and
* locality runs, where consecutive lineitems reference nearby parts from one
  catalogue region (a supplier shipment) — the next key shares the base and
  differs only in its low 8 bits.

Result: the high byte of the key repeats between consecutive rows ~47% of the
time while the low byte is essentially uniform — a clean high/low split.

### IPv4 (`FLS_I32`)

Client addresses as they appear in a web access log, packed as the raw 32
address bits:

```
 31      24 23      16 15       8 7        0
┌──────────┬──────────┬──────────┬──────────┐
│ octet A  │ octet B  │ octet C  │ octet D  │
└──────────┴──────────┴──────────┴──────────┘
     └────────── network ─────────┘└─ host ─┘
```

**These values are the address bits reinterpreted as a signed `int32`**, matching
the convention used elsewhere for this data. Addresses at or above `128.0.0.0`
therefore come out **negative** (min `-2142842706`, max `2124427971`). Consumers
must cast to `uint32_t` before splitting bit ranges.

The addresses are sampled from **real allocated address space**, not synthesised:
`ipv4_prefixes.csv` holds 4096 real IPv4 ranges taken with a deterministic stride
from the DB-IP city-level IPv4 database
(`EncodingsPlayground/Datasets/IPv4/dbip-city-ipv4.csv`), which preserves the real
global distribution of first octets and allocation sizes. Ranges are then drawn
with Zipfian popularity, and each visit emits a short burst of addresses from the
same range (a session).

Observed: 3032 distinct /24 networks, 1451 distinct /16s, and 63% of consecutive
rows share a /24 — so the network part is heavily repeated while the host byte
churns.

The prefix pool is committed so `generate.py` is self-contained and needs no
access to the 229 MB DB-IP source.

## Regenerating

```bash
# exactly reproduces the committed CSVs (byte-identical), including
# ../encodings/subintsplit_i64/
python3 generate.py
```

### Large datasets (`--out-dir`)

Benchmarks want 1048576-row inputs. Those are ~40 MB in total and are
deliberately **not committed** — they would dwarf the rest of the repository and
make every PR touching them unreviewable, and they are exactly reproducible from
this script anyway. Generate them into a scratch directory instead:

```bash
python3 generate.py --out-dir /tmp/sis-big --rows 1048576
```

`--out-dir DIR` writes **only the three benchmark datasets** as
`DIR/<name>/{generated.csv,schema.json}` — the same per-dataset layout as this
directory, so anything that can read `data/generated/subintsplit/` can read `DIR`
unchanged. Directories are created as needed, and nothing under `data/` is
touched.

`../encodings/subintsplit_i64/` is **not** written by an `--out-dir` run. It is a
committed test fixture: `test/src/expression_tests/subintsplit_test.cpp` depends
on it being exactly 65536 rows, so regenerating it at a different `--rows` would
silently change what the test tests. The script therefore refuses `--rows` other
than 65536 unless `--out-dir` is given; `--force` overrides this, but only use it
if you genuinely intend to change the committed fixture (and then commit it).

All RNGs are seeded, so a given `--rows` always produces the same bytes,
regardless of the output directory. Only the Python standard library is required.

### Rebuilding the IPv4 prefix pool

```bash
python3 generate.py --rebuild-prefixes \
    --dbip /path/to/dbip-city-ipv4.csv
```

### A fourth, real-world dataset

`extract_real_snowflake.py` samples real Twitter snowflake IDs from
`EncodingsPlayground/Datasets/TwitterSnowflake/tweet_ids.parquet` (30.7M rows,
sibling checkout) instead of simulating the field structure:

```bash
python3 extract_real_snowflake.py --out-dir /tmp/sis-big --rows 1048576
```

Writes `snowflake_i64_real/` alongside the other datasets. Requires `pyarrow`
(only for this script). `run_subintsplit_tables.sh` includes it automatically
when the parquet source is present; the benchmark otherwise skips it.
