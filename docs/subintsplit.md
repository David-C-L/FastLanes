# SubIntSplit

An integer encoding for values whose bits carry distinct semantic fields.

Written for someone who knows FastLanes but not SubIntSplit. The encoding originates in
Meta's Nimble; this document covers the FastLanes port, which differs from it in ways that
matter (see [Divergences](#divergences-from-the-nimble-implementation)).

## What it does, and why

A Twitter snowflake ID is not one number. It is three:

```
 63                    22        12         0
  +---------------------+---------+---------+
  |  timestamp (41b)    | machine | seq     |
  +---------------------+---------+---------+
                         (10b)     (12b)
```

Those fields behave completely differently. Within any 1024-row vector the timestamp barely
moves, the machine id takes one of a few dozen values, and the sequence number churns. But
FFOR sees only the concatenation: it subtracts the vector's minimum and bit-packs the
residual, and the residual of the *whole* value is wide because the low fields dominate it.

SubIntSplit cuts each value into contiguous bit ranges and FFORs each range separately, per
vector, with its own base and bit width. Each field then pays its own narrow width.

On the real Twitter snowflake dataset (`snowflake_i64_real`) this is 1.62x compression against
1.43x for the best existing encoding (delta), at 2^20 rows. The synthetic snowflake generator
used before the real dataset was wired in shows a larger, structurally cleaner margin (3.23x
against 2.17x) — see [Results](#results) for both, side by side.

**Where it does nothing.** On TPC-H partkey and IPv4 the selector chooses a single section
and the result is plain FFOR to within 3 bytes. This is the design working: absent
exploitable bit-range structure it degrades to FFOR rather than regressing.

## Format

Per column, in segment order:

```
section 0     : bitpacked | base | bitwidth      per-vector, the enc_ffor_opr triple
...
section K-1   : bitpacked | base | bitwidth
header        : uint8 n_sections, then n_sections bytes of bit_start    block-based
```

`bit_end` is implied — section *s* ends where section *s+1* starts, and the last ends at
`8 * sizeof(PT) - 1`. Sections are LSB-first and always tile the full width.

**The header is last, deliberately.** This is the only operator in the codebase with a
*dynamic* segment count. Every other one knows how many operands it owns at construction;
this one cannot know until it has read K. Placing the header at `cur_operand - 0` lets the
decoder read K first, locate the 3K section segments below it, then rewind `cur_operand` by
`3K + 1`.

It is block-based (`MakeBlockBased()`) so `PhysicalExpr::Size` amortises it across the
rowgroup. Without that, wizard selection — which encodes only a sample of vectors — would
charge the full header against a handful of vectors and unfairly penalise the encoding.

## Split selection

`src/include/fls/expression/subintsplit_selector.hpp`. A dynamic program over bit positions:
`dp[k][i]` is the minimum cost of covering bits `[0, i)` with exactly *k* sections, and the
answer is the best over `k <= max_sections`. Carrying the section count as a DP dimension
(rather than rejecting oversized plans afterwards) gives the true optimum among admissible
plans instead of a fallback with no optimality guarantee.

The cost oracle is **measured, not modelled**. Because FastLanes bit-packs per vector with a
per-vector FFOR base, the cost of a candidate range is exactly

```
sum over sampled vectors of  1024 * bit_width(max - min)  of that bit range
```

scaled to the full column. Nimble needs seven closed-form encoding-size models here; the port
needs none, because the thing being predicted is directly computable.

Validation: on the synthetic snowflake column, the selector estimated 19.31 bits/value and the
encoder achieved 19.6, and it recovers the assumed field boundaries from the data alone —
`bit_starts = [0, 12, 22]`, exactly the synthetic generator's layout. On the real Twitter
snowflake IDs (`snowflake_i64_real`) it settles on a 4-way split instead,
`bit_starts = [0, 12, 17, 22]` — real IDs aren't as cleanly aligned to the idealized
timestamp/machine/sequence boundary as the synthetic generator assumes, which the DP picks up on
its own rather than needing to be told.

### Two things worth knowing before tuning it

**FFOR's base already absorbs constant high bits.** A column of small values in a 64-bit type
does *not* benefit from splitting the empty high bits off: FFOR subtracts the minimum, so
those bits are already free, and a second section would only add segments and a base. This is
the single biggest reason the FastLanes port yields a smaller margin than the Nimble version
on the same data.

**FastLanes rowgroups are 64 vectors.** Per-vector metadata (3 segments × 64 vectors × K
sections) is proportionally heavier here than in Nimble's long streams, so the DP settles on
fewer sections than Nimble does. `max_sections` defaults to 4.

`split_penalty` is the compression-versus-decode-speed knob, and it is a real one rather than
a fudge factor: each extra section costs three more segments *and* one more pass over the
output in bulk decode.

## Access paths

### Bulk

Per section, `unffor` into scratch, then accumulate into the output — section 0 writes, later
sections OR in, with the branch hoisted out of the inner loop.

Cost is linear in section count: 0.126 ms/rowgroup for 3 sections against 0.054 for plain
FFOR. **Bulk decode is where the compression is paid for.**

### Point

`unffor_single` (`src/include/fls/primitive/unffor_single.hpp`) reaches one value by position
arithmetic, so a point read costs K scattered word reads instead of a decoded vector.

FastLanes had no point path at all before this — `unffor` is all-or-nothing over a vector and
`RandomAccessor::RandomAccess` has long been commented out.

The layout arithmetic is derived from the *generated* kernels in
`src/alp/src/fastlanes_gen_{ffor,unffor}.cpp`, which are the ground truth. Re-verify against
them if the code generator ever changes:

- a vector splits into `LANES = 128 / sizeof(PT)` lanes of `8 * sizeof(PT)` slots;
- unpacked output is in natural index order, so `lane = idx % LANES`, `slot = idx / LANES`;
- **packed word *w* of every lane is contiguous** — `in[w * LANES + lane]`. The lane stride is
  1 and the word stride is LANES. This is the part most people get backwards;
- within a lane, slots are a little-endian stream of `bw`-bit fields starting at `slot * bw`.

Because slots-per-lane equals the word width exactly, fields tile a lane with no remainder, so
the straddle read can never run past the last packed word. `bw == 0` stores nothing and must
not dereference the buffer; `bw == TW` must not compute its mask as `(1 << bw) - 1` nor take
the straddle branch. Both are UB if written naively.

`test/src/primitive_tests/unffor_single_test.cpp` compares against the generated kernels for
every element type, every bit width, and all 1024 indices. It is the tripwire: if the
generator changes, it fails loudly instead of corrupting point reads silently.

### Gather

- `GatherPointwise` — reaches only the requested rows, sections outermost so each section's
  bit width, base and pointer load once per gather rather than once per row.
- `GatherDecoded` — decodes the vector, then gathers.
- `Gather` — picks between them at `gather_decode_threshold` (default 128).

`GatherDecoded` is not just a fallback. It is the like-for-like comparison against Nimble's
`bulkScan`, which also decodes a whole span and gathers out of it. Within a vector, "decode
the span" and "decode the vector" are the same operation, because `unffor` cannot decode part
of one.

Measured crossover, at 2²⁰ rows with all indices pre-generated outside the timed loop:

| column | sections | median bit width | crossover |
|---|---|---|---|
| tpch_partkey_i32 | 1 | 18 | ~35–40 rows |
| snowflake_i64 | 3 | 5,6,8 | ~110–120 rows |
| ipv4_i32 | 1 | 32 | ~150 rows |

**`gather_decode_threshold` is a single constant (128) and the data says it should not be.** The
crossover depends on section count *and* bit width — two single-section columns here differ by
4× — so 128 sits inside the snowflake and IPv4 crossovers but is far above TPC-H's. A value
derived at decode time from `bit_starts.size()` and the median bit width would fit the measured
data much better. Left as a constant for now, and flagged here rather than silently retuned,
because choosing the formula needs more columns than three.

## Results

`snowflake_i64_real` is the default/headline dataset (see [Data source](#data-source) above).
The numbers below are from it; the synthetic `snowflake_i64` column is kept as a secondary
reference alongside it, since it isolates the encoding's behavior on a cleaner, idealized field
layout than real IDs actually have.

Full per-dataset tables, including the gather sweep and FastLanes' own wizard choice with and
without SubIntSplit available, are generated into **[`tables/subintsplit.md`](../tables/subintsplit.md)**
by `scripts/run_subintsplit_tables.sh`. Raw numbers land in
`benchmark/result/subintsplit/subintsplit.csv`.

Headline, real Twitter snowflake IDs at 2²⁰ rows (1 048 576 rows, 16 rowgroups, ~8 MB raw):

| encoding | ratio | bulk decode | point (decode+index) |
|---|---|---|---|
| ffor | 1.29× | 0.026 ms/rg | 0.516 µs |
| delta | 1.43× | 0.042 ms/rg | 3.968 µs |
| ffor_slpatch | 1.28× | 0.083 ms/rg | 0.568 µs |
| **subintsplit** | **1.62×** | 0.187 ms/rg | 1.785 µs |

The synthetic snowflake reference dataset, same row count, shows a larger margin:

| encoding | ratio | bulk decode | point (decode+index) |
|---|---|---|---|
| ffor | 2.11× | 0.041 ms/rg | 1.047 µs |
| delta | 2.17× | 0.027 ms/rg | 5.822 µs |
| ffor_slpatch | 2.11× | 0.059 ms/rg | 1.166 µs |
| **subintsplit** | **3.23×** | 0.070 ms/rg | 1.373 µs |

SubIntSplit's *native* point path on the real dataset reads 0.156 µs/probe — 11.4× faster than
decoding its own vector (1.785 µs), and 3.3× faster than FFOR's decode-then-index (0.516 µs). See
the comparability note below before quoting either number.

**The wizard picks SubIntSplit on its own**, on both datasets. On the real dataset, with a
default `Connection` it selects `EXP_SUBINTSPLIT_I64` (1.62×); with `disable_encoding` applied to
both SubIntSplit tokens it falls back to `EXP_DELTA_I64` at 1.43×. That ablation is what
`Connection::disable_encoding` exists for. (On the synthetic dataset the same ablation is 3.23×
against 2.16×.)

TPC-H partkey and IPv4: the selector chooses a single section, so the result is plain FFOR plus
48 bytes (16 rowgroups × a 3-byte layout header) — unaffected by which snowflake dataset is
default, since these are separate columns. On IPv4 the wizard prefers `EXP_DICT_I32_FFOR_U16` at
1.01× whether or not SubIntSplit is available.

Encode is ~7.3× slower than FFOR on the real dataset (1319 ms vs 180 ms) — the DP sweep runs once
per column per rowgroup — though still far cheaper than `ffor_slpatch`. The gap is smaller on the
synthetic dataset (~4.5×, 949 ms vs 122 ms); not investigated further here.

### Reading these numbers honestly

**Bulk** is comparable across all encodings.

**Gather** is comparable only through `GatherDecoded`. `GatherPointwise` is a capability the
others do not have.

**Point access is not a like-for-like speed test.** Every other encoding must decode the
containing vector and index into it. The main table's `Point` column reports exactly that for
every row, including SubIntSplit — `point_decode_then_index` — so that column *is* comparable.
SubIntSplit's position-arithmetic path is reported separately, in the native random-access
table, precisely because it is a *capability* gap rather than a faster implementation of the
same operation. Three framings to keep in view:

- native point access (0.156 µs, real dataset) against decoding its own vector (1.785 µs):
  **11.4× faster**;
- against FFOR's decode-then-index (0.516 µs): **3.3× faster** — but that is SubIntSplit doing a
  different, cheaper operation, not beating FFOR at the same one;
- against a single-section FFOR column read *the same way*, splitting must **lose**, because K
  scattered reads cost more than one. Splitting buys compression on this path, not speed.

**Working set.** Benchmarks run at 2²⁰ rows — ~8 MB raw, ~2.5 MB compressed per column — which
exceeds L2 and stresses L3. Numbers taken earlier in this repo's history at 65 536 rows were
entirely cache-resident and are not comparable with these.

**Encode time is not like-for-like either.** Forced-encoding rows evaluate a single candidate;
the wizard rows search the whole pool plus the dictionary pool and run `Cast()` and every
pre-pass. The generated tables bold those two groups independently for this reason.

## Limited codec-set comparison

The tables above compare SubIntSplit against FastLanes' *entire* wizard pool — nine top-level
candidates plus Dictionary's own three-way index-width search, several of which (Delta,
`ffor_slpatch`, RLE with an SLP patch, cross-row RLE) are more elaborate than a conservative
encoder would necessarily ship. `bench_subintsplit_limited` (same driver structure as
`bench_subintsplit`, sharing all of its measurement code via
`BenchSubIntSplitCommon.hpp`) restricts the field to six simpler, more familiar codecs:
**RLE, Constant, Dictionary, FFOR, Uncompressed, FrequencyPartition** — plus SubIntSplit itself.
This isolates SubIntSplit's contribution against codecs a smaller or more conservative encoder
would actually have, rather than against FastLanes' full, more elaborate candidate pool.

Mechanically: the per-codec forced rows (`uncompressed`, `rle`, `dict`, `ffor`, `frequency`,
`subintsplit`) each use `Connection::force_schema_pool` with exactly that codec's token(s); the
two wizard rows (`wizard_limited_with_sis`/`wizard_limited_without_sis`) use
`Connection::disable_encoding` to remove everything from the default pool *outside* the six-codec
set (`Delta`, `ffor_slpatch`, `RLE` with an SLP patch, cross-row RLE — both i32 and i64 widths),
leaving the wizard to search only among the six.

**Constant is not a forced row.** `force_schema_pool` cannot select `EXP_CONSTANT_I32/I64` — it
bypasses the wizard's poolable-candidate path entirely (`constant_check` is an automatic
structural pre-pass in `wizard.cpp` that runs *before* the candidate pool is even consulted, and
is skipped altogether once a pool is forced). None of this benchmark's datasets are genuinely
constant-valued, so a forced Constant row would be a no-op or a build-time error either way. It
remains a real member of the six-codec set for the two wizard-limited rows, where it can still
apply automatically to any column that happens to qualify — it simply cannot be exercised as its
own forced comparison row here.

Full results: **[`tables/subintsplit_limited.md`](../tables/subintsplit_limited.md)**, generated
by the same `scripts/run_subintsplit_tables.sh` that produces the main table above. Raw numbers
land in `benchmark/result/subintsplit/subintsplit_limited.csv`. The same comparability caveats
from [Reading these numbers honestly](#reading-these-numbers-honestly) apply unchanged.

## Divergences from the Nimble implementation

| | Nimble | Here | Why |
|---|---|---|---|
| Section codec | per-section choice from Trivial/Dict/RLE/… | always FFOR | FastLanes has no mechanism for a dynamically chosen nested encoding; FFOR is its native primitive, and it makes the DP's cost oracle measurable |
| Section storage | narrowest type that fits | column's unsigned type | costs nothing in size (`bw * 1024 / 8` either way), only decode bandwidth |
| Cost model | seven closed-form models | measured FFOR widths | see above |
| Point access | `skip(n)` + `materialize(1)`, forward cursor | O(1) position arithmetic | FastLanes' fixed vectors make it possible; Nimble's cursor is stateful and forward-only |
| Types | 32/64-bit ints and floats | `i64`, `i32` | scope |
| Nulls | supported | not supported | FastLanes handles nulls via separate `EXP_NULL_*` expressions |

**Closing the gap.** Per-section codecs are the interesting one: `physical_operator` already
admits `sp<PhysicalExpr>` as an alternative, so a section could carry a nested expression, with
the per-section token persisted in the header. Narrow section storage types are a smaller,
purely mechanical win on bulk decode — the path most in need of one.

## Reproducing

```bash
export FASTLANES_DATA_DIR=$PWD/build/_deps/data-src   # avoids re-downloading ~1 GB
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DFLS_BUILD_TESTING=ON
cmake --build build --parallel

./build/test/src/primitive_tests/primitive_test          # unffor_single vs generated kernels
./build/test/src/unit_tests/unit_test                    # selector
./build/test/src/expression_tests/test_subintsplit        # round-trip
./build/test/src/expression_tests/test_subintsplit_access # point + gather vs full decode

# benchmarks + tables in one step: generates 2^20-row datasets, builds, runs, renders
scripts/run_subintsplit_tables.sh
```

That writes `benchmark/result/subintsplit/subintsplit.csv` and
[`tables/subintsplit.md`](../tables/subintsplit.md). `--skip-bench` re-renders the tables from
the existing CSV, `--rows N` changes the dataset size.

The 2^20-row datasets are ~38 MB and are therefore **not committed** — the driver regenerates
them deterministically from fixed seeds into `bench-build/subintsplit-data`. The committed
65 536-row datasets under `data/generated/` remain the correctness-test inputs and are never
regenerated at another size; `generate.py` refuses to, because `test_subintsplit` depends on
them.

### Data source

`snowflake_i64_real` — real Twitter snowflake IDs — is the headline SubIntSplit dataset; the
synthetic `snowflake_i64` (same field layout, simulated) is a secondary reference, not the
main result.

`snowflake_i64_real` is produced by `data/generated/subintsplit/extract_real_snowflake.py`,
which samples from `../../../EncodingsPlayground/Datasets/TwitterSnowflake/tweet_ids.parquet`
relative to the FastLanes checkout root — a sibling `EncodingsPlayground` checkout, 30.7M rows,
not committed to this repo — and requires `pyarrow`. `scripts/run_subintsplit_tables.sh` runs
this step automatically, best-effort: if the parquet or `pyarrow` is missing, it prints a
skip notice and the pipeline falls back to `snowflake_i64` instead. Get the real dataset by
placing an `EncodingsPlayground` checkout containing that parquet file next to `FastLanes/` and
installing `pyarrow`.

Note: several *pre-existing* test failures in this repo are unrelated to SubIntSplit —
datasets such as `data/generated/encodings/frequency_dbl` and
`data/generated/single_columns/fls_str` ship a `schema.json` with no `generated.csv`, so those
tests fail with "csv file is not found" on a clean checkout.

## Source map

| File | Role |
|---|---|
| `src/include/fls/expression/subintsplit_selector.hpp` | DP split selection |
| `src/include/fls/expression/subintsplit_operator.hpp` | operator declarations, storage layout |
| `src/expression/subintsplit_operator.cpp` | encode, decode, point, gather |
| `src/include/fls/primitive/unffor_single.hpp` | single-value unpack |
| `benchmark/bench_subintsplit/` | benchmark driver + shared harness (`BenchSubIntSplitCommon.hpp`) |
| `benchmark/bench_subintsplit_limited/` | limited codec-set benchmark driver |
| `scripts/run_subintsplit_tables.sh` | generate + build + run + render (both benchmark binaries) |
| `scripts/render_subintsplit_tables.py` | Markdown table generator |
| `tables/subintsplit.md` | generated comparison tables |
| `tables/subintsplit_limited.md` | generated limited codec-set comparison tables |
| `data/generated/subintsplit/` | datasets and generator |
