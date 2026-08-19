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

On the snowflake dataset this is 3.23x compression against 2.17x for the best existing
encoding, at 2^20 rows.

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

Validation: the selector estimated 19.31 bits/value for the snowflake column and the encoder
achieved 19.6. It also recovers the true field boundaries from the data alone — on the real
dataset it picks `bit_starts = [0, 12, 22]`, which is exactly the snowflake layout.

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

Full per-dataset tables, including the gather sweep and FastLanes' own wizard choice with and
without SubIntSplit available, are generated into **[`tables/subintsplit.md`](../tables/subintsplit.md)**
by `scripts/run_subintsplit_tables.sh`. Raw numbers land in
`benchmark/result/subintsplit/subintsplit.csv`.

Headline, snowflake IDs at 2²⁰ rows (1 048 576 rows, 16 rowgroups, ~8 MB raw):

| encoding | ratio | bulk decode | point (decode+index) |
|---|---|---|---|
| ffor | 2.11× | 0.054 ms/rg | 1.18 µs |
| delta | 2.17× | 0.055 ms/rg | 8.72 µs |
| ffor_slpatch | 2.11× | 0.038 ms/rg | 1.20 µs |
| **subintsplit** | **3.23×** | 0.126 ms/rg | 3.79 µs |

SubIntSplit's *native* point path reads 0.214 µs/probe — 18× faster than decoding its own
vector, and 5.5× faster than FFOR's decode-then-index. See the comparability note below before
quoting that number.

**The wizard picks SubIntSplit on its own.** With a default `Connection` it selects
`EXP_SUBINTSPLIT_I64` for the snowflake column (3.23×); with `disable_encoding` applied to both
SubIntSplit tokens it falls back to `EXP_DELTA_I64` at 2.16×. That ablation is what
`Connection::disable_encoding` exists for.

TPC-H partkey and IPv4: the selector chooses a single section, so the result is plain FFOR plus
48 bytes (16 rowgroups × a 3-byte layout header). On IPv4 the wizard prefers
`EXP_DICT_I32_FFOR_U16` at 1.01× whether or not SubIntSplit is available.

Encode is ~4.5× slower than FFOR — the DP sweep runs once per column per rowgroup — though still
far cheaper than `ffor_slpatch`.

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

- native point access (0.214 µs) against decoding its own vector (3.79 µs): **18× faster**;
- against FFOR's decode-then-index (1.18 µs): **5.5× faster** — but that is SubIntSplit doing a
  different, cheaper operation, not beating FFOR at the same one;
- against a single-section FFOR column read *the same way*, splitting must **lose**, because K
  scattered reads cost more than one. Splitting buys compression on this path, not speed.

**Working set.** Benchmarks run at 2²⁰ rows — ~8 MB raw, ~2.5 MB compressed per column — which
exceeds L2 and stresses L3. Numbers taken earlier in this repo's history at 65 536 rows were
entirely cache-resident and are not comparable with these.

**Encode time is not like-for-like either.** Forced-encoding rows evaluate a single candidate;
the wizard rows search the whole pool plus the dictionary pool and run `Cast()` and every
pre-pass. The generated tables bold those two groups independently for this reason.

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
| `benchmark/bench_subintsplit/` | benchmark driver |
| `scripts/run_subintsplit_tables.sh` | generate + build + run + render |
| `scripts/render_subintsplit_tables.py` | Markdown table generator |
| `tables/subintsplit.md` | generated comparison tables |
| `data/generated/subintsplit/` | datasets and generator |
