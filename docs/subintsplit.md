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

SubIntSplit cuts each value into contiguous bit ranges chosen once per column by a DP over bit
positions, then, independently, picks each range's own codec by real measured cost from
`{Uncompressed, Constant, RLE, Dictionary, FFOR, FFOR_SLPATCH, FrequencyPartition}` — the same
seven codecs the [limited comparison](#limited-codec-set-comparison) below measures. A field that
is genuinely just "the low bits of a wide number" pays its own narrow FFOR width, same as
before; a field that happens to be constant, low-cardinality, or dominated by one repeated value
can now do better than FFOR at that width. See [Split selection](#split-selection) for the DP
(unchanged) and [Format](#format) for how a section's codec is chosen and persisted (new).

On the real Twitter snowflake dataset (`snowflake_i64_real`) this is 1.71x compression against
1.43x for the best existing encoding (delta), at 2^20 rows. The synthetic snowflake generator
used before the real dataset was wired in shows a larger, structurally cleaner margin (3.77x
against 2.17x) — see [Results](#results) for both, side by side.

**Where it does less.** On TPC-H partkey the selector still chooses a single section that
resolves to plain FFOR, unaffected (absent exploitable bit-range structure, splitting has
nothing to isolate and the section-codec search converges on the same answer bit-splitting was
already giving). IPv4 is more interesting now than it used to be: the single section it picks
no longer bottoms out at FFOR the way it always did before per-section codec choice existed —
on this run it resolves to a non-FFOR codec at effectively the same ratio (Dictionary is what
the plain wizard separately picks for this column too), which costs point/gather access relative
to FFOR's O(1) path. This is the real trade-off per-section selection introduces, not a special
case: see [Reading these numbers honestly](#reading-these-numbers-honestly).

## Format

Per column, in segment order:

```
section 0     : however many segments its chosen codec's own encoder produces
...
section K-1   : ditto
header        : uint8 n_sections, then n_sections 4-byte records     block-based
                (bit_start: 1B, OperatorToken: 2B, operand count: 1B)
```

`bit_end` is implied — section *s* ends where section *s+1* starts, and the last ends at
`8 * sizeof(PT) - 1`. Sections are LSB-first and always tile the full width.

Each section's codec is chosen independently by `subintsplit::select_section_encoding`
(`subintsplit_section_selector.hpp`) from real measured cost — the same `TryExpr`/`ChooseBestExpr`
oracle the top-level wizard uses, plus an explicit `constant_visit` pre-pass so Constant is
reachable at all (it can never be selected through `force_schema_pool`, which is why it's the one
codec that can't be exercised as a forced row in the [limited
comparison](#limited-codec-set-comparison)). A plain-FFOR section builds a real `enc_ffor_opr` via
`Interpreter::Encoding::Interpret` and stores its usual bitpacked/bitwidth/base triple; a
multi-operator codec (FFOR_SLPATCH: unffor + slpatch; RLE: unffor + rsum + rle_map; Dictionary:
unffor + dict) stores however many segments its own chain produces, and the header's per-section
operand count records exactly that, so the decoder never has to assume a fixed stride.

**The header is last, deliberately.** This is the only operator in the codebase with a
*dynamic* segment count. Every other one knows how many operands it owns at construction;
this one cannot know until it has read K (and, since sections can now differ in operand count
too, how many operands each individual section owns). Placing the header at `cur_operand - 0`
lets the decoder read K and every section's persisted operand count first, locate each section's
segments below it via a running prefix sum of those counts, then rewind `cur_operand` by the
total.

It is block-based (`MakeBlockBased()`) so `PhysicalExpr::Size` amortises it across the
rowgroup. Without that, wizard selection — which encodes only a sample of vectors — would
charge the full header against a handful of vectors and unfairly penalise the encoding.

## Split selection

Unchanged by per-section codec selection ([Format](#format)): this DP still only ever measures
FFOR width when deciding where to split, exactly as it did before that existed. Mirrors Nimble's
own two-phase design in spirit — Nimble's split DP also uses a cheap multi-codec cost *estimate*
purely to steer where splits go, then a separate, later step (`encodeNested`, Nimble's ordinary
top-level encoder-selection machinery) picks each section's real codec on real per-section
statistics. This port keeps the DP's cost model to the one thing that's exactly, not
approximately, computable (FFOR width), and does the real per-section pick afterward the same
way Nimble does — on real data, via the same machinery the top-level wizard already uses.

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

**Now instance-dependent.** Before per-section codec selection existed, every section was FFOR,
so every access path below had one cost shape. Now a section can be Uncompressed, Constant, RLE,
Dictionary, FFOR, FFOR_SLPATCH, or FrequencyPartition, and only some of those support the O(1)
paths described here — see the per-path notes for which.

### Bulk

Per section, decode into scratch (an inline `unffor` for a plain-FFOR section; that section's own
child codec's normal decode path — `dec_rle_map_opr`, `dec_dict_opr`, etc. — for anything else),
then accumulate into the output — section 0 writes, later sections OR in, with the branch hoisted
out of the inner loop. Every codec is driven the same way regardless of which one a given section
resolved to, so this part of the design doesn't change shape as codecs vary.

Cost is linear in section count: 0.126 ms/rowgroup for 3 sections against 0.054 for plain
FFOR. **Bulk decode is where the compression is paid for.** (This figure predates per-section
codec selection; a non-FFOR section adds that codec's own decode cost on top, not just another
`unffor` pass — see [Results](#results) for current bulk numbers.)

### Point

**Only sound for a plain-FFOR or Constant section.** `unffor_single`
(`src/include/fls/primitive/unffor_single.hpp`) reaches one value by position arithmetic, so a
point read costs K scattered word reads instead of a decoded vector — but only when nothing (a
patch list, a dictionary indirection, a run-length map) sits between the packed bits and the
logical value. FFOR_SLPATCH's patch list makes a bare `unffor_single` read wrong for a patched
position, so it doesn't qualify either, despite being FFOR-shaped underneath. Every section that
doesn't qualify decodes its own vector once per `PointTo(vec_idx)` and caches it, amortizing over
however many rows are actually read out of that vector before the next `PointTo` call — cheaper
than decoding the whole SubIntSplit value, since only the disqualified sections pay it, but not
the scattered-word-read speed the paragraph below describes.

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

- `GatherPointwise` — reaches only the requested rows. For a plain-FFOR or Constant section,
  sections outermost so each section's bit width, base and pointer load once per gather rather
  than once per row. For every other section, reads out of the same per-`PointTo` decoded cache
  `PointAccess` uses, so a batch of rows against one vector still only costs that section one
  decode, not one per row.
- `GatherDecoded` — decodes the vector, then gathers.
- `Gather` — picks between them at `gather_decode_threshold` (default 128).

`GatherDecoded` is not just a fallback. It is the like-for-like comparison against Nimble's
`bulkScan`, which also decodes a whole span and gathers out of it. Within a vector, "decode
the span" and "decode the vector" are the same operation, because `unffor` cannot decode part
of one.

Measured crossover, at 2²⁰ rows with all indices pre-generated outside the timed loop. Predates
per-section codec selection — every section measured here was plain FFOR, so this describes the
scattered-read-vs-decode trade-off in the all-FFOR case only; a column with non-FFOR sections
today has a different (and generally lower, since non-FFOR sections already pay a per-vector
decode either way) effective crossover. Not re-measured for this change:

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
| ffor | 1.29× | 0.012 ms/rg | 0.292 µs |
| delta | 1.43× | 0.020 ms/rg | 1.875 µs |
| ffor_slpatch | 1.28× | 0.028 ms/rg | 0.636 µs |
| **subintsplit** | **1.71×** | 0.280 ms/rg | 9.480 µs |

The synthetic snowflake reference dataset, same row count, shows a larger margin:

| encoding | ratio | bulk decode | point (decode+index) |
|---|---|---|---|
| ffor | 2.11× | 0.018 ms/rg | 0.491 µs |
| delta | 2.17× | 0.031 ms/rg | 3.501 µs |
| ffor_slpatch | 2.11× | 0.020 ms/rg | 0.471 µs |
| **subintsplit** | **3.77×** | 0.215 ms/rg | 5.370 µs |

SubIntSplit's *native* point path on the real dataset reads 4.229 µs/probe — cheaper than
`point_decode_then_index` (9.480 µs) but nowhere near the FFOR-only-era numbers this section used
to quote, because the real dataset's plan (`bit_starts=[0,12,17,22]`) now resolves 3 of its 4
sections to non-FFOR codecs (`sis_layout`'s `bw_med=non_ffor;non_ffor;non_ffor` in the raw CSV):
only section 0 keeps the O(1) position-arithmetic path, the other three fall back to
decode-and-cache. See [Reading these numbers honestly](#reading-these-numbers-honestly) before
quoting any of this.

**The wizard picks SubIntSplit on its own**, on both datasets. On the real dataset, with a
default `Connection` it selects `EXP_SUBINTSPLIT_I64` (1.71×); with `disable_encoding` applied to
both SubIntSplit tokens it falls back to `EXP_DELTA_I64` at 1.43×. That ablation is what
`Connection::disable_encoding` exists for. (On the synthetic dataset the same ablation is 3.77×
against 2.16×.)

TPC-H partkey: the selector still chooses a single section that resolves to plain FFOR, same
shape as before per-section codec selection existed. IPv4 no longer degenerates as cleanly: its
single section now resolves to a non-FFOR codec (`sis_layout` reports `bw_med=non_ffor` for it)
at essentially the same ratio (~1.00×) FFOR alone gets, but loses FFOR's O(1) point/gather path in
the process — the wizard separately prefers `EXP_DICT_I32_FFOR_U16` for this column at ~1.01×
whether or not SubIntSplit is available, so SubIntSplit and Dictionary are, unsurprisingly,
converging on similar answers for the same data.

Encode is much slower now, on every dataset, for a structural reason rather than a regression:
choosing each section's codec runs the same measured-cost search
(`TryExpr`/`ChooseBestExpr`) the top-level wizard runs for a whole column, once per section, at
construction time. On the real dataset that's ~4.4s against FFOR's ~0.11s (about 40×, up from a
~7.3× DP-only gap before this existed); on the synthetic dataset ~1.0s against ~0.12s (about
8.5×, up from ~4.5×). Not optimized further here — the DP sweep itself is unchanged and still
cheap; the added cost is entirely the new per-section selection step, run once per column per
rowgroup, same as the DP always was.

### Reading these numbers honestly

**Bulk** is comparable across all encodings.

**Gather** is comparable only through `GatherDecoded`. `GatherPointwise` is a capability the
others do not have.

**Point access is not a like-for-like speed test, and is no longer a uniform one either.** Every
other encoding must decode the containing vector and index into it. The main table's `Point`
column reports exactly that for every row, including SubIntSplit — `point_decode_then_index` — so
that column *is* comparable. SubIntSplit's position-arithmetic path is reported separately, in
the native random-access table, precisely because it is a *capability* gap rather than a faster
implementation of the same operation — but that capability now only covers however many of a
given instance's sections are plain-FFOR-or-Constant, not all of them by construction the way it
did before per-section codec selection existed. Framings to keep in view:

- native point access on the real dataset (4.229 µs, 1 of 4 sections plain FFOR) against decoding
  the whole value (9.480 µs, all 4 sections): still faster, but by a much smaller margin than the
  11.4× this section used to report from an all-FFOR plan — 3 of the 4 sections now pay a decode
  either way, native or not;
- native point access against FFOR's own decode-then-index (0.292 µs): **slower** now, on the
  real dataset — the opposite of what this section used to say, and the direct cost of most
  sections no longer being FFOR;
- against a single-section FFOR column read *the same way*, splitting must still **lose** when
  every section genuinely is FFOR, because K scattered reads cost more than one — that part is
  unchanged. Whether splitting wins or loses on point access more generally now depends on the
  instance: it can still win when the DP's split happens to land on data where most sections
  resolve to FFOR/Constant, and can lose badly (as on the real dataset today) when it doesn't.
  Compression and point-access speed are no longer coupled the way "always FFOR" made them.

**Working set.** Benchmarks run at 2²⁰ rows — ~8 MB raw, ~2.5 MB compressed per column — which
exceeds L2 and stresses L3. Numbers taken earlier in this repo's history at 65 536 rows were
entirely cache-resident and are not comparable with these.

**Encode time is not like-for-like either.** Forced-encoding rows evaluate a single candidate;
the wizard rows search the whole pool plus the dictionary pool and run `Cast()` and every
pre-pass. The generated tables bold those two groups independently for this reason.

## Limited codec-set comparison

The tables above compare SubIntSplit against FastLanes' *entire* wizard pool — nine top-level
candidates plus Dictionary's own three-way index-width search, several of which (Delta,
RLE with an SLP patch, cross-row RLE) are more elaborate than a conservative encoder would
necessarily ship. `bench_subintsplit_limited` (same driver structure as `bench_subintsplit`,
sharing all of its measurement code via `BenchSubIntSplitCommon.hpp`) restricts the field to
seven simpler, more familiar codecs: **RLE, Constant, Dictionary, FFOR, `ffor_slpatch`,
Uncompressed, FrequencyPartition** — plus SubIntSplit itself. `ffor_slpatch` (FFOR plus an
exception patch list for outlier values) is included on the theory that it's the one member of
the excluded set closest in spirit to SubIntSplit's own per-section approach — both are trying to
handle a bit-range that's *almost* but not quite uniform. This isolates SubIntSplit's contribution
against codecs a smaller or more conservative encoder would actually have, rather than against
FastLanes' full, more elaborate candidate pool.

Mechanically: the per-codec forced rows (`uncompressed`, `rle`, `dict`, `ffor`, `ffor_slpatch`,
`frequency`, `subintsplit`) each use `Connection::force_schema_pool` with exactly that codec's
token(s); the two wizard rows (`wizard_limited_with_sis`/`wizard_limited_without_sis`) use
`Connection::disable_encoding` to remove everything from the default pool *outside* the
seven-codec set (`Delta`, `RLE` with an SLP patch, cross-row RLE — both i32 and i64 widths),
leaving the wizard to search only among the seven.

**Constant is not a forced row here**, for the same reason it originally motivated
`subintsplit_section_selector.hpp`'s own design: `force_schema_pool` cannot select
`EXP_CONSTANT_I32/I64` — it bypasses the wizard's poolable-candidate path entirely
(`constant_check` is an automatic structural pre-pass in `wizard.cpp` that runs *before* the
candidate pool is even consulted, and is skipped altogether once a pool is forced). None of this
benchmark's datasets are genuinely constant-valued at the top level, so a forced Constant row
would be a no-op or a build-time error either way. It remains a real member of the codec set for
the two wizard-limited rows, where it can still apply automatically to any column that happens to
qualify — it simply cannot be exercised as its own forced comparison row here. SubIntSplit's own
per-section selector doesn't have this limitation ([Format](#format)): it runs the same
`constant_visit` pre-pass directly on each section's own values rather than going through
`force_schema_pool`, so a genuinely constant *section* — as opposed to a constant top-level column
— is reachable, and does get picked when the DP happens to isolate one (see
`SubIntSplitSectionSelector.PicksConstant`/`PicksConstantI32` in
`test/src/unit_tests/subintsplit_section_selector_test.cpp`).

Full results: **[`tables/subintsplit_limited.md`](../tables/subintsplit_limited.md)**, generated
by the same `scripts/run_subintsplit_tables.sh` that produces the main table above. Raw numbers
land in `benchmark/result/subintsplit/subintsplit_limited.csv`. The same comparability caveats
from [Reading these numbers honestly](#reading-these-numbers-honestly) apply unchanged.

## Divergences from the Nimble implementation

| | Nimble | Here | Why |
|---|---|---|---|
| Section codec | per-section choice from Trivial/Dict/RLE/… | per-section choice from {Uncompressed, Constant, RLE, Dictionary, FFOR, FFOR_SLPATCH, FrequencyPartition} | closed, in spirit: `physical_operator`'s `sp<PhysicalExpr>` alternative lets a section carry a real nested child expression, chosen the same way Nimble's `encodeNested` chooses one — real measured cost, run after the split DP, not the DP's own model. Narrower in scope than Nimble's full codec zoo (no ALP/PFOR/Varint) |
| Section storage | narrowest type that fits | column's unsigned type | costs nothing in size (`bw * 1024 / 8` either way), only decode bandwidth |
| Cost model (splits) | seven closed-form models | measured FFOR widths | see [Split selection](#split-selection) |
| Cost model (per-section codec) | real statistics via `encodeNested`, same as top-level columns | real statistics via the same `TryExpr`/`ChooseBestExpr` oracle the top-level wizard uses | same design as Nimble's, reusing FastLanes' existing measured-cost machinery instead of duplicating it |
| Point access | `skip(n)` + `materialize(1)`, forward cursor | O(1) position arithmetic for plain-FFOR/Constant sections; decode-and-cache per `PointTo(vec_idx)` for everything else | FastLanes' fixed vectors make O(1) possible for the codecs where nothing sits between the packed bits and the value; Nimble's cursor is stateful and forward-only regardless of codec |
| Types | 32/64-bit ints and floats | `i64`, `i32` | scope |
| Nulls | supported | not supported | FastLanes handles nulls via separate `EXP_NULL_*` expressions |

**What's still open.** Narrow section storage types are a smaller, purely mechanical win on bulk
decode this port hasn't taken — the path most in need of one. The per-section codec set is also
narrower than Nimble's (seven codecs, matching `bench_subintsplit_limited`'s set, rather than
Nimble's full zoo including ALP/PFOR/Varint) — nothing structural prevents widening it, the
selector's candidate pool (`subintsplit_section_selector.hpp`) is just a fixed list today.

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
| `src/include/fls/expression/subintsplit_section_selector.hpp` | per-section codec selection (real measured cost) |
| `src/include/fls/wizard/wizard_internal.hpp` | exposes the wizard's `TryExpr`/`ChooseBestExpr`/`gather_statistics`/`constant_visit` for the section selector to reuse |
| `src/include/fls/expression/subintsplit_operator.hpp` | operator declarations, storage layout |
| `src/expression/subintsplit_operator.cpp` | encode, decode, point, gather, per-section decode-chain dispatch |
| `src/include/fls/primitive/unffor_single.hpp` | single-value unpack (plain-FFOR/Constant sections only) |
| `benchmark/bench_subintsplit/` | benchmark driver + shared harness (`BenchSubIntSplitCommon.hpp`) |
| `benchmark/bench_subintsplit_limited/` | limited codec-set benchmark driver |
| `scripts/run_subintsplit_tables.sh` | generate + build + run + render (both benchmark binaries) |
| `scripts/render_subintsplit_tables.py` | Markdown table generator |
| `tables/subintsplit.md` | generated comparison tables |
| `tables/subintsplit_limited.md` | generated limited codec-set comparison tables |
| `data/generated/subintsplit/` | datasets and generator |
