# ────────────────────────────────────────────────────────
# |                      FastLanes                       |
# ────────────────────────────────────────────────────────
# scripts/render_subintsplit_tables.py
# ────────────────────────────────────────────────────────
#!/usr/bin/env python3
"""Render the SubIntSplit benchmark CSV into grouped Markdown comparison tables.

Reads the tidy/long benchmark CSV produced by the SubIntSplit benchmark and
emits one Markdown section per dataset: a main comparison table (compression,
encode, bulk decode, the gather sweep and point access), a secondary table for
SubIntSplit's native random-access paths, and footnotes explaining what is and
is not a like-for-like comparison.

Standard library only, on purpose: the repository declares no Python runtime
dependencies and the other result-processing scripts under `benchmark/result/`
are stdlib-only too.

Usage:
    python3 scripts/render_subintsplit_tables.py                 # write tables/subintsplit.md
    python3 scripts/render_subintsplit_tables.py --stdout        # print instead
    python3 scripts/render_subintsplit_tables.py --check         # CI staleness check
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "benchmark" / "result" / "subintsplit" / "subintsplit.csv"
DEFAULT_OUT = REPO_ROOT / "tables" / "subintsplit.md"

REGEN_COMMAND = "scripts/run_subintsplit_tables.sh"

MISSING = "–"  # en dash
DIVIDER_CELL = "━━━"  # heavy horizontal, U+2501

#######################################################################
# Contract: datasets, groups, row keys, gather widths
#######################################################################

# Presentation order. Anything the CSV mentions that is not listed here is kept
# and appended after the known entries rather than silently dropped.
DATASET_ORDER = ["snowflake_i64_real", "snowflake_i64", "tpch_partkey_i32", "ipv4_i32"]
GROUP_ORDER = ["baseline", "codec", "wizard"]

# The baseline group is never bolded: an uncompressed row "winning" a decode
# column is trivially true and reads as a false claim.
UNBOLDED_GROUPS = {"baseline"}

ENCODING_ORDER = {
    "baseline": ["uncompressed"],
    "codec": ["ffor", "delta", "ffor_slpatch", "subintsplit"],
    "wizard": ["wizard_with_sis", "wizard_without_sis"],
}

# Fallback used when the CSV predates the `group` column.
FALLBACK_GROUP_OF = {
    enc: group for group, encs in ENCODING_ORDER.items() for enc in encs
}
FALLBACK_GROUP = "other"

ROW_LABEL = {
    "uncompressed": "uncompressed",
    "ffor": "ffor",
    "delta": "delta",
    "ffor_slpatch": "ffor_slpatch",
    "subintsplit": "subintsplit",
    "wizard_with_sis": "FastLanes wizard (SIS available)",
    "wizard_without_sis": "FastLanes wizard (SIS disabled)",
}

DATASET_LABEL = {
    "snowflake_i64_real": "snowflake_i64 (real)",
    "snowflake_i64": "snowflake_i64 (synthetic reference)",
    "tpch_partkey_i32": "tpch_partkey_i32",
    "ipv4_i32": "ipv4_i32",
}

GATHER_WIDTHS = [1, 4, 16, 64, 256, 1024]

# Metric names that changed after the first CSV revision. Keeping the aliases
# lets the script render the older committed CSV without special-casing.
METRIC_ALIASES = {
    "point_via_bulk": "point_decode_then_index",
    "sections": "sis_layout",
}

#######################################################################
# Formatters
#######################################################################


def _fmt(digits: int):
    def f(value: float) -> str:
        return f"{value:.{digits}f}"

    return f


fmt_ratio = _fmt(2)
fmt_ms = _fmt(1)
fmt_ms3 = _fmt(3)
fmt_us = _fmt(3)

# The single declarative column table. Adding a metric to the main table is one
# line here; nothing else in the file needs to know about it.
#   (metric, detail, header, direction, formatter)
# direction: "max" = higher is better, "min" = lower is better, None = no bold.
COLUMNS = [
    ("compression_ratio", 0, "Ratio (×)", "max", fmt_ratio),
    ("encode_time", 0, "Encode (ms)", "min", fmt_ms),
    ("bulk_decode", 0, "Bulk (ms/rg)", "min", fmt_ms3),
] + [
    ("gather_decode_then_index", n, f"Gather n={n}", "min", fmt_us)
    for n in GATHER_WIDTHS
] + [
    ("point_decode_then_index", 0, "Point (µs)", "min", fmt_us),
]

# Secondary table: SubIntSplit's native random-access paths.
NATIVE_COLUMNS = [
    ("gather_pointwise", "Gather pointwise (µs/gather)", fmt_us),
    ("gather_decoded", "Gather decoded (µs/gather)", fmt_us),
]

#######################################################################
# Loading
#######################################################################


class Table:
    """Everything read out of the CSV, keyed for rendering."""

    def __init__(self) -> None:
        # (dataset, encoding) -> {(metric, detail): (value, unit, note)}
        self.cells: dict = {}
        # (dataset, encoding) -> group name
        self.group_of: dict = {}
        # dataset -> ordered set of encodings, in first-seen order
        self.encodings: "OrderedDict[str, OrderedDict]" = OrderedDict()

    def add(self, dataset, encoding, group, metric, detail, value, unit, note):
        row_key = (dataset, encoding)
        self.encodings.setdefault(dataset, OrderedDict())[encoding] = True
        self.group_of[row_key] = group
        cell_key = (metric, detail)
        bucket = self.cells.setdefault(row_key, {})
        if cell_key in bucket:
            warn(
                f"duplicate metric {metric!r} (detail={detail}) for "
                f"{dataset}/{encoding}: last one wins"
            )
        bucket[cell_key] = (value, unit, note)

    def get(self, dataset, encoding, metric, detail=0):
        return self.cells.get((dataset, encoding), {}).get((metric, detail))

    def value(self, dataset, encoding, metric, detail=0):
        cell = self.get(dataset, encoding, metric, detail)
        return None if cell is None else cell[0]

    def note(self, dataset, encoding, metric, detail=0):
        cell = self.get(dataset, encoding, metric, detail)
        return None if cell is None else cell[2]


def warn(message: str) -> None:
    print(f"render_subintsplit_tables: warning: {message}", file=sys.stderr)


def load(csv_path: Path) -> Table:
    table = Table()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{csv_path}: empty CSV")
        fields = {name.strip() for name in reader.fieldnames}
        for required in ("dataset", "encoding", "metric", "value"):
            if required not in fields:
                raise SystemExit(f"{csv_path}: missing required column {required!r}")
        has_group = "group" in fields
        has_note = "note" in fields
        if not has_group:
            warn(
                f"{csv_path}: no 'group' column; inferring groups from encoding names"
            )
        if not has_note:
            warn(f"{csv_path}: no 'note' column; schema/type/layout details omitted")

        for lineno, record in enumerate(reader, start=2):
            if record.get("dataset") is None or not record["dataset"].strip():
                continue
            dataset = record["dataset"].strip()
            encoding = (record.get("encoding") or "").strip()
            metric = (record.get("metric") or "").strip()
            metric = METRIC_ALIASES.get(metric, metric)
            note = (record.get("note") or "").strip() if has_note else ""
            unit = (record.get("unit") or "").strip()

            group = (record.get("group") or "").strip() if has_group else ""
            if not group:
                group = FALLBACK_GROUP_OF.get(encoding, FALLBACK_GROUP)

            try:
                value = float(record.get("value") or "nan")
            except ValueError:
                warn(f"{csv_path}:{lineno}: unparsable value {record.get('value')!r}")
                value = float("nan")
            try:
                detail = int(float(record.get("detail") or 0))
            except ValueError:
                detail = 0

            table.add(dataset, encoding, group, metric, detail, value, unit, note)
    return table


#######################################################################
# Row ordering
#######################################################################


def ordered_datasets(table: Table) -> list:
    seen = list(table.encodings.keys())
    known = [d for d in DATASET_ORDER if d in seen]
    extra = [d for d in seen if d not in DATASET_ORDER]
    return known + extra


def ordered_groups(table: Table, dataset: str) -> list:
    present = []
    for encoding in table.encodings[dataset]:
        group = table.group_of[(dataset, encoding)]
        if group not in present:
            present.append(group)
    known = [g for g in GROUP_ORDER if g in present]
    extra = [g for g in present if g not in GROUP_ORDER]
    return known + extra


def ordered_encodings(table: Table, dataset: str, group: str) -> list:
    members = [
        enc
        for enc in table.encodings[dataset]
        if table.group_of[(dataset, enc)] == group
    ]
    preferred = ENCODING_ORDER.get(group, [])
    known = [e for e in preferred if e in members]
    # Unknown row keys are appended after the known ones instead of dropped.
    extra = [e for e in members if e not in preferred]
    return known + extra


#######################################################################
# Cell formatting and per-group winner selection
#######################################################################


def is_finite(value) -> bool:
    return value is not None and isinstance(value, float) and math.isfinite(value)


def format_cell(table, dataset, encoding, metric, detail, formatter):
    """Return the display string, or None when the cell is absent/non-finite."""
    value = table.value(dataset, encoding, metric, detail)
    if not is_finite(value):
        return None
    return formatter(value)


def winners_for(formatted_by_encoding: dict, direction: str) -> set:
    """Pick the winning encodings by comparing the *formatted* strings.

    Comparing displayed precision rather than raw floats is deliberate: two
    cells that both print `2.19` must either both be bold or neither, otherwise
    the reader sees a bug. A group with fewer than two eligible cells gets no
    bold at all, because marking a sole candidate as the winner is meaningless.
    """
    eligible = {enc: text for enc, text in formatted_by_encoding.items() if text}
    if len(eligible) < 2 or direction is None:
        return set()
    keyed = {}
    for enc, text in eligible.items():
        try:
            keyed[enc] = float(text)
        except ValueError:
            continue
    if len(keyed) < 2:
        return set()
    best = max(keyed.values()) if direction == "max" else min(keyed.values())
    return {enc for enc, number in keyed.items() if number == best}


def escape(text: str) -> str:
    return text.replace("|", "\\|")


def find_any_detail(table: Table, dataset: str, encoding: str, metric: str):
    """Return (detail, cell) for *metric* whatever its `detail` happens to be."""
    for (name, detail), cell in table.cells.get((dataset, encoding), {}).items():
        if name == metric:
            return detail, cell
    return 0, None


def encoding_used_cell(table: Table, dataset: str, encoding: str) -> str:
    schema = table.note(dataset, encoding, "schema")
    data_type = table.note(dataset, encoding, "data_type")
    parts = []
    if schema:
        parts.append(schema)
    if data_type:
        parts.append(f"({data_type})" if parts else data_type)
    head = " ".join(parts)

    detail, layout_cell = find_any_detail(table, dataset, encoding, "sis_layout")
    layout = ""
    if layout_cell is not None:
        value, _unit, note = layout_cell
        # The section count lives in `detail`; the older CSV put it in `value`
        # under the metric name `sections`.
        count = detail if detail else (int(value) if is_finite(value) else None)
        starts = ""
        for chunk in (note or "").split("|"):
            if chunk.startswith("starts="):
                starts = chunk
                break
        pieces = []
        if count:
            pieces.append(f"{count} section{'s' if count != 1 else ''}")
        if starts:
            pieces.append(starts)
        layout = " ".join(pieces)

    cell = ", ".join(part for part in (head, layout) if part)
    return escape(cell) if cell else MISSING


#######################################################################
# Rendering
#######################################################################


def render_main_table(table: Table, dataset: str) -> list:
    groups = ordered_groups(table, dataset)
    rows_by_group = OrderedDict(
        (group, ordered_encodings(table, dataset, group)) for group in groups
    )

    # Per group, per column: which encodings are bold.
    bold = {}
    for group, encodings in rows_by_group.items():
        for metric, detail, _header, direction, formatter in COLUMNS:
            formatted = {
                enc: format_cell(table, dataset, enc, metric, detail, formatter)
                for enc in encodings
            }
            if group in UNBOLDED_GROUPS:
                continue
            for enc in winners_for(formatted, direction):
                bold[(enc, metric, detail)] = True

    headers = ["Encoding"] + [column[2] for column in COLUMNS] + ["Encoding used"]
    align = [":---"] + ["---:" for _ in COLUMNS] + [":---"]

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]

    first_group = True
    for group, encodings in rows_by_group.items():
        if not encodings:
            continue
        if not first_group:
            lines.append("| " + " | ".join([DIVIDER_CELL] * len(headers)) + " |")
        first_group = False
        for encoding in encodings:
            cells = [ROW_LABEL.get(encoding, encoding)]
            for metric, detail, _header, _direction, formatter in COLUMNS:
                text = format_cell(table, dataset, encoding, metric, detail, formatter)
                if text is None:
                    cells.append(MISSING)
                elif bold.get((encoding, metric, detail)):
                    cells.append(f"**{text}**")
                else:
                    cells.append(text)
            cells.append(encoding_used_cell(table, dataset, encoding))
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_native_table(table: Table, dataset: str) -> list:
    """SubIntSplit's own random-access paths, which the others do not have."""
    sis_rows = [
        enc
        for enc in table.encodings[dataset]
        if any(
            table.get(dataset, enc, metric, n) is not None
            for metric, _h, _f in NATIVE_COLUMNS
            for n in GATHER_WIDTHS
        )
        or table.get(dataset, enc, "point_access") is not None
    ]
    if not sis_rows:
        return []

    lines = ["### SubIntSplit native random access", ""]
    lines.append(
        "These are paths SubIntSplit *has* and the other encodings do not. They are a"
        " capability gap, not a like-for-like speed win: every other encoding reaches a"
        " single value by decoding the containing vector and indexing into it, so the"
        " comparable numbers are the `Point` and `Gather` columns of the main table."
        " `Gather decoded` is the only row here that is measured the same way as those."
    )
    lines.append("")

    for encoding in sis_rows:
        point = format_cell(table, dataset, encoding, "point_access", 0, fmt_us)
        label = ROW_LABEL.get(encoding, encoding)
        lines.append(
            f"`{label}` native point access (position arithmetic, no vector decode): "
            f"{point if point else MISSING} µs/probe"
        )
        lines.append("")
        headers = ["n"] + [header for _m, header, _f in NATIVE_COLUMNS]
        align = ["---:"] * len(headers)
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(align) + " |")
        for n in GATHER_WIDTHS:
            cells = [str(n)]
            for metric, _header, formatter in NATIVE_COLUMNS:
                text = format_cell(table, dataset, encoding, metric, n, formatter)
                cells.append(text if text is not None else MISSING)
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


FOOTNOTES = [
    "**Encode time is not like-for-like.** The forced codec rows evaluate a single"
    " candidate encoding; the wizard rows search the whole encoding pool and run"
    " `Cast()` and every pre-pass. Compare forced rows with forced rows and wizard"
    " rows with wizard rows, which is why winners are computed per group.",
    "**A wizard win can come from narrowing, not from the encoding.** Wizard rows may"
    " narrow the physical type via `Cast()`, so part of a compression win can be the"
    " narrower type rather than the encoding choice. The post-`Cast()` physical type is"
    " therefore shown in *Encoding used*.",
    "**Point access is a capability gap.** Encodings other than SubIntSplit must decode"
    " the containing vector and index into it; the `Point` column reports that for them"
    " and the same decode-then-index path for SubIntSplit, so the column stays"
    " comparable. SubIntSplit's own position-arithmetic path is in the secondary table"
    " and is not the same operation.",
    "**The working set exceeds L2.** At 2^20 rows the raw column is about 8 MB, so none"
    " of these measurements are served out of L2.",
    f"Missing measurements are shown as `{MISSING}`. Bold marks the best value within a"
    " group for that column, chosen by comparing the values *as displayed*, so ties at"
    " display precision are bold together; a column with fewer than two eligible values"
    " in a group is left unbolded, and the baseline group is never bolded.",
]


def render(table: Table) -> str:
    lines = [
        "# SubIntSplit benchmark tables",
        "",
        f"<!-- Generated by `scripts/render_subintsplit_tables.py`."
        f" Regenerate with `{REGEN_COMMAND}`. Do not edit by hand. -->",
        "",
        f"Regenerate with `{REGEN_COMMAND}`.",
        "",
    ]

    for dataset in ordered_datasets(table):
        lines.append(f"## {DATASET_LABEL.get(dataset, dataset)}")
        lines.append("")
        lines.extend(render_main_table(table, dataset))
        lines.append("")
        native = render_native_table(table, dataset)
        if native:
            lines.extend(native)

    lines.append("## Notes")
    lines.append("")
    for note in FOOTNOTES:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


#######################################################################
# Entry point
#######################################################################


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="input CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output Markdown")
    parser.add_argument(
        "--stdout", action="store_true", help="print the render instead of writing it"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the render differs from what is on disk",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"render_subintsplit_tables: {args.csv}: no such file", file=sys.stderr)
        return 2

    text = render(load(args.csv))

    if args.stdout:
        sys.stdout.write(text)
        return 0

    if args.check:
        if not args.out.exists():
            print(
                f"render_subintsplit_tables: {args.out} does not exist;"
                f" run {REGEN_COMMAND}",
                file=sys.stderr,
            )
            return 1
        if args.out.read_text(encoding="utf-8") != text:
            print(
                f"render_subintsplit_tables: {args.out} is stale;"
                f" run {REGEN_COMMAND}",
                file=sys.stderr,
            )
            return 1
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
