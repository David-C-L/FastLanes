#!/usr/bin/env python3
"""Build FastLanes benchmark datasets from the shared column corpus.

The cross-format comparison needs the three frameworks to encode the *same
rows in the same order*. Before this script FastLanes drew its real Snowflake
column from a random offset and sorted it ascending, while BtrBlocks and Nimble
read the leading rows in file order -- so the FastLanes column was neither the
same rows nor the same arrangement, and its frame of reference had a
near-monotone sequence to work on that the others never saw.

Every dataset produced here follows one rule, shared with the BtrBlocks side
(`tools/subintsplit/make_cross_format_datasets.py`):

    take the LEADING `--rows` values of the source column, in FILE ORDER,
    then apply `--order` if a different arrival order is wanted.

Sources are the one-value-per-line `.txt` column dumps under
EncodingsPlayground/Datasets, which is the same corpus the Nimble ML-ID
benchmark reads through `--mlidc_file`, so all three harnesses start from
byte-identical input.

Usage
-----
    python3 scripts/make_cross_format_datasets.py --out-dir /tmp/xfmt --rows 1048576
    python3 scripts/make_cross_format_datasets.py --out-dir /tmp/xfmt --columns snowflake,publicbi_npi

Writes DIR/<name>/{generated.csv,schema.json,README.MD} plus a manifest at
DIR/manifest.csv for $FLS_SUBINTSPLIT_MANIFEST.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

# The corpus root, relative to this file: FastLanes/scripts/ -> MetaNimbleProject/
CORPUS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "EncodingsPlayground", "Datasets"))

# Canonical column name -> (source .txt, element type).
#
# The two Public BI entries are the ones the paper measures, and naming them
# here is the point: FastLanes' own Public BI results carry an `NPI` column in
# seven different tables and an `Id1` in one, with nothing recording which was
# meant. These two are what EncodingsPlayground/Datasets/PublicBI/README.md
# extracted, so they are the ones that correspond to `publicbi_npi` and
# `publicbi_id1` in our results, and any other table's NPI is a different
# column that would make the comparison meaningless rather than incomplete.
COLUMNS = {
    "snowflake": (
        "TwitterSnowflake/csv/tweet_id.txt", "INT64",
        "Real Twitter snowflake ids, leading rows in file order "
        "(the source is 99.96% descending)."),
    "xmark_prepost_full": (
        "XMark/csv/prepost_id_full.txt", "INT64",
        "XMark prepost traversal ids, full document."),
    "osm_h3_r9_fileorder": (
        "OSM/csv/h3_r9_fileorder.txt", "INT64",
        "OSM GB H3 resolution-9 cell ids, file order."),
    "osm_s2_l30_fileorder": (
        "OSM/csv/s2_l30_fileorder.txt", "INT64",
        "OSM GB S2 level-30 cell ids, file order."),
    "publicbi_npi": (
        "PublicBI/Medicare1/csv/NPI.txt", "INT32",
        "Public BI Medicare1.NPI, nulls dropped, first 2M rows in file order."),
    "publicbi_id1": (
        "PublicBI/Corporations/csv/Id1.txt", "INT32",
        "Public BI Corporations.Id1, file order."),
}

SCHEMA_TYPE = {"INT32": "FLS_I32", "INT64": "FLS_I64"}

RANGE = {"INT32": (-(2 ** 31), 2 ** 31 - 1), "INT64": (-(2 ** 63), 2 ** 63 - 1)}


def read_leading(path, rows):
    """The leading `rows` values, in file order. Never a random offset."""
    values = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values.append(int(line))
            if len(values) == rows:
                break
    return values


def arrange(values, order, seed):
    if order == "file":
        return values
    if order == "sorted":
        return sorted(values)
    if order == "shuffled":
        shuffled = list(values)
        random.Random(seed).shuffle(shuffled)
        return shuffled
    raise ValueError(f"unknown order: {order}")


def write_dataset(out_dir, name, values, element_type, description, order):
    target = os.path.join(out_dir, name)
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "generated.csv"), "w") as handle:
        handle.write("\n".join(str(value) for value in values))
        handle.write("\n")
    with open(os.path.join(target, "schema.json"), "w") as handle:
        json.dump({"columns": [{"name": "COLUMN_0",
                                "type": SCHEMA_TYPE[element_type]}]}, handle, indent=2)
    with open(os.path.join(target, "README.MD"), "w") as handle:
        handle.write(
            f"- {description}\n"
            f"- {len(values)} leading rows of the source column, taken in file order\n"
            f"- arrival order applied afterwards: {order}\n"
            f"- element type: {element_type}\n"
            "- produced by FastLanes/scripts/make_cross_format_datasets.py\n"
            "- the same rule produces the BtrBlocks and Nimble inputs, so all\n"
            "  three frameworks encode the same rows in the same order\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rows", type=int, default=1_048_576)
    parser.add_argument("--order", default="file",
                        choices=("file", "sorted", "shuffled"))
    parser.add_argument("--seed", type=int, default=42,
                        help="shuffle seed; unused for the other orders")
    parser.add_argument("--columns", default="",
                        help="comma-separated subset; empty means every column")
    args = parser.parse_args()

    wanted = ([name.strip() for name in args.columns.split(",") if name.strip()]
              or list(COLUMNS))
    unknown = [name for name in wanted if name not in COLUMNS]
    if unknown:
        sys.exit(f"unknown column(s): {', '.join(unknown)}")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = ["name,dir,type"]
    for name in wanted:
        relative, element_type, description = COLUMNS[name]
        source = os.path.join(CORPUS, relative)
        if not os.path.exists(source):
            print(f"-- skipping {name}: no source at {source}")
            continue
        values = read_leading(source, args.rows)
        if not values:
            print(f"-- skipping {name}: source is empty")
            continue
        low, high = RANGE[element_type]
        out_of_range = [value for value in values if not low <= value <= high]
        if out_of_range:
            # Truncating would silently change the column, and a column that is
            # not the paper's column makes the comparison meaningless.
            sys.exit(f"{name}: {len(out_of_range)} value(s) do not fit "
                     f"{element_type}, first is {out_of_range[0]}")
        values = arrange(values, args.order, args.seed)
        write_dataset(args.out_dir, name, values, element_type, description,
                      args.order)
        manifest.append(f"{name},{os.path.join(args.out_dir, name)},{element_type}")
        print(f"{name:24s} rows={len(values):>9,}  type={element_type}  "
              f"order={args.order}")

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    with open(manifest_path, "w") as handle:
        handle.write("\n".join(manifest) + "\n")
    print(f"\nmanifest: {manifest_path}")
    print(f"run with: FLS_SUBINTSPLIT_MANIFEST={manifest_path} ./bench_subintsplit")


if __name__ == "__main__":
    main()
