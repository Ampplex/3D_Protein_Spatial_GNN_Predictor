from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

PDB_RE = re.compile(r"^[0-9A-Z]{4}$")


def parse_pdb_ids(line: str) -> list[str]:
    tokens = re.split(r"[\s,;]+", line.strip().upper())
    return [t for t in tokens if PDB_RE.fullmatch(t)]


def extract_section(text: str, start: str, end: str | None) -> list[str]:
    m = re.search(re.escape(start), text, flags=re.I)
    if not m:
        return []
    tail = text[m.end():]
    if end:
        e = re.search(re.escape(end), tail, flags=re.I)
        if e:
            tail = tail[:e.start()]
    return tail.splitlines()


def build(input_file: Path, output_file: Path) -> None:
    text = input_file.read_text(errors="ignore")

    # Intended for the original Dobson & Doig supplementary list:
    # 691 enzymes followed by 487 non-enzymes.
    enzyme_lines = extract_section(text, "691 Enzymes", "487 non-enzymes")
    nonenzyme_lines = extract_section(text, "487 non-enzymes", None)

    enzymes = []
    for line in enzyme_lines:
        enzymes.extend(parse_pdb_ids(line))

    nonenzymes = []
    for line in nonenzyme_lines:
        nonenzymes.extend(parse_pdb_ids(line))

    # Deduplicate while preserving order.
    enzymes = list(dict.fromkeys(enzymes))
    nonenzymes = list(dict.fromkeys(nonenzymes))

    rows = [(pdb, 1) for pdb in enzymes] + [(pdb, 0) for pdb in nonenzymes]

    if not rows:
        raise RuntimeError(
            "No PDB IDs were parsed. Check that the input is the original supplementary text and that the section headers are present."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_id", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_file}")
    print(f"Enzymes: {len(enzymes)}")
    print(f"Non-enzymes: {len(nonenzymes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--output", default="data/metadata_dd.csv")
    args = parser.parse_args()
    build(Path(args.input_file), Path(args.output))
