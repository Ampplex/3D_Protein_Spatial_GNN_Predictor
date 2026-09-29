from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

REQUIRED = {"pdb_id", "label", "label_source"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata.csv")
    args = parser.parse_args()

    path = Path(args.metadata)
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise SystemExit("metadata.csv is empty")

    missing = REQUIRED - set(rows[0])
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    ids = [r["pdb_id"].strip().upper() for r in rows]
    labels = []
    for r in rows:
        try:
            label = int(r["label"])
        except ValueError:
            raise SystemExit(f"Invalid label for {r['pdb_id']}: {r['label']!r}")
        if label not in (0, 1):
            raise SystemExit(f"Label must be 0/1: {r['pdb_id']}")
        labels.append(label)

    duplicates = [pid for pid, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise SystemExit(f"Duplicate PDB IDs: {duplicates[:10]}")

    if not all(len(pid) == 4 for pid in ids):
        raise SystemExit("PDB IDs must be 4-character IDs")

    counts = Counter(labels)
    print(f"Rows: {len(rows)}")
    print(f"Enzymes: {counts[1]}")
    print(f"Non-enzymes: {counts[0]}")
    print("Manifest schema: OK")


if __name__ == "__main__":
    main()
