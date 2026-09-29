from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

FIELDS = [
    "pdb_id",
    "label",
    "label_source",
    "ec_annotations",
    "title",
    "experimental_method",
    "resolution",
]


def write_manifest(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})
