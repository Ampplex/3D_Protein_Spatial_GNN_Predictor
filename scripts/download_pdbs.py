from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests


PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download(pdb_id: str, out_dir: Path, overwrite: bool = False) -> bool:
    pdb_id = pdb_id.strip().upper()
    target_pdb = out_dir / f"{pdb_id}.pdb"
    target_cif = out_dir / f"{pdb_id}.cif"

    if (target_pdb.exists() or target_cif.exists()) and not overwrite:
        return True

    response = requests.get(PDB_URL.format(pdb_id=pdb_id), timeout=60)
    if response.status_code == 200:
        target_pdb.write_text(response.text, encoding="utf-8")
        return True

    cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response_cif = requests.get(cif_url, timeout=60)
    response_cif.raise_for_status()
    target_cif.write_text(response_cif.text, encoding="utf-8")
    return True

    response = requests.get(PDB_URL.format(pdb_id=pdb_id), timeout=60)
    response.raise_for_status()
    target.write_text(response.text, encoding="utf-8")
    return True


def choose_ids(rows: list[dict[str, str]], limit: int | None, balanced: bool) -> list[str]:
    if limit is None:
        return [r["pdb_id"].strip().upper() for r in rows]
    if limit <= 0:
        return []

    if not balanced:
        return [r["pdb_id"].strip().upper() for r in rows[:limit]]

    by_label = {0: [], 1: []}
    for row in rows:
        try:
            label = int(row["label"])
        except (KeyError, ValueError):
            continue
        if label in by_label:
            by_label[label].append(row["pdb_id"].strip().upper())

    if not by_label[0] or not by_label[1]:
        raise ValueError(
            "Balanced download requested, but metadata.csv does not contain both labels."
        )

    # Deterministic round-robin: 0,1,0,1,...
    selected: list[str] = []
    i0 = i1 = 0
    while len(selected) < limit and (i0 < len(by_label[0]) or i1 < len(by_label[1])):
        if i0 < len(by_label[0]) and len(selected) < limit:
            selected.append(by_label[0][i0])
            i0 += 1
        if i1 < len(by_label[1]) and len(selected) < limit:
            selected.append(by_label[1][i1])
            i1 += 1

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download labelled PDB structures from an existing manifest."
    )
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--output-dir", default="data/raw/pdb")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of structures."
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="When --limit is used, select a deterministic class-balanced subset."
    )
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    ids = choose_ids(rows, args.limit, args.balanced)

    if args.balanced and args.limit is not None:
        counts = {0: 0, 1: 0}
        lookup = {r["pdb_id"].strip().upper(): int(r["label"]) for r in rows}
        for pdb_id in ids:
            counts[lookup[pdb_id]] += 1
        print(f"Balanced pilot selection: class0={counts[0]}, class1={counts[1]}")

    success = 0
    failed: list[tuple[str, str]] = []

    for i, pdb_id in enumerate(ids, start=1):
        try:
            download(pdb_id, output_dir, args.overwrite)
            success += 1
            print(f"[{i}/{len(ids)}] {pdb_id} OK")
        except requests.HTTPError as exc:
            failed.append((pdb_id, f"HTTP {exc.response.status_code if exc.response else 'error'}"))
            print(f"[{i}/{len(ids)}] {pdb_id} FAILED: {failed[-1][1]}")
        except Exception as exc:
            failed.append((pdb_id, str(exc)))
            print(f"[{i}/{len(ids)}] {pdb_id} FAILED: {exc}")
        time.sleep(args.sleep)

    print(f"Downloaded/available: {success}/{len(ids)}")
    if failed:
        print("Failures:")
        for pdb_id, error in failed:
            print(f"  {pdb_id}: {error}")


if __name__ == "__main__":
    main()
