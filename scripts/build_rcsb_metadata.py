from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.manifest import write_manifest
from src.rcsb_api import get_entry, search_entries


POSITIVE_QUERY = {
    "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_polymer_entity.rcsb_ec_lineage.id",
            "operator": "exists",
        },
    },
    "return_type": "entry",
}


def text_phrase(attribute: str, value: str) -> dict:
    return {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": attribute,
                "operator": "contains_phrase",
                "value": value,
            },
        }
    }


def group_and(queries: list[dict]) -> dict:
    return {
        "type": "group",
        "logical_operator": "and",
        "nodes": [q["query"] for q in queries],
    }


def group_or(queries: list[dict]) -> dict:
    return {
        "type": "group",
        "logical_operator": "or",
        "nodes": [q["query"] for q in queries],
    }


def build_negative_query(keywords: list[str]) -> dict:
    # We deliberately exclude every entry having an EC annotation.
    no_ec = {
        "type": "group",
        "logical_operator": "not",
        "nodes": [POSITIVE_QUERY["query"]],
    }

    keyword_queries = [
        text_phrase("struct_keywords.pdbx_keywords", keyword)
        for keyword in keywords
    ]

    return {
        "query": group_and([
            {"query": no_ec},
            {"query": group_or(keyword_queries)},
        ]),
        "return_type": "entry",
    }


def compact_entry(entry: dict, pdb_id: str) -> dict:
    exptl = entry.get("exptl", [])
    method = "; ".join(sorted({item.get("method", "") for item in exptl if item.get("method")}))

    resolution = ""
    refine = entry.get("rcsb_entry_info", {})
    values = refine.get("resolution_combined") or []
    if values:
        resolution = values[0]

    ec_values = []
    for polymer in entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", []) or []:
        # The entry endpoint does not inline entity annotations. EC provenance is
        # therefore retained from the search criterion, not reconstructed here.
        _ = polymer

    title = entry.get("struct", {}).get("title", "")
    return {
        "pdb_id": pdb_id.upper(),
        "title": title,
        "experimental_method": method,
        "resolution": resolution,
    }


def enrich(ids: list[str], label: int, source: str) -> list[dict]:
    rows = []
    for i, pdb_id in enumerate(ids, start=1):
        try:
            entry = get_entry(pdb_id)
            row = compact_entry(entry, pdb_id)
            row.update({
                "label": label,
                "label_source": source,
                "ec_annotations": "EC annotation exists" if label == 1 else "",
            })
            rows.append(row)
        except Exception as exc:
            print(f"WARNING: failed to fetch {pdb_id}: {exc}")
        if i % 25 == 0:
            print(f"Enriched {i}/{len(ids)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a documented RCSB enzyme/non-enzyme pilot manifest.")
    parser.add_argument("--enzymes", type=int, default=100)
    parser.add_argument("--non-enzymes", type=int, default=100)
    parser.add_argument("--output", default="data/metadata.csv")
    parser.add_argument(
        "--negative-keyword",
        action="append",
        default=[
            "STRUCTURAL PROTEIN",
            "DNA-BINDING PROTEIN",
            "RNA-BINDING PROTEIN",
            "TRANSPORT PROTEIN",
        ],
        help="RCSB structure keyword used to define explicit non-enzymatic candidates. Repeat for multiple keywords.",
    )
    args = parser.parse_args()

    print(f"Querying {args.enzymes} enzyme entries...")
    enzyme_ids = search_entries(POSITIVE_QUERY, args.enzymes)

    print(f"Querying {args.non_enzymes} non-enzyme candidates...")
    negative_query = build_negative_query(args.negative_keyword)
    non_enzyme_ids = search_entries(negative_query, args.non_enzymes)

    # Enzyme set wins in the unlikely event that an entry appears in both.
    enzyme_set = set(enzyme_ids)
    non_enzyme_ids = [pid for pid in non_enzyme_ids if pid not in enzyme_set]

    print(f"Retrieved {len(enzyme_ids)} enzyme entries")
    print(f"Retrieved {len(non_enzyme_ids)} non-enzyme candidates")

    rows = enrich(enzyme_ids, 1, "RCSB EC-lineage exists")
    rows += enrich(
        non_enzyme_ids,
        0,
        "RCSB structure-keyword candidate + no EC lineage",
    )

    # Deterministic order and duplicate removal.
    deduped = OrderedDict()
    for row in rows:
        deduped[row["pdb_id"]] = row

    final_rows = list(deduped.values())
    write_manifest(final_rows, args.output)

    positives = sum(r["label"] == 1 for r in final_rows)
    negatives = sum(r["label"] == 0 for r in final_rows)
    print(f"Wrote {len(final_rows)} rows -> {args.output}")
    print(f"Label 1 (enzyme): {positives}")
    print(f"Label 0 (non-enzyme candidates): {negatives}")
    print("NOTE: label 0 is an RCSB-derived candidate class and requires validation before being described as ground truth.")


if __name__ == "__main__":
    main()
