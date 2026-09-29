from __future__ import annotations

import argparse
import re
from pathlib import Path


def scan_text_files(root: Path) -> list[tuple[Path, str]]:
    hits = []
    patterns = [re.compile(r"(?i)\bpdb\b"), re.compile(r"(?i)\b[a-z0-9]{4}\.pdb\b")]
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 50_000_000:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if any(p.search(text) for p in patterns):
            hits.append((path, text[:5000]))
    return hits


def audit(root: Path) -> None:
    print("=== PROTEINS provenance audit ===")
    print(f"Root: {root.resolve()}")

    try:
        from torch_geometric.datasets import TUDataset
    except Exception as exc:
        print("PyTorch Geometric is unavailable:", exc)
        print("Install with: pip install torch-geometric")
        return

    dataset = TUDataset(root=str(root), name="PROTEINS")

    print(f"Dataset: {dataset}")
    print(f"Graphs: {len(dataset)}")
    print(f"num_node_features: {dataset.num_node_features}")
    print(f"num_node_labels: {dataset.num_node_labels}")
    print(f"num_node_attributes: {dataset.num_node_attributes}")
    print(f"num_classes: {dataset.num_classes}")

    sample = dataset[0]
    print("\nFirst graph fields:")
    for key in sorted(sample.keys()):
        value = getattr(sample, key)
        shape = getattr(value, "shape", None)
        print(f"  {key}: {shape if shape is not None else type(value).__name__}")

    raw_dir = Path(dataset.raw_dir)
    processed_dir = Path(dataset.processed_dir)
    print(f"\nRaw directory: {raw_dir}")
    print(f"Processed directory: {processed_dir}")

    print("\nScanning dataset text files for PDB identifiers...")
    hits = scan_text_files(raw_dir)
    if not hits:
        print("NO explicit PDB IDs found in the raw PROTEINS files.")
    else:
        print(f"Found {len(hits)} text files containing possible PDB references:")
        for path, text in hits:
            print(f"  {path}")
            for line in text.splitlines():
                if re.search(r"(?i)\b[a-z0-9]{4}(?:\.pdb)?\b", line):
                    print("    ", line[:200])
                    break

    print("\nPROVENANCE VERDICT")
    print("------------------")
    if not hits:
        print("The standard PROTEINS TUDataset files do not provide a direct PDB-ID mapping.")
        print("Do NOT assign PDB IDs by graph order. That would create an unverified mapping.")
    else:
        print("Possible PDB references exist; inspect them manually before creating a mapping.")

    print("\nImportant dataset-semantics check:")
    print("The public PROTEINS benchmark is not a raw residue+C-alpha dataset.")
    print("Its published TUDataset metadata reports 1113 graphs, 2 classes, ~39 nodes/graph, and one node attribute.")
    print("For a raw-PDB residue graph, use the original D&D protein-structure benchmark provenance instead of assuming PROTEINS graph IDs map to PDB IDs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data")
    args = parser.parse_args()
    audit(Path(args.root))
