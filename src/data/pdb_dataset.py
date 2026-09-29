from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .graph_builder import build_spatial_graph
from .pdb_parser import extract_residues


class PDBDataset(Dataset):
    """Labelled raw-PDB dataset converted into cached PyG graphs.

    Expected layout:
        root/
          metadata.csv
          raw/pdb/<pdb_id>.pdb
          processed/<pdb_id>.pt

    Only structures whose PDB file is actually present are included. This makes
    a limited pilot download safe without silently changing labels or ordering.
    """

    def __init__(
        self,
        root: str | Path,
        cutoff_angstrom: float = 8.0,
        chain_id: str | None = None,
        force_reprocess: bool = False,
        feature_mode: str = "baseline",
    ) -> None:
        self.root = Path(root)
        self.cutoff_angstrom = float(cutoff_angstrom)
        self.chain_id = chain_id
        self.force_reprocess = force_reprocess
        self.feature_mode = feature_mode

        if self.cutoff_angstrom <= 0:
            raise ValueError("cutoff_angstrom must be positive")

        self.metadata_path = self.root / "metadata.csv"
        self.raw_pdb_dir = self.root / "raw" / "pdb"
        self.processed_dir = self.root / f"processed_{self.feature_mode}"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        all_rows = self._read_metadata()
        self.metadata = [
            row for row in all_rows
            if (self.raw_pdb_dir / f"{row['pdb_id']}.pdb").exists() or (self.raw_pdb_dir / f"{row['pdb_id']}.cif").exists()
        ]

        missing = len(all_rows) - len(self.metadata)
        if missing:
            print(f"Structures available locally: {len(self.metadata)}/{len(all_rows)}")
            print(f"Structures not downloaded yet: {missing}")

        if not self.metadata:
            raise ValueError(
                f"No labelled PDB files found under {self.raw_pdb_dir}. "
                f"Metadata source: {self.metadata_path}"
            )

        self._ensure_processed()

    def _read_metadata(self) -> list[dict]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata manifest: {self.metadata_path}")

        rows: list[dict] = []
        with self.metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"pdb_id", "label"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(
                    "metadata.csv must contain at least: pdb_id,label"
                )

            seen: set[str] = set()
            for row in reader:
                pdb_id = (row.get("pdb_id") or "").strip().upper()
                label_raw = (row.get("label") or "").strip()
                if not pdb_id and not label_raw:
                    continue
                if len(pdb_id) != 4:
                    raise ValueError(f"Invalid PDB ID: {pdb_id!r}")
                try:
                    label = int(label_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid label for {pdb_id}: {label_raw!r}"
                    ) from exc
                if label not in (0, 1):
                    raise ValueError(f"Label for {pdb_id} must be 0 or 1")
                if pdb_id in seen:
                    raise ValueError(f"Duplicate PDB ID in metadata.csv: {pdb_id}")
                seen.add(pdb_id)

                clean = dict(row)
                clean["pdb_id"] = pdb_id
                clean["label"] = label
                rows.append(clean)

        return rows

    def _ensure_processed(self) -> None:
        for row in self.metadata:
            pdb_id = row["pdb_id"]
            output = self.processed_dir / f"{pdb_id}.pt"

            if output.exists() and not self.force_reprocess:
                continue

            pdb_path = self.raw_pdb_dir / f"{pdb_id}.pdb"
            cif_path = self.raw_pdb_dir / f"{pdb_id}.cif"
            if pdb_path.exists():
                path = pdb_path
            elif cif_path.exists():
                path = cif_path
            else:
                # This should only be possible if files disappear mid-run.
                print(f"Skipping {pdb_id}: missing PDB/CIF file")
                continue

            records = extract_residues(path, chain_id=self.chain_id)
            result = build_spatial_graph(
                records,
                cutoff_angstrom=self.cutoff_angstrom,
                feature_mode=self.feature_mode
            )

            data = result.data
            data.y = torch.tensor([row["label"]], dtype=torch.long)

            torch.save(data, output)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx):
        # Support numpy.int64 or torch tensor scalars passed by Subset/DataLoader
        try:
            idx = int(idx)
        except Exception:
            raise TypeError("PDBDataset indices must be convertible to int")
        
        row = self.metadata[idx]
        path = self.processed_dir / f"{row['pdb_id']}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Cached graph missing: {path}")
        return torch.load(path, weights_only=False)

    def rebuild(self) -> None:
        self.force_reprocess = True
        self._ensure_processed()
        self.force_reprocess = False
