from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser


@dataclass(frozen=True)
class ResidueRecord:
    """One standard amino-acid residue represented by its C-alpha atom."""
    residue_id: str
    residue_name: str
    chain_id: str
    sequence_index: int
    coord: np.ndarray


# Standard 20 amino acids. Non-standard residues are ignored for this first phase.
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def extract_residues(
    pdb_path: str | Path,
    chain_id: str | None = None,
    model_index: int = 0,
) -> list[ResidueRecord]:
    """Extract standard residues and their C-alpha coordinates from a PDB file.

    A single chain may be selected with ``chain_id``. When ``chain_id`` is None,
    all chains are traversed and each residue keeps its chain identifier.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    if pdb_path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    models = list(structure.get_models())
    if model_index < 0 or model_index >= len(models):
        raise ValueError(
            f"model_index={model_index} is invalid; PDB contains {len(models)} model(s)"
        )

    model = models[model_index]
    records: list[ResidueRecord] = []
    global_index = 0

    for chain in model.get_chains():
        if chain_id is not None and chain.id != chain_id:
            continue

        chain_index = 0
        for residue in chain.get_residues():
            resname = residue.get_resname().strip().upper()
            hetflag, resseq, icode = residue.id

            # Keep standard amino-acid residues only.
            if resname not in AA3_TO_1 or hetflag != " ":
                continue
            if "CA" not in residue:
                continue

            coord = np.asarray(residue["CA"].coord, dtype=np.float32)
            residue_id = f"{chain.id}:{resseq}{icode.strip()}"
            records.append(
                ResidueRecord(
                    residue_id=residue_id,
                    residue_name=resname,
                    chain_id=chain.id,
                    sequence_index=chain_index,
                    coord=coord,
                )
            )
            chain_index += 1
            global_index += 1

    if not records:
        raise ValueError(f"No standard residues with C-alpha atoms found in {pdb_path}")

    return records
