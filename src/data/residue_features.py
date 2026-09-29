from __future__ import annotations

import numpy as np

from .pdb_parser import AA3_TO_1, ResidueRecord

AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Kyte-Doolittle hydropathy values.
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Simple residue-level charge at approximately neutral pH.
CHARGE = {
    "A": 0.0, "R": 1.0, "N": 0.0, "D": -1.0, "C": 0.0,
    "Q": 0.0, "E": -1.0, "G": 0.0, "H": 0.1, "I": 0.0,
    "L": 0.0, "K": 1.0, "M": 0.0, "F": 0.0, "P": 0.0,
    "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0, "V": 0.0,
}

# Average residue molecular masses in Da.
MASS = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.15,
    "Q": 146.14, "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}


def residue_feature(record: ResidueRecord, mode: str = "baseline") -> np.ndarray:
    """Return the 23-D feature vector used by the structural phase.

    20-D one-hot amino-acid identity + hydrophobicity + charge + mass.
    """
    aa = AA3_TO_1[record.residue_name]
    one_hot = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    one_hot[AA_TO_INDEX[aa]] = 1.0

    # Normalize the three scalar features to more stable ranges for training.
    hydrophobicity = HYDROPATHY[aa] / 4.5
    charge = CHARGE[aa]
    mass = MASS[aa] / 200.0

    if mode == "A1":
        return one_hot.astype(np.float32)
    elif mode == "A2":
        return np.asarray([hydrophobicity, charge, mass], dtype=np.float32)
    else:
        return np.concatenate(
            [one_hot, np.asarray([hydrophobicity, charge, mass], dtype=np.float32)]
        ).astype(np.float32)


def build_feature_matrix(records: list[ResidueRecord], mode: str = "baseline") -> np.ndarray:
    if not records:
        raise ValueError("Cannot build features from an empty residue list")
    return np.stack([residue_feature(r, mode) for r in records], axis=0)
