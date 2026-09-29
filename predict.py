import argparse
from pathlib import Path
import torch

from src.data.pdb_parser import extract_residues
from src.data.graph_builder import build_spatial_graph
from src.models.structural_gnns import StructuralGNN

def main():
    parser = argparse.ArgumentParser(description="Predict if a PDB/CIF structure is an enzyme")
    parser.add_argument("input_file", type=str, help="Path to .pdb or .cif file")
    parser.add_argument("--model", type=str, default="GAT", choices=["GCN", "GraphSAGE", "GAT"])
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    checkpoint_path = Path(f"checkpoints/{args.model.lower()}_structural.pt")
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint {checkpoint_path} not found.")
        return
        
    print(f"Loading {args.model} checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model = StructuralGNN(
        model_type=checkpoint["model_type"],
        in_channels=checkpoint["input_dim"],
        hidden_channels=checkpoint["hidden_channels"],
        embedding_dim=checkpoint["embedding_dim"],
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Parsing {input_path.name}...")
    records = extract_residues(input_path)
    
    cutoff = checkpoint.get("cutoff_angstrom", 8.0)
    res = build_spatial_graph(records, cutoff_angstrom=cutoff, feature_mode="baseline")
    data = res.data
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    
    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = logits.argmax(dim=1).item()
        
    class_names = {0: "Non-Enzyme", 1: "Enzyme"}
    
    print("\n" + "="*40)
    print(f"Prediction for {input_path.name}:")
    print(f"Result: {class_names[pred_class]}")
    print(f"Confidence:")
    print(f"  Non-Enzyme: {probs[0]:.2%}")
    print(f"  Enzyme:     {probs[1]:.2%}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
