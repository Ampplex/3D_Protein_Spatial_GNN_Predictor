# 🧬 3D Protein Spatial GNN Predictor

This project implements a complete, end-to-end Machine Learning pipeline that represents 3D protein structures as mathematical graphs and uses Graph Neural Networks (GNNs) to classify their biological function (Enzyme vs. Non-Enzyme). 

It was built as a Project-Based Learning (PBL) initiative to demonstrate data engineering, representation learning, spatial graph topologies, and ablation testing in computational biology.

## 🚀 Live Web App
The model is deployed via an interactive Streamlit dashboard. Users can upload raw `.pdb` or `.cif` files and visualize the 3D protein structure alongside the GNN's predictions in real-time.

## 🧠 Architecture Pipeline

1. **Data Acquisition:** Protein structures are downloaded directly from the RCSB Protein Data Bank in `.pdb` or `.cif` format.
2. **Residue Extraction:** The atomic structure is parsed to extract standard amino-acid residues based on their Alpha-Carbon (Cα) coordinates.
3. **Feature Engineering:** Each residue is assigned a 23-dimensional feature vector:
   - 20-D one-hot sequence identity
   - 3-D physicochemical properties (Hydrophobicity, Charge, Mass)
4. **Spatial Graph Construction:** A directed graph is built where nodes are residues, and spatial edges are drawn between any two residues whose Cα atoms are within an **8 Å** distance cutoff.
5. **Deep Learning:** The graph is passed through a Graph Neural Network (GAT, GraphSAGE, or GCN) with global mean pooling to predict the overarching protein class.

## 📊 Experimental Results

We evaluated three different architectures (GCN, GraphSAGE, GAT) on a custom dataset of 200 spatial protein graphs.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GCN** | 0.8500 | 0.7692 | 1.0000 | 0.8696 | 0.8900 |
| **GraphSAGE** | 0.9000 | 0.8333 | 1.0000 | 0.9091 | 0.8700 |
| **GAT** | 0.9000 | 0.8333 | 1.0000 | 0.9091 | 0.8900 |

*GraphSAGE and GAT tied for the highest performance, heavily benefiting from their ability to dynamically aggregate spatial neighborhoods.*

### Ablation Study
To prove *why* the model works, we conducted a rigorous ablation study isolating specific features and graph topologies:

1. **Baseline:** Full 23-features + 3D spatial edges
2. **A1 (Identity-only):** Only the 20-D sequence identity + 3D spatial edges
3. **A2 (Physicochemical-only):** Only the 3-D scalar properties + 3D spatial edges
4. **A3 (Sequence Graph):** Full 23-features + 1D sequence topology (edges only between i and i+1)
5. **A4 (Spatial + Distance):** Full 23-features + explicit geometric scalar distances applied as edge weights/attributes.

**Key Findings:** The ablation proved that the 20-D amino-acid identity (A1) drives the vast majority of the feature signal, completely outperforming pure physicochemical features (A2). Furthermore, stripping away the 3D folding topology (A3) degraded model confidence, proving that the explicit spatial representation is superior to a linear amino acid sequence sequence.

## 🛠️ Local Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GNN_protein.git
cd GNN_protein
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the interactive web dashboard:
```bash
export PYTHONPATH=.
streamlit run app.py
```
4. Run inference from the terminal:
```bash
export PYTHONPATH=.
python predict.py data/raw/pdb/101M.pdb --model GAT
```

## ⚠️ Limitations & Future Work
This PBL project trained on a pilot dataset of 200 heuristically-labeled proteins to prove pipeline viability. Because of the small dataset size, the model may over-index on the positive "Enzyme" class when presented with out-of-distribution structures in the wild. Future work involves scaling the training pipeline to 50,000+ curated protein graphs using distributed compute.
