import streamlit as st
import torch
import py3Dmol
from streamlit.components.v1 import html
import tempfile
from pathlib import Path
import sys

# Ensure local imports work
sys.path.append(str(Path(__file__).parent))

from src.data.pdb_parser import extract_residues
from src.data.graph_builder import build_spatial_graph
from src.models.structural_gnns import StructuralGNN

st.set_page_config(page_title="Protein GNN Predictor", layout="wide", page_icon="🧬")

@st.cache_resource
def load_model(model_name):
    checkpoint_path = Path(f"checkpoints/{model_name.lower()}_structural.pt")
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
    return model, checkpoint.get("cutoff_angstrom", 8.0)

def render_protein(pdb_data, file_format):
    view = py3Dmol.view(width=800, height=500)
    fmt = "cif" if file_format.lower() == ".cif" else "pdb"
    view.addModel(pdb_data, fmt)
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.setBackgroundColor('white')
    view.zoomTo()
    return view._make_html()

st.title("🧬 3D Protein GNN Predictor")
st.markdown("Upload a protein structure (PDB/CIF) and use our custom Spatial GNN to classify its biological function!")

col1, col2 = st.columns([1, 1])

with col1:
    model_choice = st.selectbox("Select GNN Architecture", ["GAT", "GraphSAGE", "GCN"])
    uploaded_file = st.file_uploader("Upload Protein File (.pdb or .cif)", type=['pdb', 'cif'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    file_contents = uploaded_file.getvalue().decode("utf-8")
    suffix = Path(uploaded_file.name).suffix
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_contents.encode("utf-8"))
        tmp_path = Path(tmp.name)
        
    with st.spinner('Parsing 3D structure and building spatial graph...'):
        try:
            records = extract_residues(tmp_path)
            model, cutoff = load_model(model_choice)
            res = build_spatial_graph(records, cutoff_angstrom=cutoff, feature_mode="baseline")
            data = res.data
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            
            with torch.no_grad():
                logits = model(data)
                probs = torch.softmax(logits, dim=1)[0]
                pred_class = logits.argmax(dim=1).item()
                
            class_names = {0: "Non-Enzyme", 1: "Enzyme"}
            color = "green" if pred_class == 1 else "gray"
            
            with col1:
                st.subheader("Prediction Results")
                st.markdown(f"### **Prediction: :{color}[{class_names[pred_class]}]**")
                
                st.write("**Confidence Scores:**")
                st.progress(float(probs[1]))
                st.write(f"Enzyme: {probs[1]:.2%}")
                
                st.progress(float(probs[0]))
                st.write(f"Non-Enzyme: {probs[0]:.2%}")
                
                st.info(f"**Graph Stats:** Built a 3D graph with **{data.num_nodes} nodes** (residues) and **{data.edge_index.size(1)} spatial edges** (< 8 Å).")
                
            with col2:
                st.subheader("3D Structure Visualization")
                html_code = render_protein(file_contents, suffix)
                html(html_code, height=500)
                
        except Exception as e:
            st.error(f"Error processing structure: {e}")
