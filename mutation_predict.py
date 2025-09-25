"""
mutation_predict.py
-------------------
Researcher-facing tool:
1. DNA mutation predictions (CSV)
2. Saliency maps for nodes (PNG)
3. Attention heatmap for GAT (PNG)
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loaders import load_dataset
from models import QuanGAT


def safe_load_state_dict(model, checkpoint_path, device="cpu"):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()

    filtered_dict = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
        else:
            skipped.append(k)

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    if skipped:
        print(f"[Warning] Skipped incompatible keys: {skipped}")
    else:
        print(f"✅ All keys loaded successfully from {checkpoint_path}")


def run_prediction(dataset_name="ppi",
                   model_path="results/model.pt",
                   output_file="results/mutation_predictions.csv",
                   saliency_dir="results/saliency",
                   attention_file="results/attention.png",
                   attention_csv="results/attention_edges.csv",
                   device="cpu"):

    os.makedirs("results", exist_ok=True)
    os.makedirs(saliency_dir, exist_ok=True)

    # ---- Load dataset ----
    datasets, _ = load_dataset(dataset_name, num_clients=3)
    sample = datasets["test"][0] if dataset_name.lower() == "ppi" else datasets["data"]

    n_features = sample.num_features
    n_classes = sample.y.shape[1] if sample.y.dim() > 1 else int(sample.y.max().item()) + 1

    # ---- Load model ----
    model = QuanGAT(n_features, hidden_channels=64, out_channels=n_classes).to(device)
    if os.path.exists(model_path):
        safe_load_state_dict(model, model_path, device)
    else:
        print(f"[Warning] No trained model found at {model_path}. Using untrained weights!")

    model.eval()
    x, edge_index, y = sample.x.to(device), sample.edge_index.to(device), sample.y.to(device)

    # ---- Predictions ----
    with torch.no_grad():
        logits = model(x, edge_index)
        if n_classes > 1:
            preds = torch.softmax(logits, dim=-1)
            pred_labels = preds.argmax(dim=1).cpu().numpy()
            confidences = preds.max(dim=1).values.cpu().numpy()
        else:
            preds = torch.sigmoid(logits).squeeze()
            pred_labels = (preds > 0.5).long().cpu().numpy()
            confidences = preds.cpu().numpy()

    df = pd.DataFrame({
        "node_id": range(len(pred_labels)),
        "true_label": y.cpu().numpy().tolist(),
        "predicted_label": pred_labels.tolist(),
        "confidence": confidences.tolist()
    })
    df.to_csv(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")

    # ---- Saliency Maps (Top-10 most confident nodes) ----
    top_indices = confidences.argsort()[-10:][::-1]  # top-10 confident
    x.requires_grad = True
    logits = model(x, edge_index)

    for node_id in top_indices:
        target_class = logits[node_id].argmax()
        loss = logits[node_id, target_class]
        model.zero_grad()
        loss.backward(retain_graph=True)
        saliency = x.grad[node_id].detach().cpu().numpy()

        plt.figure(figsize=(10, 4))
        sns.barplot(x=list(range(len(saliency))), y=saliency)
        plt.title(f"Saliency map | Node {node_id} | Predicted class {target_class.item()}")
        plt.xlabel("Feature index")
        plt.ylabel("Gradient magnitude")
        saliency_path = os.path.join(saliency_dir, f"saliency_node{node_id}_class{target_class.item()}.png")
        plt.savefig(saliency_path)
        plt.close()
        print(f"✅ Saliency map saved to {saliency_path}")

    # ---- Attention Weights ----
    try:
        _, (attn1, attn2) = model(x, edge_index, return_attention=True)
        attn_weights = attn1[1].detach().cpu().numpy()  # (edge_index, weights)
        edges = attn1[0].cpu().numpy()

        # Save histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(attn_weights, bins=50, kde=True, color="blue")
        plt.title("Distribution of GAT attention weights")
        plt.xlabel("Attention weight")
        plt.ylabel("Frequency")
        plt.savefig(attention_file)
        plt.close()
        print(f"✅ Attention weights histogram saved to {attention_file}")

        # Save top edges
        df_edges = pd.DataFrame({
            "src_node": edges[0],
            "dst_node": edges[1],
            "attention_weight": attn_weights
        }).sort_values("attention_weight", ascending=False).head(50)
        df_edges.to_csv(attention_csv, index=False)
        print(f"✅ Top attention edges saved to {attention_csv}")

    except Exception as e:
        print(f"[Warning] Could not extract attention: {e}")


if __name__ == "__main__":
    run_prediction(dataset_name="ppi")





