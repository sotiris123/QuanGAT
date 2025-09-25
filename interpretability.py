# interpretability.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from models import QuanGAT
from evaluation import evaluate_global
from loaders import load_dataset


# =========================
# Feature Dictionaries
# =========================
def get_feature_dict(dataset_name):
    if dataset_name.lower() == "ppi":
        return {i: f"GO/Expression_feature_{i}" for i in range(50)}
    elif dataset_name.lower() in ["ogbn-proteins", "ogbn_proteins"]:
        return {
            0: "Gene expression (tissue-specific)",
            1: "Protein abundance",
            2: "Interaction confidence score",
            3: "Gene co-expression signal",
            4: "Pathway membership",
            5: "Phylogenetic profile",
            6: "Ortholog frequency",
            7: "Functional domain similarity"
        }
    else:
        return None


# -------------------------
# 1. Saliency Map
# -------------------------
def compute_saliency(model, sample, mutation_dict, device="cpu",
                     savepath="results/saliency.png", topk=10, target_node=0):
    model.eval()

    x, edge_index = sample.x.to(device), sample.edge_index.to(device)
    x = x.clone().detach().requires_grad_(True)

    logits = model(x, edge_index)
    target_class = logits[target_node].argmax().item()

    grads = torch.autograd.grad(
        outputs=logits[target_node, target_class],
        inputs=x,
        retain_graph=False,
        allow_unused=True
    )[0]

    if grads is None:
        print(f"[Warning] No gradients for node {target_node}")
        return {}

    saliency = grads[target_node].detach().cpu().numpy().flatten()
    top_idx = np.argsort(-np.abs(saliency))[:topk]
    top_saliency = saliency[top_idx]
    top_labels = [mutation_dict.get(i, f"Feature_{i}") for i in top_idx]

    plt.figure(figsize=(12, 5))
    plt.bar(top_labels, top_saliency)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top-{topk} Features for Node {target_node} (class {target_class})")
    plt.ylabel("Gradient importance")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

    print(f"[Saved] Saliency map for node {target_node} -> {savepath}")
    return dict(zip(top_labels, top_saliency))


# -------------------------
# 2. Attention Heatmap
# -------------------------
def visualize_attention(model, data, device="cpu", savepath="results/attention.png", topk=10):
    model.eval()
    x, edge_index = data.x.to(device), data.edge_index.to(device)

    try:
        with torch.no_grad():
            _, (alpha,) = model.gat1(x, edge_index, return_attention_weights=True)
        alpha = alpha.cpu().numpy()
    except Exception as e:
        print(f"[Warning] Could not extract attention weights: {e}")
        return None

    G = nx.Graph()
    edges = edge_index.cpu().numpy().T
    for i, (u, v) in enumerate(edges):
        G.add_edge(u, v, weight=alpha[i])

    sorted_edges = sorted(G.edges(data=True), key=lambda x: -x[2]['weight'])[:topk]
    top_edges = [(u, v, d['weight']) for u, v, d in sorted_edges]

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    weights = [w for (_, _, w) in top_edges]
    nx.draw(
        G, pos, node_size=40, edgelist=[(u, v) for (u, v, _) in top_edges],
        edge_color=weights, edge_cmap=plt.cm.plasma, with_labels=False
    )
    plt.title(f"Top-{topk} Protein–Protein Interactions (by attention)")
    plt.savefig(savepath)
    plt.close()

    print(f"[Saved] Attention heatmap -> {savepath}")
    return top_edges


# -------------------------
# 3. Report Generator
# -------------------------
def generate_report(metrics, all_saliencies, top_edges,
                    saliency_paths, attention_path,
                    savepath="results/report.md"):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    with open(savepath, "w") as f:
        f.write("# Mutation Prediction Report (QuanGAT, Researcher-ready)\n\n")

        f.write("## Model Performance\n")
        f.write(f"- **F1-score:** {metrics['f1']:.4f}\n")
        f.write(f"- **AUROC:** {metrics['auroc']:.4f}\n")
        f.write(f"- **AUPRC:** {metrics['auprc']:.4f}\n\n")

        f.write("## Saliency Analysis (Top Nodes)\n")
        for node_id, sal_dict in all_saliencies.items():
            f.write(f"### Node {node_id}\n")
            for mut, score in sal_dict.items():
                f.write(f"- {mut}: {score:.4f}\n")
            f.write(f"\n![Saliency Node {node_id}]({saliency_paths[node_id]})\n\n")

        f.write("## Key Protein–Protein Interactions\n")
        if top_edges:
            for u, v, w in top_edges:
                f.write(f"- Protein {u} ↔ Protein {v}: attention={w:.4f}\n")
            f.write(f"\n![Attention Heatmap]({attention_path})\n\n")
        else:
            f.write("_No attention heatmap generated._\n")

    print(f"[Saved] Researcher report -> {savepath}")


# -------------------------
# 4. Example Usage
# -------------------------
if __name__ == "__main__":
    dataset_name = "ppi"   # change to "ogbn-proteins"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets, clients = load_dataset(dataset_name)
    if dataset_name == "ppi":
        test_dataset = datasets["test"]
        sample = test_dataset[0]
        n_features, n_classes = test_dataset.num_features, test_dataset.num_classes
    else:  # ogbn-proteins
        data = datasets["data"]
        sample = data
        n_features, n_classes = data.num_features, data.y.max().item() + 1

    model = QuanGAT(in_channels=n_features, hidden_channels=64,
                    out_channels=n_classes).to(device)

    ckpt_path = "results/model.pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[Loaded] Trained QuanGAT model from {ckpt_path}")
    else:
        print(f"[Warning] No trained model found at {ckpt_path}. Using untrained weights!")

    # Evaluate
    metrics = evaluate_global(model, sample, device=device)

    os.makedirs("results", exist_ok=True)
    attention_path = "results/attention.png"

    mutation_dict = get_feature_dict(dataset_name)

    # Compute saliency for top 5 nodes
    degrees = sample.edge_index[0].bincount()
    top_nodes = torch.topk(degrees, k=5).indices.tolist()

    all_saliencies, saliency_paths = {}, {}
    for node_id in top_nodes:
        sal_path = f"results/saliency_node{node_id}.png"
        sal_dict = compute_saliency(model, sample, mutation_dict,
                                    device=device, savepath=sal_path,
                                    target_node=node_id)
        all_saliencies[node_id] = sal_dict
        saliency_paths[node_id] = sal_path

    top_edges = visualize_attention(model, sample, device=device, savepath=attention_path)
    generate_report(metrics, all_saliencies, top_edges, saliency_paths, attention_path)
