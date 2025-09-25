# evaluation.py
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

@torch.no_grad()
def evaluate_global(model, data, device="cpu"):
    model.eval()

    if isinstance(data, tuple):
        # Case: (ogbn_data, test_idx)
        pyg_data, idx = data
        x, edge_index, y = pyg_data.x.to(device), pyg_data.edge_index.to(device), pyg_data.y.to(device)
        logits = model(x, edge_index)
        logits, y = logits[idx], y[idx]

    else:
        # Case: PPI graph (no test_mask)
        x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
        logits = model(x, edge_index)

    y_true = y.cpu().numpy()
    y_pred = logits.argmax(dim=-1).cpu().numpy()

    # Handle multi-label (PPI) vs single-label (OGBN)
    if y_true.ndim > 1 and y_true.shape[1] > 1:  # multilabel case
        y_prob = torch.sigmoid(logits).cpu().numpy()
        f1 = f1_score(y_true, y_prob > 0.5, average="micro")
        auroc = roc_auc_score(y_true, y_prob, average="micro")
        auprc = average_precision_score(y_true, y_prob, average="micro")
    else:  # single-label classification
        y_prob = torch.softmax(logits, dim=-1).cpu().numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        try:
            auroc = roc_auc_score(y_true, y_prob[:, 1]) if y_prob.shape[1] > 1 else 0.0
        except Exception:
            auroc = 0.0
        try:
            auprc = average_precision_score(y_true, y_prob[:, 1]) if y_prob.shape[1] > 1 else 0.0
        except Exception:
            auprc = 0.0

    return {"f1": f1, "auroc": auroc, "auprc": auprc}
