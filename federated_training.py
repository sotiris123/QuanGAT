# federated_training.py
import copy
from typing import List
import torch
from torch_geometric.data import Data
from local_training import train_local
from evaluation import evaluate_global

def federated_train(
    model_class,
    clients: List[Data],
    global_rounds: int,
    in_channels: int,
    hidden: int,
    out_channels: int,
    lr: float = 0.01,
    local_epochs: int = 5,
    device: str = "cpu",
):
    example_y = clients[0].y
    multilabel = (example_y.dim() == 2)
    loss_fn = torch.nn.BCEWithLogitsLoss() if multilabel else torch.nn.CrossEntropyLoss()
    global_model = model_class(in_channels, hidden, out_channels).to(device)
    history = []

    for rnd in range(1, global_rounds + 1):
        local_states = []
        for c in clients:
            m = model_class(in_channels, hidden, out_channels).to(device)
            m.load_state_dict(copy.deepcopy(global_model.state_dict()))
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            train_local(m, c, opt, loss_fn, epochs=local_epochs, device=device)
            local_states.append(copy.deepcopy(m.state_dict()))
            del m

        new_state = copy.deepcopy(local_states[0])
        for k in new_state.keys():
            new_state[k] = sum(s[k] for s in local_states) / len(local_states)
        global_model.load_state_dict(new_state)

        # Average metrics across all clients
        all_metrics = [evaluate_global(global_model, c, device=device) for c in clients]
        metrics = {
            "f1": sum(m["f1"] for m in all_metrics) / len(all_metrics),
            "auroc": sum(m["auroc"] for m in all_metrics) / len(all_metrics),
            "auprc": sum(m["auprc"] for m in all_metrics) / len(all_metrics),
        }
        metrics["round"] = rnd
        history.append(metrics)
        print(f"Round {rnd:03d} | F1={metrics['f1']:.4f} | AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f}")

    return history
