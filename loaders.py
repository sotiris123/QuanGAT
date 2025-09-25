# loaders.py
import torch
from torch_geometric.datasets import PPI
from ogb.nodeproppred import PygNodePropPredDataset

def load_dataset(dataset_name, num_clients=3):
    if dataset_name.lower() == "ppi":
        train_dataset = PPI(root="data/PPI", split="train")
        val_dataset = PPI(root="data/PPI", split="val")
        test_dataset = PPI(root="data/PPI", split="test")

        data = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset])
        clients = [train_dataset, val_dataset, test_dataset][:num_clients]

        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}, clients

    elif dataset_name.lower() in ["ogbn-proteins", "ogbn_proteins"]:
        dataset = PygNodePropPredDataset(name="ogbn-proteins", root="data/OGBN")
        split_idx = dataset.get_idx_split()
        data = dataset[0]

        return {"data": data, "split": split_idx}, [data]

    else:
        raise ValueError("Unsupported dataset: choose 'ppi' or 'ogbn-proteins'")
