# main.py
from loaders import load_dataset
from models import QuanGAT
from federated_training import federated_train

if __name__ == "__main__":
    dataset_name = "ppi"  # "ppi", "ogbn-proteins", or "string" (demo only)
    data, clients = load_dataset(dataset_name, num_clients=5)

    use_quantum = True  # switch to False for baseline GAT
    Model = QuanGAT if use_quantum else GAT

    out_channels = data.y.shape[1] if data.y.dim() == 2 else int(data.y.max().item()) + 1

    history = federated_train(
        model_class=Model,
        clients=clients,
        global_rounds=20,
        in_channels=data.num_features,
        hidden=256,
        out_channels=out_channels,
        lr=0.05,
        local_epochs=10,
        device="cpu"
    )

    print("\nFinal results:", history[-1])
