import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pennylane as qml


class QuanGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_qubits=4):
        super().__init__()

        # Project input features (e.g., 50 from PPI) -> n_qubits
        self.feature_proj = torch.nn.Linear(in_channels, n_qubits)

        # Quantum device
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (3, n_qubits)}
        self.quantum = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.fc = torch.nn.Linear(n_qubits, hidden_channels)

        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=4, concat=True)
        self.gat2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index, return_attention=False):
        # Step 1: project features
        x = self.feature_proj(x)

        # Step 2: quantum embedding
        qx = self.quantum(x)

        # Step 3: classical fully connected
        x = self.fc(qx)

        # Step 4: GAT layers
        if return_attention:
            x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
            x = F.relu(x)
            x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
            return x, (attn1, attn2)
        else:
            x = self.gat1(x, edge_index).relu()
            x = self.gat2(x, edge_index)
            return x
