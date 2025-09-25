# Federated Graph Attention Network with Quantum Embeddings

This repository contains code for **federated graph neural networks with
quantum encodings** (QuanGAT) applied to **DNA mutation prediction and
protein--protein interaction datasets**.\
It combines **federated learning**, **graph attention networks (GATs)**,
and **quantum neural network (QNN) embeddings** for robust and
interpretable predictions.

------------------------------------------------------------------------

## 📂 Project Structure

    fedqnn/
    │── main.py                  # Run federated training (PPI/STRING/OGBN/Mutations)
    │── loaders.py               # Dataset loaders
    │── models.py                # QuanGAT model (GAT + QNN encoder)
    │── federated_training.py    # Federated averaging + training loop
    │── evaluation.py            # Evaluation metrics (F1, AUROC, AUPRC)
    │── local_training.py        # Local client training
    │── interpretability.py      # Saliency maps + attention analysis
    │── mutation_predict.py      # DNA mutation predictions (standalone)
    │── requirements.txt         # Dependencies
    │── README.md                # This file

------------------------------------------------------------------------

## Installation

Create a fresh Python environment:

``` bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows PowerShell
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Usage

### 1. Federated Training

Run federated training on a dataset (default = PPI):

``` bash
python main.py
```

You can change the dataset inside `main.py`:

``` python
dataset_name = "ppi"   # options: "ppi", "string", "ogbn-proteins", "dna_mutations"
```

This produces a trained global model (`global_model.pth`) and prints
evaluation metrics per round.

------------------------------------------------------------------------

### 2. DNA Mutation Predictions

After training, generate predictions on DNA mutations:

``` bash
python mutation_predict.py
```

This creates:

    mutation_predictions.csv

with columns: - `node_id`: mutation/protein site ID\
- `true_label`: ground-truth label\
- `predicted_label`: model prediction\
- `confidence`: prediction probability

------------------------------------------------------------------------

### 3. Interpretability Analysis

Run saliency + attention analysis:

``` bash
python interpretability.py
```

Outputs: - `saliency.csv`: feature attributions per node\
- `attention.csv`: edge attention weights\
- Attention heatmap visualization

------------------------------------------------------------------------

Mutation predictions are stored in `mutation_predictions.csv` and can be
post-processed for case studies.

------------------------------------------------------------------------

## Datasets

-   **PPI**: Protein--Protein Interaction dataset.
-   **STRING**: Human protein interactions.
-   **OGBN-Proteins**: Large-scale protein graph.
-   **DNA Mutations**: Placeholder for mutation datasets (e.g., ClinVar,
    TCGA, COSMIC).

------------------------------------------------------------------------
