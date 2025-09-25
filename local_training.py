# local_training.py
import torch

def train_local(model, data, optimizer, loss_fn, epochs=1, device="cpu"):
    model.to(device)
    data = data.to(device)
    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        if data.y.dim() == 1:  
            # single-label classification
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        else:  
            # multi-label classification
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask].float())

        loss.backward()
        optimizer.step()

    return loss.item()
