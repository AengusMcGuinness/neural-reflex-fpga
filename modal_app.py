"""Remote trainer for the causal temporal conv net.

Run only after a session exists:

    python3 -m modal run modal_app.py --features-npy data/sessions/<id>/features.npy \\
        --targets-npy data/sessions/<id>/targets.npy
"""

import modal

app = modal.App("neural-reflex-angle-tcn")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.24", "scipy>=1.11", "torch>=2.2")
    .add_local_python_source("neural_reflex")
)


@app.function(image=image, gpu="T4", timeout=30 * 60)
def train_tcn(features: list, targets: list, epochs: int = 40):
    import numpy as np
    import torch
    from torch import nn

    from neural_reflex.tcn import AngleTCN

    inputs = torch.tensor(np.asarray(features), dtype=torch.float32)
    labels = torch.tensor(np.asarray(targets), dtype=torch.float32)
    if inputs.ndim != 3 or inputs.shape[1] != 8:
        raise ValueError("features must have shape (windows, 8, time)")
    model = AngleTCN(n_joints=labels.shape[-1]).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs.to("cuda")), labels.to("cuda"))
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(inputs.to("cuda")).cpu()
        final_loss = float(loss_fn(predictions, labels))
    state = {name: value.cpu().tolist() for name, value in model.state_dict().items()}
    return {"train_loss": final_loss, "state": state}


@app.local_entrypoint()
def main(features_npy: str, targets_npy: str, epochs: int = 40):
    import numpy as np
    features = np.load(features_npy)
    targets = np.load(targets_npy)
    result = train_tcn.remote(features.tolist(), targets.tolist(), epochs)
    print({"train_loss": result["train_loss"], "saved_tensors": len(result["state"])})
