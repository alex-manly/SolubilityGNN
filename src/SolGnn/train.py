from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn.functional as F

@dataclass
class FitResult:
    best_val_mse: float
    best_epoch: int
    history: Dict[str, list]
    best_state_dict: dict

@torch.no_grad()
def evaluate(model, loader, device: str) -> Dict[str, float]:
    model.eval()
    ys, yps = [], []
    for batch in loader:
        batch = batch.to(device)
        y_true = batch.y.view(-1).float()
        y_pred = model(batch).view(-1).float()
        ys.append(y_true)
        yps.append(y_pred)
    y_true = torch.cat(ys)
    y_pred = torch.cat(yps)
    mse = F.mse_loss(y_pred, y_true).item()
    rmse = mse**.5
    return {"mse": mse, "rmse": rmse}

def fit(model, train_loader, val_loader, cfg, lr = 0.001) -> FitResult:
    device = cfg.device
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr = lr)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0

    history = {"train_mse": [], "val_mse": [], "val_rmse": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0

        for batch in train_loader:
            batch = batch.to(device)
            y_true = batch.y.view(-1).float()

            opt.zero_grad(set_to_none=True)
            y_pred = model(batch).view(-1).float()

            loss = F.mse_loss(y_pred, y_true)
            loss.backward()
            opt.step()

            running_loss += loss.item() * y_true.numel()
            n += y_true.numel()

        train_mse = running_loss / max(n, 1)
        val_metrics = evaluate(model, val_loader, device=device)

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_metrics["mse"])
        history["val_rmse"].append(val_metrics["rmse"])

        # early stopping on val MSE
        improved = (best_val - val_metrics["mse"]) > cfg.min_delta
        if improved:
            best_val = val_metrics["mse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

        if epoch % 20 == 0 or epoch == 1: # print every 20 epochs
            print(
                f"Epoch {epoch:4d} | train_mse={train_mse:.4g} | "
                f"val_mse={val_metrics['mse']:.4g}"
            )

    return FitResult(
        best_val_mse=best_val,
        best_epoch=best_epoch,
        history=history,
        best_state_dict=best_state,
    )
