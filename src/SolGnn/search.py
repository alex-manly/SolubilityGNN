from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
import torch

from SolGnn.config import project_root
from SolGnn.train import fit, evaluate


def run_search(
    make_model: Callable[[Dict[str, Any]], torch.nn.Module],
    loaders: Dict[str, Any],  # {"train": ..., "val": ..., "test": ...}
    train_config,
    space: Dict[str, List[Any]],
    csv_relpath: str = "results/grid_search.csv",
) -> None:

    out_path = project_root() / Path(csv_relpath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = True
    if out_path.exists(): # restart-friendly
        header = False

    keys = list(space.keys())
    combos = list(product(*[space[k] for k in keys]))

    rows = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        torch.manual_seed(train_config.seed)

        model = make_model(params)
        result = fit(model, loaders["train"], loaders["val"], train_config)

        model.load_state_dict(result.best_state_dict)
        val = evaluate(model, loaders["val"], device=train_config.device)
        test = evaluate(model, loaders["test"], device=train_config.device)

        row = {
            "trial": i,
            **params,
            "best_epoch": result.best_epoch,
            "val_mse": val["mse"],
            "val_rmse": val["rmse"],
            "test_mse": test["mse"],
            "test_rmse": test["rmse"],
        }
        rows.append(row)

        pd.DataFrame([row]).to_csv(out_path, mode="a", header=header, index=False)

        print(f"[{i+1}/{len(combos)}] val_mse={row['val_mse']:.4g} params={params}")

        # ------- save info for later
        model_name = "" # will be mostly unqiue to this set of parameters
        for k, c in zip(keys, combo):
            model_name += str(k) + "=" + str(c)

        torch.save( # save model state
            result.best_state_dict,
            project_root() / "results" / "models" / (model_name + ".pt")
        )
        pd.DataFrame(result.history).to_csv( # save training history
            project_root() / "results" / "history" / (model_name + ".csv"),
            index = True
        )


