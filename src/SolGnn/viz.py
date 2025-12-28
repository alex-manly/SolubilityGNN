import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import torch
from typing import Optional,Tuple


def hist_and_qq_plot(y_true, y_pred, prob_func=None):

    residuals = y_pred - y_true

    fig, (ax_hist, ax_qq) = plt.subplots(
        ncols=2, figsize=(11, 5)
    )

    # histogram and pdf fit
    ax_hist.hist(residuals, bins=75, density=True, alpha=0.6)

    x = np.linspace(residuals.min(), residuals.max(), 400)

    if prob_func is None:
        prob_func = norm
    mu, sigma = prob_func.fit(residuals)
    pdf = prob_func.pdf(x, loc = mu, scale = sigma)
    ax_hist.plot(x, pdf)
    ax_hist.axvline(0.0, linestyle="--", linewidth=1)

    ax_hist.set_xlabel("Residual (Pred − True)")
    ax_hist.set_ylabel("Probability density")


    ax_hist.text(
        0.95, 0.95,
        f"$\\mu$ = {round(float(mu),3 )}\n$\\sigma$ = {round(float(sigma), 3)}",
        transform=ax_hist.transAxes,
        ha="right",
        va="top"
    )

    # QQ Plot
    n = len(residuals)
    sorted_res = np.sort(residuals)

    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = prob_func.ppf(probs, loc = mu, scale = sigma)

    ax_qq.plot(theo, sorted_res,"o", alpha=0.6)
    lo = sorted_res.min()
    hi = sorted_res.max()
    ax_qq.plot([lo, hi], [lo, hi], linestyle="--")

    ax_qq.set_xlim(lo, hi)
    ax_qq.set_ylim(lo, hi)

    ax_qq.set_xlabel("Theoretical quantiles")
    ax_qq.set_ylabel("Observed quantiles")

    plt.tight_layout()
    plt.show()

def parity_plot(y_true, y_pred):
    fig,ax = plt.subplots(ncols=2,nrows=2, figsize=(6, 5), gridspec_kw={"width_ratios": [1.0, 0.15],"height_ratios":[.15,1.0]}, constrained_layout=True)

    lo,hi = y_true.min(), y_true.max()
    lim = (1.0*lo, 1.2*hi)

    #--------- Bottom left; parity plot ---------[
    ax[1,0].plot(y_true, y_pred, 'o', alpha = 0.3)
    ax[1,0].plot([lo,hi], [lo,hi], '-')
    ax[1,0].set_ylabel("Prediction ($\\log S$)")
    ax[1,0].set_xlabel("True Value ($\\log S$)")
    ax[1,0].set_xlim(lim)
    ax[1,0].set_ylim(lim)

    #------------- Bottom right; histogram --------
    mu = y_pred.mean()
    sigma = y_pred.std()
    y_pdf = np.linspace(lo, hi, 100)
    pdf = norm.pdf(y_pdf, loc = mu, scale = sigma)
    ax[1,1].hist(y_pred, bins=75, density=True, orientation="horizontal", alpha=0.5)
    ax[1,1].plot(pdf, y_pdf)
    ax[1,1].set_yticks([])
    ax[1,1].set_xticks([])
    ax[1,1].set_axis_off()
    ax[1,1].set_ylim(lim)

    #------------- Top left; histogram --------
    mu = y_true.mean()
    sigma = y_true.std()
    x_pdf = np.linspace(lo, hi, 100)
    pdf = norm.pdf(y_pdf, loc = mu, scale = sigma)
    ax[0,0].hist(y_true, bins=75, density=True, orientation="vertical", alpha=0.5)
    ax[0,0].plot(x_pdf, pdf)
    ax[0,0].set_yticks([])
    ax[0,0].set_xticks([])
    ax[0,0].set_axis_off()
    ax[0,0].set_xlim(lim)

    #------------ Top right; hide ---------
    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])
    ax[0,1].set_axis_off()

    #------------ Some metrics ----------
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    ax[1,0].text(
        lo,
        hi-3,
        f"""
        $r^2 = {round(r2,3)}$\n
        $RMSE = {round(rmse, 3)}$\n
        $MAE = {round(mae,3)}$
        """,
        linespacing = 0.5
    )


    plt.show()

@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    y_scaler=None,
) -> Tuple[np.ndarray, np.ndarray]:
# gets the model's inference for a set of test samples
    model.eval()

    y_true_chunks = []
    y_pred_chunks = []

    for batch in test_loader:
        batch = batch.to(device)

        pred = model(batch)

        # Flatten predictions and targets to 1D per sample
        pred = pred.view(-1).detach().cpu().numpy()
        true = batch.y.view(-1).detach().cpu().numpy()

        y_pred_chunks.append(pred)
        y_true_chunks.append(true)

    y_pred = np.concatenate(y_pred_chunks, axis=0)
    y_true = np.concatenate(y_true_chunks, axis=0)

    if y_scaler is not None: # optionally get "real" unscaled values
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
        y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)

    return y_true, y_pred