import argparse
import copy
import csv
import math
import os
import random
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "unggcn_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "unggcn_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from modules.UnGGSL import UnGGSL
from utils.data_loader import load_data
from utils.parser import parse_args as model_parse_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_csv(path, rows, fieldnames):
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expected_checkpoint_names(args):
    return [
        (
            f"modelmodel_dataset_{args.dataset}_dim{args.dim}"
            f"_hops{args.context_hops}_lr{args.lr}_lw{args.lw}"
            f"_beta{args.beta}_prioralpha{args.prior_alpha}"
            f"_priorbeta{args.prior_beta}_noise_{args.noise_ratio}.ckpt"
        ),
        (
            f"model_dataset_{args.dataset}_noise_{args.noise_ratio}"
            f"_dim{args.dim}_hops{args.context_hops}"
            f"_beta{args.beta}_lr{args.lr}.ckpt"
        ),
    ]


def checkpoint_matches(path, args):
    patterns = [
        (
            re.compile(
                rf"^modelmodel_dataset_{re.escape(args.dataset)}"
                rf"_dim(?P<dim>\d+)_hops(?P<hops>\d+)"
                rf"_lr(?P<lr>[^_]+)_lw(?P<lw>[^_]+)"
                rf"_beta(?P<beta>[^_]+)_prioralpha(?P<alpha>[^_]+)"
                rf"_priorbeta(?P<prior_beta>[^_]+)"
                rf"_noise_(?P<noise>[^_]+)\.ckpt$"
            ),
            True,
        ),
        (
            re.compile(
                rf"^model_dataset_{re.escape(args.dataset)}"
                rf"_noise_(?P<noise>[^_]+)_dim(?P<dim>\d+)"
                rf"_hops(?P<hops>\d+)_beta(?P<beta>[^_]+)"
                rf"_lr(?P<lr>[^_]+)\.ckpt$"
            ),
            False,
        ),
    ]

    for pattern, has_prior_config in patterns:
        match = pattern.match(path.name)
        if not match:
            continue
        values = match.groupdict()
        try:
            base_matches = (
                int(values["dim"]) == int(args.dim)
                and int(values["hops"]) == int(args.context_hops)
                and math.isclose(float(values["lr"]), float(args.lr))
                and math.isclose(float(values["beta"]), float(args.beta))
                and math.isclose(float(values["noise"]), float(args.noise_ratio))
            )
            if not has_prior_config:
                return base_matches
            return (
                base_matches
                and math.isclose(float(values["lw"]), float(args.lw))
                and math.isclose(float(values["alpha"]), float(args.prior_alpha))
                and math.isclose(float(values["prior_beta"]), float(args.prior_beta))
            )
        except ValueError:
            return False
    return False


def checkpoint_candidates(args):
    model_root = Path(args.model_dir)
    search_dirs = [model_root / args.dataset, model_root]

    for directory in search_dirs:
        for filename in expected_checkpoint_names(args):
            path = directory / filename
            if path.is_file():
                return [path]

    matches = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        matches.extend(
            path for path in directory.glob("*.ckpt")
            if checkpoint_matches(path, args)
        )
    return sorted(set(matches))


def find_checkpoint(args):
    candidates = checkpoint_candidates(args)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        paths = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Multiple full-model checkpoints match the requested configuration:\n"
            f"  {paths}"
        )

    model_root = Path(args.model_dir)
    searched = "\n  ".join(
        str(directory / filename)
        for directory in (model_root / args.dataset, model_root)
        for filename in expected_checkpoint_names(args)
    )
    command = (
        f"python main.py --dataset {args.dataset} --noise_ratio {args.noise_ratio} "
        f"--dim {args.dim} --context_hops {args.context_hops} --beta {args.beta} "
        f"--lr {args.lr} --lw {args.lw} --prior_alpha {args.prior_alpha} "
        f"--prior_beta {args.prior_beta}"
    )
    raise FileNotFoundError(
        f"Missing checkpoint for dataset={args.dataset}, noise_ratio={args.noise_ratio}, "
        f"configuration=(dim={args.dim}, hops={args.context_hops}, beta={args.beta}, "
        f"lr={args.lr}).\nSearched:\n  {searched}\nTrain it with:\n  {command}"
    )


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state_dict-like mapping.")
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def load_model_bundle(noise_ratio, base_args):
    args = copy.deepcopy(base_args)
    args.noise_ratio = float(noise_ratio)
    args.cuda = bool(args.cuda and torch.cuda.is_available())
    train_cf, user_dict, n_params, norm_mat, norm_mat_var = load_data(args)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model = UnGGSL(n_params, args, norm_mat, norm_mat_var).to(device)
    ckpt_path = find_checkpoint(args)
    print(f"Loading checkpoint: {ckpt_path}")
    state = unwrap_state_dict(torch.load(ckpt_path, map_location=device))
    model.load_state_dict(state)
    model.eval()
    return {
        "args": args,
        "model": model,
        "device": device,
        "train_cf": train_cf,
        "user_dict": user_dict,
        "n_params": n_params,
        "ckpt_path": ckpt_path,
    }


@torch.no_grad()
def representations(model):
    user_mu, item_mu, user_var, item_var = model.generate(split=True)
    return {
        "user_mu": user_mu.detach(),
        "item_mu": item_mu.detach(),
        "user_var": user_var.detach(),
        "item_var": item_var.detach(),
    }


def attention_from_variance(var, beta):
    return F.softplus(-var, beta=beta)


def sample_edges(edges, max_edges, seed):
    if edges is None or len(edges) == 0:
        return np.empty((0, 2), dtype=np.int32)
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    if len(edges) <= max_edges:
        return edges
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(edges), size=max_edges, replace=False)
    return edges[idx]


@torch.no_grad()
def predictive_uncertainty(edges, reps, device, batch_size=4096):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return np.array([], dtype=np.float64)

    out = []
    user_mu = reps["user_mu"]
    user_var = reps["user_var"]
    item_mu = reps["item_mu"]
    item_var = reps["item_var"]
    for start in range(0, len(edges), batch_size):
        batch = edges[start:start + batch_size]
        users = torch.as_tensor(batch[:, 0], dtype=torch.long, device=device)
        items = torch.as_tensor(batch[:, 1], dtype=torch.long, device=device)
        u_mu = user_mu[users]
        u_var = user_var[users]
        i_mu = item_mu[items]
        i_var = item_var[items]
        var_y = (u_var * i_var + u_var * i_mu.pow(2) + u_mu.pow(2) * i_var).sum(dim=-1)
        out.append(var_y.detach().cpu().numpy())
    return np.concatenate(out)


def match_clean_noisy_edges(clean_edges, noisy_edges, max_pairs, seed):
    clean_by_user = {}
    for user, item in np.asarray(clean_edges, dtype=np.int64).reshape(-1, 2):
        clean_by_user.setdefault(int(user), []).append(int(item))

    eligible_noisy = [
        (int(user), int(item))
        for user, item in np.asarray(noisy_edges, dtype=np.int64).reshape(-1, 2)
        if int(user) in clean_by_user
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible_noisy)
    eligible_noisy = eligible_noisy[:max_pairs]

    matched_clean = []
    matched_noisy = []
    for user, noisy_item in eligible_noisy:
        clean_item = int(rng.choice(clean_by_user[user]))
        matched_clean.append((user, clean_item))
        matched_noisy.append((user, noisy_item))
    return (
        np.asarray(matched_clean, dtype=np.int64).reshape(-1, 2),
        np.asarray(matched_noisy, dtype=np.int64).reshape(-1, 2),
    )


@torch.no_grad()
def dimension_uncertainty(edges, reps, device):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return np.empty((0, reps["user_mu"].shape[1]), dtype=np.float64)
    users = torch.as_tensor(edges[:, 0], dtype=torch.long, device=device)
    items = torch.as_tensor(edges[:, 1], dtype=torch.long, device=device)
    u_mu = reps["user_mu"][users]
    u_var = reps["user_var"][users]
    i_mu = reps["item_mu"][items]
    i_var = reps["item_var"][items]
    contributions = u_var * i_var + u_var * i_mu.pow(2) + u_mu.pow(2) * i_var
    return contributions.detach().cpu().numpy()


def concentration_metrics(contributions):
    contributions = np.asarray(contributions, dtype=np.float64)
    probs = contributions / np.maximum(contributions.sum(axis=1, keepdims=True), 1e-12)
    dim = contributions.shape[1]
    top_k = max(1, int(math.ceil(dim * 0.2)))
    top_share = np.sort(probs, axis=1)[:, -top_k:].sum(axis=1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1) / math.log(dim)
    return top_share, entropy


def paired_stats(clean, noisy):
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    delta = noisy - clean
    result = {
        "n_pairs": int(len(delta)),
        "clean_mean": float(np.mean(clean)),
        "noisy_mean": float(np.mean(noisy)),
        "mean_paired_delta": float(np.mean(delta)),
        "median_paired_delta": float(np.median(delta)),
        "noisy_greater_fraction": float(np.mean(delta > 0)),
        "cohen_dz": float(np.mean(delta) / np.std(delta, ddof=1))
        if len(delta) > 1 and np.std(delta, ddof=1) > 0 else np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p": np.nan,
    }
    try:
        from scipy.stats import wilcoxon
        stat, p_value = wilcoxon(noisy, clean, alternative="greater")
        result["wilcoxon_stat"] = float(stat)
        result["wilcoxon_p"] = float(p_value)
    except (ImportError, ValueError):
        pass
    return result


def aggregate_paired_by_user(user_ids, clean_values, noisy_values):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    clean_values = np.asarray(clean_values, dtype=np.float64)
    noisy_values = np.asarray(noisy_values, dtype=np.float64)
    unique_users = np.unique(user_ids)
    clean_user = np.asarray([
        clean_values[user_ids == user_id].mean() for user_id in unique_users
    ])
    noisy_user = np.asarray([
        noisy_values[user_ids == user_id].mean() for user_id in unique_users
    ])
    return unique_users, clean_user, noisy_user


def run_motivation(args, noise_ratio, save_root, max_samples):
    if noise_ratio <= 0:
        raise ValueError("The motivation experiment requires --noise_ratio > 0.")

    bundle = load_model_bundle(noise_ratio, args)
    reps = representations(bundle["model"])
    clean_edges, noisy_edges = match_clean_noisy_edges(
        bundle["n_params"]["clean_train_cf"],
        bundle["n_params"]["injected_noise_edges"],
        max_samples,
        args.seed,
    )
    if len(noisy_edges) == 0:
        raise ValueError(
            "No injected noisy edges could be matched to clean interactions. "
            "Increase the noise ratio or check the dataset."
        )

    clean_dim_unc = dimension_uncertainty(clean_edges, reps, bundle["device"])
    noisy_dim_unc = dimension_uncertainty(noisy_edges, reps, bundle["device"])
    clean_edge_unc = clean_dim_unc.sum(axis=1)
    noisy_edge_unc = noisy_dim_unc.sum(axis=1)
    clean_top_share, clean_entropy = concentration_metrics(clean_dim_unc)
    noisy_top_share, noisy_entropy = concentration_metrics(noisy_dim_unc)

    log_delta = np.log(noisy_dim_unc + 1e-12) - np.log(clean_dim_unc + 1e-12)
    pair_order = np.argsort(log_delta.mean(axis=1))[::-1]
    dim_order = np.argsort(log_delta.mean(axis=0))[::-1]
    heatmap_sorted = log_delta[pair_order[:min(40, len(pair_order))]][:, dim_order]
    save_dir = Path(save_root) / "motivation"
    ensure_dir(save_dir)

    pair_rows = []
    for pair_id, (clean_edge, noisy_edge) in enumerate(zip(clean_edges, noisy_edges)):
        pair_rows.append({
            "pair_id": pair_id,
            "user_id": int(noisy_edge[0]),
            "clean_item_id": int(clean_edge[1]),
            "noisy_item_id": int(noisy_edge[1]),
            "clean_predictive_variance": float(clean_edge_unc[pair_id]),
            "noisy_predictive_variance": float(noisy_edge_unc[pair_id]),
            "predictive_variance_delta": float(noisy_edge_unc[pair_id] - clean_edge_unc[pair_id]),
            "clean_top20_dimension_share": float(clean_top_share[pair_id]),
            "noisy_top20_dimension_share": float(noisy_top_share[pair_id]),
            "clean_normalized_entropy": float(clean_entropy[pair_id]),
            "noisy_normalized_entropy": float(noisy_entropy[pair_id]),
        })
    write_csv(
        save_dir / "paired_edge_metrics.csv",
        pair_rows,
        [
            "pair_id", "user_id", "clean_item_id", "noisy_item_id",
            "clean_predictive_variance", "noisy_predictive_variance", "predictive_variance_delta",
            "clean_top20_dimension_share", "noisy_top20_dimension_share",
            "clean_normalized_entropy", "noisy_normalized_entropy",
        ],
    )

    mean_clean_dim = clean_dim_unc.mean(axis=0)
    mean_noisy_dim = noisy_dim_unc.mean(axis=0)
    user_ids = torch.as_tensor(noisy_edges[:, 0], dtype=torch.long, device=bundle["device"])
    item_ids = torch.as_tensor(noisy_edges[:, 1], dtype=torch.long, device=bundle["device"])
    endpoint_var = torch.cat([reps["user_var"][user_ids], reps["item_var"][item_ids]], dim=0)
    endpoint_weight = attention_from_variance(endpoint_var, args.beta)
    flat_var = endpoint_var.detach().cpu().numpy().reshape(-1)
    flat_weight = endpoint_weight.detach().cpu().numpy().reshape(-1)

    dimension_rows = []
    for rank, dim in enumerate(dim_order):
        dimension_rows.append({
            "dimension": int(dim),
            "dimension_rank": int(rank),
            "mean_clean_variance_contribution": float(mean_clean_dim[dim]),
            "mean_noisy_variance_contribution": float(mean_noisy_dim[dim]),
            "mean_log_ratio": float(log_delta[:, dim].mean()),
        })
    write_csv(
        save_dir / "dimension_uncertainty_summary.csv",
        dimension_rows,
        [
            "dimension", "dimension_rank", "mean_clean_variance_contribution",
            "mean_noisy_variance_contribution", "mean_log_ratio",
        ],
    )

    pair_user_ids = noisy_edges[:, 0]
    user_level_metrics = {}
    stats_rows = []
    for metric, clean_values, noisy_values in [
        ("predictive_variance", clean_edge_unc, noisy_edge_unc),
        ("top20_dimension_share", clean_top_share, noisy_top_share),
        ("negative_normalized_entropy", -clean_entropy, -noisy_entropy),
    ]:
        _, clean_user_values, noisy_user_values = aggregate_paired_by_user(
            pair_user_ids, clean_values, noisy_values
        )
        user_level_metrics[metric] = (clean_user_values, noisy_user_values)
        stats_rows.append({
            "metric": metric,
            "analysis_unit": "user",
            **paired_stats(clean_user_values, noisy_user_values),
        })
    write_csv(save_dir / "motivation_statistics.csv", stats_rows, list(stats_rows[0].keys()))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    clean_user_unc, noisy_user_unc = user_level_metrics["predictive_variance"]
    axes[0].boxplot(
        [clean_user_unc, noisy_user_unc],
        labels=["Matched clean", "Injected noisy"],
        showfliers=False,
        patch_artist=True,
    )
    axes[0].set_title("(a) User-level predictive uncertainty")
    axes[0].set_ylabel(r"$\mathrm{Var}[Y_{ui}]$")
    axes[0].grid(axis="y", alpha=0.25)

    max_abs = max(float(np.percentile(np.abs(heatmap_sorted), 95)), 1e-6)
    im = axes[1].imshow(
        heatmap_sorted,
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    axes[1].set_title("(b) Dimension-wise uncertainty shift")
    axes[1].set_xlabel("Dimensions sorted by mean shift")
    axes[1].set_ylabel("Matched interaction pairs")
    fig.colorbar(im, ax=axes[1], label="log noisy / clean contribution")

    quantile_edges = np.unique(np.quantile(flat_var, np.linspace(0, 1, 11)))
    if len(quantile_edges) > 1:
        bin_ids = np.digitize(flat_var, quantile_edges[1:-1], right=True)
        bin_x = []
        bin_y = []
        for idx in range(len(quantile_edges) - 1):
            mask = bin_ids == idx
            if np.any(mask):
                bin_x.append(float(np.median(flat_var[mask])))
                bin_y.append(float(np.mean(flat_weight[mask])))
        axes[2].plot(bin_x, bin_y, marker="o", linewidth=2, color="#4C72B0")
    axes[2].set_xscale("log")
    axes[2].set_title("(c) Variance-guided attenuation")
    axes[2].set_xlabel("Endpoint variance (quantile-bin median)")
    axes[2].set_ylabel("Message-passing weight")
    axes[2].grid(alpha=0.25)

    fig.suptitle(f"Motivation Validation on Matched Interactions (noise={noise_ratio:g})")
    fig.tight_layout()
    fig.savefig(save_dir / "motivation_validation.png", dpi=220)
    plt.close(fig)

    print(f"Saved {len(noisy_edges)} matched pairs and motivation results to {save_dir}")


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return float(np.corrcoef(x, y)[0, 1]), np.nan


def spearman_corr(x, y):
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def mann_whitney(clean_values, noisy_values):
    if len(clean_values) == 0 or len(noisy_values) == 0:
        return np.nan, np.nan
    try:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(clean_values, noisy_values, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def run_noise_uncertainty(args, noise_ratios, save_root, edge_sample_size):
    save_dir = Path(save_root) / "noise_uncertainty"
    ensure_dir(save_dir)

    summary_rows = []
    sample_rows = []
    burden_rows = []
    burden_corr_rows = []
    all_clean_unc = []
    all_noisy_unc = []

    for noise_ratio in noise_ratios:
        bundle = load_model_bundle(noise_ratio, args)
        reps = representations(bundle["model"])
        raw_user_var_mean = float(torch.exp(2 * bundle["model"].user_logsigma).mean().detach().cpu().item())
        raw_item_var_mean = float(torch.exp(2 * bundle["model"].item_logsigma).mean().detach().cpu().item())
        post_user_var = reps["user_var"].mean(dim=1).detach().cpu().numpy()
        post_item_var = reps["item_var"].mean(dim=1).detach().cpu().numpy()
        post_user_var_mean = float(np.mean(post_user_var))
        post_item_var_mean = float(np.mean(post_item_var))

        clean_edges = sample_edges(
            bundle["n_params"]["clean_train_cf"], edge_sample_size, args.seed + int(float(noise_ratio) * 1000)
        )
        noisy_edges = sample_edges(
            bundle["n_params"]["injected_noise_edges"], edge_sample_size, args.seed + 17 + int(float(noise_ratio) * 1000)
        )
        clean_unc = predictive_uncertainty(clean_edges, reps, bundle["device"])
        noisy_unc = predictive_uncertainty(noisy_edges, reps, bundle["device"])

        if len(bundle["n_params"]["injected_noise_edges"]) > 0:
            noise_counts = np.bincount(
                bundle["n_params"]["injected_noise_edges"][:, 0],
                minlength=bundle["n_params"]["n_users"],
            )
            mask = noise_counts > 0
            pearson_r, pearson_p = pearson_corr(noise_counts[mask], post_user_var[mask])
            spearman_r, spearman_p = spearman_corr(noise_counts[mask], post_user_var[mask])
            burden_corr_rows.append({
                "noise_ratio": float(noise_ratio),
                "num_users_with_noise": int(mask.sum()),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            })
            for user_id in np.where(mask)[0]:
                burden_rows.append({
                    "noise_ratio": float(noise_ratio),
                    "user_id": int(user_id),
                    "noise_count": int(noise_counts[user_id]),
                    "mean_user_variance": float(post_user_var[user_id]),
                })

        if len(clean_unc) > 0:
            all_clean_unc.append(clean_unc)
        if len(noisy_unc) > 0:
            all_noisy_unc.append(noisy_unc)

        summary_rows.append({
            "noise_ratio": float(noise_ratio),
            "mean_raw_user_variance": raw_user_var_mean,
            "mean_raw_item_variance": raw_item_var_mean,
            "mean_post_user_variance": post_user_var_mean,
            "mean_post_item_variance": post_item_var_mean,
            "mean_clean_edge_uncertainty": float(np.mean(clean_unc)) if len(clean_unc) else np.nan,
            "mean_noisy_edge_uncertainty": float(np.mean(noisy_unc)) if len(noisy_unc) else np.nan,
            "num_clean_edge_samples": int(len(clean_unc)),
            "num_noisy_edge_samples": int(len(noisy_unc)),
        })

        for value in clean_unc:
            sample_rows.append({"noise_ratio": float(noise_ratio), "edge_type": "clean", "predictive_uncertainty": float(value)})
        for value in noisy_unc:
            sample_rows.append({"noise_ratio": float(noise_ratio), "edge_type": "noisy", "predictive_uncertainty": float(value)})

    write_csv(
        save_dir / "variance_vs_noise_ratio.csv",
        summary_rows,
        [
            "noise_ratio", "mean_raw_user_variance", "mean_raw_item_variance",
            "mean_post_user_variance", "mean_post_item_variance",
            "mean_clean_edge_uncertainty", "mean_noisy_edge_uncertainty",
            "num_clean_edge_samples", "num_noisy_edge_samples",
        ],
    )
    write_csv(
        save_dir / "clean_vs_noisy_uncertainty.csv",
        sample_rows,
        ["noise_ratio", "edge_type", "predictive_uncertainty"],
    )
    write_csv(
        save_dir / "user_noise_burden.csv",
        burden_rows,
        ["noise_ratio", "user_id", "noise_count", "mean_user_variance"],
    )
    write_csv(
        save_dir / "noise_burden_correlation.csv",
        burden_corr_rows,
        ["noise_ratio", "num_users_with_noise", "pearson_r", "pearson_p", "spearman_r", "spearman_p"],
    )

    x = np.array([row["noise_ratio"] for row in summary_rows])
    raw_user_y = np.array([row["mean_raw_user_variance"] for row in summary_rows])
    raw_item_y = np.array([row["mean_raw_item_variance"] for row in summary_rows])
    post_user_y = np.array([row["mean_post_user_variance"] for row in summary_rows])
    post_item_y = np.array([row["mean_post_item_variance"] for row in summary_rows])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, raw_user_y, marker="o", linestyle="--", linewidth=2, label="Raw user variance")
    ax.plot(x, raw_item_y, marker="s", linestyle="--", linewidth=2, label="Raw item variance")
    ax.plot(x, post_user_y, marker="o", linewidth=2, label="Post-GCN user variance")
    ax.plot(x, post_item_y, marker="s", linewidth=2, label="Post-GCN item variance")
    ax.set_title("Mean Variance under Different Noise Ratios")
    ax.set_xlabel("Noise ratio")
    ax.set_ylabel("Mean variance")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_dir / "variance_vs_noise_ratio.png", dpi=220)
    plt.close(fig)

    plot_data = []
    labels = []
    positions = []
    pos = 1
    for row in summary_rows:
        ratio = row["noise_ratio"]
        clean_values = [r["predictive_uncertainty"] for r in sample_rows if r["noise_ratio"] == ratio and r["edge_type"] == "clean"]
        noisy_values = [r["predictive_uncertainty"] for r in sample_rows if r["noise_ratio"] == ratio and r["edge_type"] == "noisy"]
        if clean_values:
            plot_data.append(clean_values)
            labels.append(f"{ratio:g}\nclean")
            positions.append(pos)
            pos += 1
        if noisy_values:
            plot_data.append(noisy_values)
            labels.append(f"{ratio:g}\nnoisy")
            positions.append(pos)
            pos += 1
        pos += 0.5

    if plot_data:
        fig, ax = plt.subplots(figsize=(max(9, len(plot_data) * 0.8), 5.5))
        ax.boxplot(plot_data, positions=positions, showfliers=False, patch_artist=True)
        ax.set_title("Predictive Uncertainty of Clean vs Injected Noisy Edges")
        ax.set_xlabel("Noise ratio / edge type")
        ax.set_ylabel("Predictive uncertainty Var[Y_ui]")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(save_dir / "clean_vs_noisy_uncertainty.png", dpi=220)
        plt.close(fig)

    if burden_rows:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for ratio in sorted({row["noise_ratio"] for row in burden_rows}):
            rows = [row for row in burden_rows if row["noise_ratio"] == ratio]
            counts = sorted({row["noise_count"] for row in rows})
            xs, ys = [], []
            for count in counts:
                vals = [row["mean_user_variance"] for row in rows if row["noise_count"] == count]
                if len(vals) >= 10:
                    xs.append(count)
                    ys.append(float(np.mean(vals)))
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=f"noise={ratio:g}")
        ax.set_title("Users with More Injected Noise Show Higher Uncertainty")
        ax.set_xlabel("Injected noisy interactions per user")
        ax.set_ylabel("Mean post-GCN user variance")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / "user_noise_burden_scatter.png", dpi=220)
        plt.close(fig)

    all_clean = np.concatenate(all_clean_unc) if all_clean_unc else np.array([])
    all_noisy = np.concatenate(all_noisy_unc) if all_noisy_unc else np.array([])
    mw_stat, mw_p = mann_whitney(all_clean, all_noisy)

    corr_rows = []
    for metric, values in [
        ("mean_raw_user_variance", raw_user_y),
        ("mean_raw_item_variance", raw_item_y),
        ("mean_post_user_variance", post_user_y),
        ("mean_post_item_variance", post_item_y),
        ("mean_clean_edge_uncertainty", np.array([row["mean_clean_edge_uncertainty"] for row in summary_rows])),
        ("mean_noisy_edge_uncertainty", np.array([row["mean_noisy_edge_uncertainty"] for row in summary_rows])),
    ]:
        mask = np.isfinite(values)
        pearson_r, pearson_p = pearson_corr(x[mask], values[mask])
        spearman_r, spearman_p = spearman_corr(x[mask], values[mask])
        corr_rows.append({
            "metric": metric,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "mannwhitney_stat_clean_vs_noisy": mw_stat if metric == "mean_noisy_edge_uncertainty" else "",
            "mannwhitney_p_clean_vs_noisy": mw_p if metric == "mean_noisy_edge_uncertainty" else "",
        })
    write_csv(
        save_dir / "correlation_stats.csv",
        corr_rows,
        [
            "metric", "pearson_r", "pearson_p", "spearman_r", "spearman_p",
            "mannwhitney_stat_clean_vs_noisy", "mannwhitney_p_clean_vs_noisy",
        ],
    )

    print(f"Saved noise-uncertainty figures and CSV files to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="UnGGCN visual experiments")
    parser.add_argument("--experiment", choices=["motivation", "noise_uncertainty"], required=True)
    parser.add_argument("--noise_ratio", type=float, default=0.3)
    parser.add_argument("--noise_ratios", type=float, nargs="+", default=[0.0,0.3])
    parser.add_argument("--save_dir", type=str, default="./analysis_results/")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="maximum number of same-user clean/noisy pairs in the motivation experiment",
    )
    parser.add_argument("--edge_sample_size", type=int, default=5000)
    known, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = model_parse_args()
    set_seed(args.seed)

    if known.experiment == "motivation":
        run_motivation(args, known.noise_ratio, known.save_dir, known.max_samples)
    else:
        run_noise_uncertainty(args, sorted(known.noise_ratios), known.save_dir, known.edge_sample_size)


if __name__ == "__main__":
    main()
