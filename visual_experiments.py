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


def checkpoint_candidates(model_dir, dataset, noise_ratio, dim, context_hops, beta, lr, disable_ump):
    model_dir = Path(model_dir)
    suffix = "_noatt" if disable_ump else ""
    exact = model_dir / (
        f"model_dataset_{dataset}_noise_{noise_ratio}_dim{dim}_hops{context_hops}"
        f"_beta{beta}_lr{lr}{suffix}.ckpt"
    )
    if exact.exists():
        return [exact]

    pattern = re.compile(
        rf"^model_dataset_{re.escape(dataset)}_noise_([^_]+)_dim(\d+)_hops(\d+)"
        rf"_beta([^_]+)_lr([^_]+?)(?:(_noatt)|_sweep_[^.]*)?\.ckpt$"
    )
    matches = []
    if model_dir.is_dir():
        for path in model_dir.iterdir():
            match = pattern.match(path.name)
            if not match:
                continue
            f_noise, f_dim, f_hops, f_beta, f_lr, f_noatt = match.groups()
            try:
                same_config = (
                    math.isclose(float(f_noise), float(noise_ratio))
                    and int(f_dim) == int(dim)
                    and int(f_hops) == int(context_hops)
                    and math.isclose(float(f_beta), float(beta))
                    and math.isclose(float(f_lr), float(lr))
                )
            except ValueError:
                same_config = False
            if same_config and bool(f_noatt) == bool(disable_ump):
                matches.append(path)
    return sorted(matches)


def find_checkpoint(args):
    candidates = checkpoint_candidates(
        args.model_dir, args.dataset, args.noise_ratio, args.dim,
        args.context_hops, args.beta, args.lr, args.disable_ump
    )
    if candidates:
        return candidates[0]
    suffix = " --disable_ump" if args.disable_ump else ""
    command = (
        f"python main.py --dataset {args.dataset} --noise_ratio {args.noise_ratio} "
        f"--dim {args.dim} --context_hops {args.context_hops} --beta {args.beta} "
        f"--lr {args.lr}{suffix}"
    )
    raise FileNotFoundError(
        f"Missing checkpoint for dataset={args.dataset}, noise_ratio={args.noise_ratio}, "
        f"disable_ump={args.disable_ump}. Train it with:\n  {command}"
    )


def load_model_bundle(noise_ratio, base_args):
    args = copy.deepcopy(base_args)
    args.noise_ratio = float(noise_ratio)
    args.cuda = bool(args.cuda and torch.cuda.is_available())
    train_cf, user_dict, n_params, norm_mat, norm_mat_var = load_data(args)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model = UnGGSL(n_params, args, norm_mat, norm_mat_var).to(device)
    ckpt_path = find_checkpoint(args)
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
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


def run_motivation(args, noise_ratio, save_root, max_samples):
    clean = load_model_bundle(0.0, args)
    noisy = load_model_bundle(noise_ratio, args)
    clean_reps = representations(clean["model"])
    noisy_reps = representations(noisy["model"])

    noisy_edges = noisy["n_params"]["injected_noise_edges"]
    if len(noisy_edges) > 0:
        candidate_users = np.unique(noisy_edges[:, 0])
    else:
        candidate_users = np.arange(noisy["n_params"]["n_users"])

    clean_user_var = clean_reps["user_var"].detach().cpu().numpy()
    noisy_user_var = noisy_reps["user_var"].detach().cpu().numpy()
    delta = np.log(noisy_user_var[candidate_users] + 1e-12) - np.log(clean_user_var[candidate_users] + 1e-12)
    user_scores = delta.mean(axis=1)
    top_idx = np.argsort(user_scores)[::-1][:max_samples]
    selected_users = candidate_users[top_idx]
    heatmap = delta[top_idx]

    dim_order = np.argsort(heatmap.mean(axis=0))[::-1]
    heatmap_sorted = heatmap[:, dim_order]
    save_dir = Path(save_root) / "motivation"
    ensure_dir(save_dir)

    heat_rows = []
    for row_id, user_id in enumerate(selected_users):
        for rank, dim in enumerate(dim_order):
            heat_rows.append({
                "user_id": int(user_id),
                "dimension": int(dim),
                "dimension_rank": int(rank),
                "clean_var": float(clean_user_var[user_id, dim]),
                "noisy_var": float(noisy_user_var[user_id, dim]),
                "delta_var": float(noisy_user_var[user_id, dim] - clean_user_var[user_id, dim]),
                "delta_log_ratio": float(heatmap[row_id, dim]),
            })
    write_csv(
        save_dir / "dimension_variance_heatmap.csv",
        heat_rows,
        ["user_id", "dimension", "dimension_rank", "clean_var", "noisy_var", "delta_var", "delta_log_ratio"],
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(heatmap_sorted, aspect="auto", cmap="coolwarm")
    ax.set_title(f"Dimension-wise Variance Increase after Noise Injection (noise={noise_ratio})")
    ax.set_xlabel("Embedding dimensions sorted by mean variance increase")
    ax.set_ylabel("Selected users")
    ax.set_yticks(np.arange(len(selected_users)))
    ax.set_yticklabels([str(int(u)) for u in selected_users], fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log noisy variance - log clean variance")
    fig.tight_layout()
    fig.savefig(save_dir / "dimension_variance_heatmap.png", dpi=220)
    plt.close(fig)

    mean_clean = clean_user_var[selected_users].mean(axis=0)
    mean_noisy = noisy_user_var[selected_users].mean(axis=0)
    mean_delta = np.log(mean_noisy + 1e-12) - np.log(mean_clean + 1e-12)
    mean_weight = attention_from_variance(
        torch.as_tensor(mean_noisy, dtype=torch.float32), args.beta
    ).numpy()
    relation_order = np.argsort(mean_noisy)[::-1]

    relation_rows = []
    for rank, dim in enumerate(relation_order):
        relation_rows.append({
            "dimension": int(dim),
            "dimension_rank": int(rank),
            "mean_clean_var": float(mean_clean[dim]),
            "mean_noisy_var": float(mean_noisy[dim]),
            "delta_log_ratio": float(mean_delta[dim]),
            "attention_weight": float(mean_weight[dim]),
        })
    write_csv(
        save_dir / "variance_weight_relation.csv",
        relation_rows,
        ["dimension", "dimension_rank", "mean_clean_var", "mean_noisy_var", "delta_log_ratio", "attention_weight"],
    )

    x = np.arange(len(relation_order))
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(x, mean_noisy[relation_order], color="#C44E52", linewidth=2, label="Mean variance")
    ax1.set_xlabel("Embedding dimensions sorted by variance")
    ax1.set_ylabel("Mean variance", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(x, mean_weight[relation_order], color="#4C72B0", linewidth=2, label="Uncertainty weight")
    ax2.set_ylabel("Uncertainty-guided weight", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_title(f"High-Variance Dimensions Receive Lower UMP Weights (noise={noise_ratio})")
    fig.tight_layout()
    fig.savefig(save_dir / "variance_weight_relation.png", dpi=220)
    plt.close(fig)

    print(f"Saved motivation figures and CSV files to {save_dir}")


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
    parser.add_argument("--noise_ratios", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--save_dir", type=str, default="./analysis_results/")
    parser.add_argument("--max_samples", type=int, default=24)
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
