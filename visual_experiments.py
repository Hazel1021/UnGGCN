import argparse
import copy
import logging
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
from matplotlib import font_manager

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
if hasattr(font_manager, "_load_fontmanager"):
    font_manager.fontManager = font_manager._load_fontmanager(try_read_cache=False)
arial_font_path = font_manager.findfont("Arial", fallback_to_default=False)
print(f"Using Matplotlib font: {arial_font_path}")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch

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


def read_remap_list(path, item_offset=0):
    id_map = {}
    path = Path(path)
    if not path.is_file():
        return id_map

    with open(path, "r") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            org_id, remap_id = parts[0], int(parts[1])
            id_map[remap_id - item_offset] = org_id
    return id_map


def load_original_id_maps(args, n_users):
    data_dir = Path(args.data_path) / args.dataset
    return {
        "user": read_remap_list(data_dir / "user_list.txt"),
        "item": read_remap_list(data_dir / "item_list.txt", item_offset=n_users),
    }


def original_id(id_map, remap_id):
    return id_map.get(int(remap_id), str(int(remap_id)))


def expected_checkpoint_names(args):
    return [
        (
            f"modelmodel_dataset_{args.dataset}_dim{args.dim}"
            f"_hops{args.context_hops}_lr{args.lr}_lw{args.lw}"
            f"_beta{args.beta}_prioralpha{args.prior_alpha}"
            f"_priorbeta{args.prior_beta}_noise_{args.noise_ratio}_noatt.ckpt"
        ),
        (
            f"model_dataset_{args.dataset}_noise_{args.noise_ratio}"
            f"_dim{args.dim}_hops{args.context_hops}"
            f"_beta{args.beta}_lr{args.lr}_noatt.ckpt"
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
    _, _, n_params, norm_mat, norm_mat_var = load_data(args)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model = UnGGSL(n_params, args, norm_mat, norm_mat_var).to(device)
    ckpt_path = find_checkpoint(args)
    print(f"Loading checkpoint: {ckpt_path}")
    state = unwrap_state_dict(torch.load(ckpt_path, map_location=device))
    model.load_state_dict(state)
    model.eval()
    return {
        "model": model,
        "device": device,
        "n_params": n_params,
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


def match_clean_noisy_edges(clean_edges, noisy_edges, seed):
    clean_by_user = {}
    for user, item in np.asarray(clean_edges, dtype=np.int64).reshape(-1, 2):
        clean_by_user.setdefault(int(user), []).append(int(item))

    noisy_by_user = {}
    for user, item in np.asarray(noisy_edges, dtype=np.int64).reshape(-1, 2):
        noisy_by_user.setdefault(int(user), []).append(int(item))

    eligible_users = sorted(set(clean_by_user) & set(noisy_by_user))
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible_users)

    matched_clean = []
    matched_noisy = []
    for user in eligible_users:
        matched_clean.append((user, int(rng.choice(clean_by_user[user]))))
        matched_noisy.append((user, int(rng.choice(noisy_by_user[user]))))
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


def sample_uncertainty_pair(edges, seed):
    rng = np.random.default_rng(seed)
    pair_idx = int(rng.integers(len(edges)))
    user_id = int(edges[pair_idx, 0])
    item_id = int(edges[pair_idx, 1])
    return user_id, item_id


def run_motivation(args, noise_ratio, save_root):
    if noise_ratio <= 0:
        raise ValueError("The motivation experiment requires --noise_ratio > 0.")

    bundle = load_model_bundle(noise_ratio, args)
    reps = representations(bundle["model"])
    clean_edges, noisy_edges = match_clean_noisy_edges(
        bundle["n_params"]["clean_train_cf"],
        bundle["n_params"]["injected_noise_edges"],
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

    mean_clean_dim = clean_dim_unc.mean(axis=0)
    mean_noisy_dim = noisy_dim_unc.mean(axis=0)
    mean_delta_dim = mean_noisy_dim - mean_clean_dim
    dim_order = np.argsort(mean_delta_dim)[::-1]

    id_maps = load_original_id_maps(args, bundle["n_params"]["n_users"])
    heat_user_id, heat_item_id = sample_uncertainty_pair(
        bundle["n_params"]["injected_noise_edges"],
        args.seed + 2025,
    )
    heat_user_org_id = original_id(id_maps["user"], heat_user_id)
    heat_item_org_id = original_id(id_maps["item"], heat_item_id)
    user_init_var = torch.exp(2.0 * bundle["model"].user_logsigma).detach().cpu().numpy()
    item_init_var = torch.exp(2.0 * bundle["model"].item_logsigma).detach().cpu().numpy()
    heatmap_values = np.vstack([
        user_init_var[heat_user_id],
        item_init_var[heat_item_id],
    ])

    save_dir = Path(save_root) / "motivation"
    ensure_dir(save_dir)

    title_fontsize = 15
    label_fontsize = 13
    tick_fontsize = 12
    colorbar_fontsize = 12
    title_pad = 14
    box_face_color = "#69C2C9"
    box_median_color = "#ED7D32"
    bar_positive_color = "#339FC4"
    bar_negative_color = "#ED7D32"
    heatmap_cmap = "GnBu_r"

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    box = axes[0].boxplot(
        [clean_edge_unc, noisy_edge_unc],
        labels=["clean", "noisy"],
        showfliers=False,
        patch_artist=True,
    )
    for patch in box["boxes"]:
        patch.set_facecolor(box_face_color)
        patch.set_alpha(0.9)
    for median in box["medians"]:
        median.set_color(box_median_color)
        median.set_linewidth(1.8)
    axes[0].set_title(
        "(a) User-level predictive uncertainty",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=title_pad,
    )
    axes[0].set_ylabel(r"$\mathrm{Var}[Y_{ui}]$", fontsize=label_fontsize)
    axes[0].tick_params(axis="both", labelsize=tick_fontsize)
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(dim_order))
    sorted_delta = mean_delta_dim[dim_order]
    colors = np.where(sorted_delta >= 0, bar_positive_color, bar_negative_color)
    axes[1].bar(x, sorted_delta, color=colors, width=0.85)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title(
        "(b) Dimension localization of noisy uncertainty",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=title_pad,
    )
    axes[1].set_xlabel("Dimensions sorted by mean noisy-clean delta", fontsize=label_fontsize)
    axes[1].set_ylabel(r"Mean $\Delta V_k$ (noisy - clean)", fontsize=label_fontsize)
    axes[1].tick_params(axis="both", labelsize=tick_fontsize)
    axes[1].grid(axis="y", alpha=0.25)

    im = axes[2].imshow(
        heatmap_values,
        aspect="auto",
        cmap=heatmap_cmap,
    )
    axes[2].set_title(
        "(c) Sampled noisy interaction uncertainty heatmap",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=title_pad,
    )
    axes[2].set_xlabel("Embedding dimension", fontsize=label_fontsize)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels([
        f"User {heat_user_id}",
        f"Item {heat_item_id}",
    ])
    axes[2].tick_params(axis="both", labelsize=tick_fontsize)
    colorbar = fig.colorbar(im, ax=axes[2])
    colorbar.set_label("Learned initial variance", fontsize=colorbar_fontsize)
    colorbar.ax.tick_params(labelsize=tick_fontsize)

    fig.tight_layout()
    fig.savefig(save_dir / "motivation_validation.png", dpi=220)
    fig.savefig(save_dir / "motivation_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved motivation figure ({len(noisy_edges)} matched pairs) to {save_dir}")
    print(
        "Sampled noisy pair for subplot (c): "
        f"user remap_id={heat_user_id}, org_id={heat_user_org_id}; "
        f"item remap_id={heat_item_id}, org_id={heat_item_org_id}"
    )


def main():
    parser = argparse.ArgumentParser(description="UnGGCN visual experiments")
    parser.add_argument("--noise_ratio", type=float, default=0.3)
    parser.add_argument("--save_dir", type=str, default="./analysis_results/")
    known, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = model_parse_args()
    set_seed(args.seed)

    run_motivation(args, known.noise_ratio, known.save_dir)


if __name__ == "__main__":
    main()
