"""
Analyze the learned initial variances stored in checkpoints.

The learnable initial variance is:

    variance = exp(2 * logsigma)

This script does not run graph convolution. It directly reads
user_logsigma and item_logsigma from checkpoints trained with different
noise ratios.

Examples:
  python analysis.py --dataset baby --noise_ratios 0.0 0.1 0.2 0.3

  python analysis.py --dataset baby --noise_ratios 0.0 0.1 0.3 0.5 \
      --variants full noconv
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.parser import parse_args as model_parse_args


VARIANT_SUFFIXES = {
    "full": "",
    "noconv": "_noconv",
}


def expected_checkpoint_names(args, noise_ratio, variant):
    suffix = VARIANT_SUFFIXES[variant]
    return [
        (
            f"modelmodel_dataset_{args.dataset}_dim{args.dim}"
            f"_hops{args.context_hops}_lr{args.lr}_lw{args.lw}"
            f"_beta{args.beta}_prioralpha{args.prior_alpha}"
            f"_priorbeta{args.prior_beta}_noise_{noise_ratio}"
            f"{suffix}.ckpt"
        ),
        (
            f"model_dataset_{args.dataset}_noise_{noise_ratio}"
            f"_dim{args.dim}_hops{args.context_hops}"
            f"_beta{args.beta}_lr{args.lr}{suffix}.ckpt"
        ),
    ]


def checkpoint_matches(path, args, noise_ratio, variant):
    suffix = re.escape(VARIANT_SUFFIXES[variant])
    patterns = [
        (
            re.compile(
                rf"^modelmodel_dataset_{re.escape(args.dataset)}"
                rf"_dim(?P<dim>\d+)_hops(?P<hops>\d+)"
                rf"_lr(?P<lr>[^_]+)_lw(?P<lw>[^_]+)"
                rf"_beta(?P<beta>[^_]+)_prioralpha(?P<alpha>[^_]+)"
                rf"_priorbeta(?P<prior_beta>[^_]+)_noise_(?P<noise>[^_]+)"
                rf"{suffix}\.ckpt$"
            ),
            True,
        ),
        (
            re.compile(
                rf"^model_dataset_{re.escape(args.dataset)}"
                rf"_noise_(?P<noise>[^_]+)_dim(?P<dim>\d+)"
                rf"_hops(?P<hops>\d+)_beta(?P<beta>[^_]+)"
                rf"_lr(?P<lr>[^_]+){suffix}\.ckpt$"
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
                and math.isclose(float(values["noise"]), float(noise_ratio))
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


def find_checkpoint(args, noise_ratio, variant):
    model_root = Path(args.model_dir)
    search_dirs = [model_root / args.dataset, model_root]
    expected_names = expected_checkpoint_names(args, noise_ratio, variant)

    for directory in search_dirs:
        for expected_name in expected_names:
            exact_path = directory / expected_name
            if exact_path.is_file():
                return exact_path

    matches = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        matches.extend(
            path for path in directory.glob("*.ckpt")
            if checkpoint_matches(path, args, noise_ratio, variant)
        )

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        paths = "\n  ".join(str(path) for path in unique_matches)
        raise RuntimeError(
            f"Multiple checkpoints match noise={noise_ratio}, variant={variant}:\n  {paths}"
        )

    searched = "\n  ".join(
        str(directory / expected_name)
        for directory in search_dirs
        for expected_name in expected_names
    )
    raise FileNotFoundError(
        f"Missing checkpoint for noise={noise_ratio}, variant={variant}.\n"
        f"Expected one of:\n  {searched}"
    )


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state_dict-like mapping.")

    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def find_parameter(state_dict, parameter_name):
    direct = state_dict.get(parameter_name)
    if torch.is_tensor(direct):
        return direct

    matches = [
        value for key, value in state_dict.items()
        if key.endswith("." + parameter_name) and torch.is_tensor(value)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Multiple parameters end with '{parameter_name}'.")
    raise KeyError(f"Checkpoint does not contain '{parameter_name}'.")


@torch.no_grad()
def load_learned_initial_variances(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)
    user_logsigma = find_parameter(state_dict, "user_logsigma").float()
    item_logsigma = find_parameter(state_dict, "item_logsigma").float()

    return {
        "user_variance": torch.exp(2.0 * user_logsigma).numpy(),
        "item_variance": torch.exp(2.0 * item_logsigma).numpy(),
    }


def summarize(values):
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(flattened)),
        "std": float(np.std(flattened)),
        "median": float(np.median(flattened)),
        "q25": float(np.quantile(flattened, 0.25)),
        "q75": float(np.quantile(flattened, 0.75)),
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
        "num_values": int(flattened.size),
    }


def save_statistics(rows, save_dir):
    output_path = Path(save_dir) / "learned_initial_variance.csv"
    fieldnames = [
        "variant", "noise_ratio", "entity", "mean", "std", "median",
        "q25", "q75", "min", "max", "num_values", "checkpoint",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {output_path}")


def plot_variance(rows, entity, save_dir):
    entity_rows = [row for row in rows if row["entity"] == entity]
    colors = {"full": "#4C72B0", "noconv": "#DD8452"}
    labels = {"full": "UnGGCN", "noconv": "w/o GConv"}

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for variant in sorted({row["variant"] for row in entity_rows}):
        variant_rows = sorted(
            (row for row in entity_rows if row["variant"] == variant),
            key=lambda row: row["noise_ratio"],
        )
        x = np.asarray([row["noise_ratio"] for row in variant_rows])
        y = np.asarray([row["mean"] for row in variant_rows])
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            color=colors[variant],
            label=labels[variant],
        )

    entity_label = entity.capitalize()
    ax.set_xlabel("Noise Ratio")
    ax.set_ylabel("Mean Learned Initial Variance")
    ax.set_title(f"{entity_label} Initial Variance under Different Noise Ratios")
    ax.grid(alpha=0.25)
    if len({row["variant"] for row in entity_rows}) > 1:
        ax.legend()
    fig.tight_layout()

    output_path = Path(save_dir) / f"{entity}_learned_initial_variance.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"  Saved {output_path}")


def print_table(rows):
    print("\nLearned initial variance: exp(2 * logsigma)")
    print("=" * 76)
    print(f"{'Variant':<12}{'Noise':>8}{'Entity':>10}{'Mean':>14}{'Median':>14}{'Std':>14}")
    print("-" * 76)
    for row in rows:
        print(
            f"{row['variant']:<12}{row['noise_ratio']:>8.2f}{row['entity']:>10}"
            f"{row['mean']:>14.6f}{row['median']:>14.6f}{row['std']:>14.6f}"
        )
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Analyze learned initial variances")
    parser.add_argument(
        "--noise_ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3],
    )
    parser.add_argument(
        "--variants",
        choices=sorted(VARIANT_SUFFIXES),
        nargs="+",
        default=["full"],
        help="full uses the standard checkpoint; noconv appends _noconv",
    )
    parser.add_argument("--save_dir", type=str, default="./analysis_results/")
    known, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = model_parse_args()
    save_dir = Path(known.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in known.variants:
        for noise_ratio in sorted(set(known.noise_ratios)):
            checkpoint_path = find_checkpoint(args, noise_ratio, variant)
            print(
                f"Loading noise={noise_ratio:g}, variant={variant}: "
                f"{checkpoint_path}"
            )
            variances = load_learned_initial_variances(checkpoint_path)

            for entity, key in (
                ("user", "user_variance"),
                ("item", "item_variance"),
            ):
                rows.append({
                    "variant": variant,
                    "noise_ratio": float(noise_ratio),
                    "entity": entity,
                    **summarize(variances[key]),
                    "checkpoint": str(checkpoint_path),
                })

    rows.sort(key=lambda row: (row["variant"], row["noise_ratio"], row["entity"]))
    print_table(rows)
    save_statistics(rows, save_dir)
    plot_variance(rows, "user", save_dir)
    plot_variance(rows, "item", save_dir)


if __name__ == "__main__":
    main()
