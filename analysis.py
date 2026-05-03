"""
方差分析脚本：加载不同噪声比例下的checkpoint，
用分组柱状图对比各噪声比例下传播前/后方差，User 和 Item 各一张图。

用法:
  python analyze_variance.py --noise_ratios 0.0 0.1 0.3 0.5 \
      --dataset pet --dim 64 --context_hops 3 --beta 0.5 --lr 0.01
"""

import os
import copy
import random
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.parser import parse_args as model_parse_args
from utils.data_loader import load_data
from modules.UnGGSL import UnGGSL


# ─── 工具函数 ───

def find_checkpoint(model_dir, dataset, noise_ratio, dim, context_hops, beta, lr):
    exact = f'model_dataset_{dataset}_noise_{noise_ratio}_dim{dim}_hops{context_hops}_beta{beta}_lr{lr}_noatt.ckpt'
    path = os.path.join(model_dir, exact)
    if os.path.exists(path):
        return path
    if os.path.isdir(model_dir):
        prefix = f'model_dataset_{dataset}_noise_{noise_ratio}'
        for f in os.listdir(model_dir):
            if f.startswith(prefix) and f.endswith('.ckpt'):
                return os.path.join(model_dir, f)
    return None


def load_model_for_noise(noise_ratio, base_args):
    args = copy.deepcopy(base_args)
    args.noise_ratio = noise_ratio
    train_cf, user_dict, n_params, norm_mat, norm_mat_var = load_data(args)
    device = torch.device("cuda:0") if args.cuda and torch.cuda.is_available() else torch.device("cpu")
    model = UnGGSL(n_params, args, norm_mat, norm_mat_var).to(device)
    ckpt_path = find_checkpoint(args.model_dir, args.dataset, noise_ratio,
                                args.dim, args.context_hops, args.beta, args.lr)
    if ckpt_path:
        print(f"  加载checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"  [WARNING] 未找到 noise={noise_ratio} 的checkpoint，使用随机初始化")
    model.eval()
    return model


@torch.no_grad()
def extract_variances(model):
    raw_user_var = torch.exp(2 * model.user_logsigma).cpu().numpy()
    raw_item_var = torch.exp(2 * model.item_logsigma).cpu().numpy()
    _, _, post_user_var, post_item_var = model.generate(split=True)
    return {
        'raw_user_var': raw_user_var,
        'raw_item_var': raw_item_var,
        'post_user_var': post_user_var.cpu().numpy(),
        'post_item_var': post_item_var.cpu().numpy(),
    }


# ─── 绘图 ───

def plot_grouped_bar(all_vars, noise_ratios, save_dir):
    """User 一张图，Item 一张图，每张图 x 轴噪声比例，两根柱子 Pre/Post"""
    labels = [str(nr) for nr in noise_ratios]
    x = np.arange(len(noise_ratios))
    width = 0.3

    # ── User ──
    pre_user = [all_vars[nr]['raw_user_var'].mean() for nr in noise_ratios]
    post_user = [all_vars[nr]['post_user_var'].mean() for nr in noise_ratios]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, pre_user, width, label='Pre-GCN', color='#999999')
    ax.bar(x + width / 2, post_user, width, label='Post-GCN', color='#4C72B0')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Noise Ratio')
    ax.set_ylabel('Mean Variance')
    ax.set_title('User Variance: Pre-GCN vs Post-GCN')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'user_variance_pre_vs_post.png'), dpi=200)
    plt.close()
    print("  ✓ user_variance_pre_vs_post.png")

    # ── Item ──
    pre_item = [all_vars[nr]['raw_item_var'].mean() for nr in noise_ratios]
    post_item = [all_vars[nr]['post_item_var'].mean() for nr in noise_ratios]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, pre_item, width, label='Pre-GCN', color='#999999')
    ax.bar(x + width / 2, post_item, width, label='Post-GCN', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Noise Ratio')
    ax.set_ylabel('Mean Variance')
    ax.set_title('Item Variance: Pre-GCN vs Post-GCN')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'item_variance_pre_vs_post.png'), dpi=200)
    plt.close()
    print("  ✓ item_variance_pre_vs_post.png")


def print_table(all_vars, noise_ratios):
    print("\n" + "=" * 75)
    print(f"{'Noise':>7} | {'Pre User':>12} | {'Post User':>12} | {'Pre Item':>12} | {'Post Item':>12}")
    print("-" * 75)
    for nr in noise_ratios:
        v = all_vars[nr]
        print(f"{nr:>7.2f} | "
              f"{v['raw_user_var'].mean():>12.6f} | {v['post_user_var'].mean():>12.6f} | "
              f"{v['raw_item_var'].mean():>12.6f} | {v['post_item_var'].mean():>12.6f}")
    print("=" * 75)


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Variance Analysis")
    parser.add_argument('--noise_ratios', type=float, nargs='+', default=[0.0, 0.1, 0.3, 0.5])
    parser.add_argument('--save_dir', type=str, default='./analysis_results/')
    known, remaining = parser.parse_known_args()

    noise_ratios = sorted(known.noise_ratios)
    save_dir = known.save_dir
    os.makedirs(save_dir, exist_ok=True)

    import sys
    sys.argv = [sys.argv[0]] + remaining
    base_args = model_parse_args()

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_vars = {}
    for nr in noise_ratios:
        print(f"\n{'='*50}")
        print(f"  noise_ratio = {nr}")
        print(f"{'='*50}")
        model = load_model_for_noise(nr, base_args)
        all_vars[nr] = extract_variances(model)

    print_table(all_vars, noise_ratios)

    print(f"\n生成图表到 {save_dir} ...")
    plot_grouped_bar(all_vars, noise_ratios, save_dir)
    print("\n分析完成！")


if __name__ == '__main__':
    main()