# UnGGCN

This is the official implementation of the paper:

> **Uncertainty-aware Gaussian Graph Convolution for Implicit Feedback Recommendation**

UnGGCN represents users and items as Gaussian distributions, where the mean captures latent preferences and the variance quantifies dimension-wise uncertainty. An uncertainty-guided message passing mechanism derives adaptive weights from variance to suppress high-uncertainty dimensions while preserving reliable signals during graph convolution.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- tqdm

## Datasets

We use three datasets: **Baby Product**, **Pet Supplies**, and **Yelp2018**.

| Dataset | Directory | #Users | #Items | #Interactions | Density | Recommended `--beta` |
|---|---|---:|---:|---:|---:|---:|
| Baby Product | `baby` | 16,916 | 7,627 | 257,152 | 0.1993% | 0.5 |
| Pet Supplies | `pet` | 112,730 | 37,966 | 1,822,357 | 0.0426% | 0.4 |
| Yelp2018 | `yelp2018` | 77,277 | 45,638 | 2,103,895 | 0.0597% | 0.3 |

## Environment Setup

```bash
conda env create -f environment.yml
conda activate unggcn
```

## Quick Start

```bash
# Baby Product
python main.py --dataset baby --lr 0.001 --beta 0.5

# Pet Supplies
python main.py --dataset pet --lr 0.001 --beta 0.4

# Yelp2018
python main.py --dataset yelp2018 --lr 0.001 --beta 0.3
```

### With noise injection

```bash
python main.py --dataset baby --lr 0.001 --beta 0.5 --noise_ratio 0.1
python main.py --dataset pet --lr 0.001 --beta 0.4 --noise_ratio 0.1
python main.py --dataset yelp2018 --lr 0.001 --beta 0.3 --noise_ratio 0.1
```

## Key Hyperparameters

| Parameter | Description | Default |
|---|---|---|
| `--dim` | Embedding dimension | 64 |
| `--context_hops` | Number of GCN layers (L) | 3 |
| `--lr` | Learning rate | 0.001 |
| `--beta` | Temperature parameter (τ); use 0.5 for `baby`, 0.4 for `pet`, and 0.3 for `yelp2018` | 0.5 |
| `--noise_ratio` | Ratio of injected noise | 0.0 |
| `--batch_size` | Training batch size | 2048 |
