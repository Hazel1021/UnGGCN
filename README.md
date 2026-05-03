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

We use three Amazon Review datasets: **Video Games**, **Baby Product**, and **Pet Supplies** with 10-core filtering.

| Dataset | #Users | #Items | #Interactions | Density |
|---|---|---|---|---|
| Video Games | 11,658 | 5,335 | 196,508 | 0.3160% |
| Baby Product | 16,916 | 7,627 | 257,152 | 0.1993% |
| Pet Supplies | 112,730 | 37,966 | 1,822,357 | 0.0426% |

## Quick Start

```bash
# Video Games
python main.py --dataset videogames --lr 0.001 

# Baby Product
python main.py --dataset baby --lr 0.001

# Pet Supplies
python main.py --dataset pet --lr 0.01
```

### With noise injection

```bash
python main.py --dataset baby --lr 0.001 --beta 0.6 --noise_ratio 0.3
```

## Key Hyperparameters

| Parameter | Description | Default |
|---|---|---|
| `--dim` | Embedding dimension | 64 |
| `--context_hops` | Number of GCN layers (L) | 3 |
| `--lr` | Learning rate | 0.001 |
| `--beta` | Temperature parameter (τ) | 0.5 |
| `--noise_ratio` | Ratio of injected noise | 0.0 |
| `--batch_size` | Training batch size | 2048 |


