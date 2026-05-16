import argparse
import ast


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_ks(value):
    if isinstance(value, (list, tuple)):
        return [int(k) for k in value]
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = [int(k) for k in value.split(",")]
    if isinstance(parsed, int):
        parsed = [parsed]
    return [int(k) for k in parsed]


def parse_args():
    parser = argparse.ArgumentParser(description="UnGGCN")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="baby",
                        help="Choose a dataset:[amazon,ali,prime_pantry,office]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== # 
    parser.add_argument("--gnn", nargs="?", default="unggcn",
                        help="Choose a recommender:[lightgcn, ngcf,vgae,vgae_w]")
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--batch_test_flag", type=str2bool, default=True, help="use batched item evaluation or not")
    parser.add_argument("--lw", type=float, default=1.0, help="weight of normal-gamma-loss")

    parser.add_argument("--K", type=int, default=1, help="number of negative in K-pair loss")

    parser.add_argument("--n_negs", type=int, default=1, help="number of candidate negative")
    parser.add_argument("--cuda", type=str2bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=5, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[10,20]',
                        help='K of ndcg@K, recall@K')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--context_hops", type=int, default=3, help="hop")
    parser.add_argument("--beta",type=float, default=0.7, help="beta for softplus")
    parser.add_argument("--prior_alpha", type=float, default=2.0,
                        help="alpha parameter of the Normal-Gamma prior")
    parser.add_argument("--prior_beta", type=float, default=1.0,
                        help="beta parameter of the Normal-Gamma prior")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="directory to save logs")
    parser.add_argument("--logsigma",type=float, default=0.0, help="init value for log sigma")
    parser.add_argument("--noise_ratio", type=float, default=0.0, help="ratio of noisy training data")
    parser.add_argument("--seed", type=int, default=2025, help="random seed")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="early stopping patience counted on validation checks")
    # ===== save model ===== #
    parser.add_argument("--save", type=str2bool, default=True, help="save model or not")
    parser.add_argument(
        "--out_dir", type=str, default="./recordings/", help="output directory "
    )
    parser.add_argument(
        "--model_dir", type=str, default="./models/", help="dir for saving")
    return parser.parse_args()
