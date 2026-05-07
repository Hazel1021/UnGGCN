import os
import random

import torch
import numpy as np
from time import time
import logging

from utils.parser import parse_args, parse_ks
from utils.data_loader import load_data
from utils.evaluate import test

n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):

    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                        train_pos_set,
                                                        n_negs*K)).to(device)
    return feed_dict


def init_logger(args):

    ump_suffix = '_noatt' if getattr(args, 'disable_ump', False) else ''
    out_dir = os.path.join(args.log_dir, args.dataset, args.gnn + ump_suffix)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = f'log_{args.dataset}_dim{args.dim}_hops{args.context_hops}_beta_{args.beta}_noise_{args.noise_ratio}.txt'
    filepath = os.path.join(out_dir, filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    logger.addHandler(logging.FileHandler(filepath))
    logger.addHandler(logging.StreamHandler())
    formatter = logging.Formatter(f'[%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S]')
    for handler in logger.handlers:
        handler.setFormatter(formatter)
    return logger


def format_metrics(prefix, metrics, k_values):
    lines = []
    for metric_name in ['ndcg', 'recall', 'precision', 'hit_ratio']:
        values = metrics[metric_name]
        metric_str = ', '.join(
            [f'{metric_name}@{k}: {values[i]:.4f}' for i, k in enumerate(k_values) if i < len(values)]
        )
        lines.append(f'  {prefix} {metric_str}')
    return lines


def train(train_args=None):
    global n_users, n_items, K, device, args

    if train_args is None:
        train_args = parse_args()
    args = train_args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if args.cuda and not torch.cuda.is_available():
        print("[warning] CUDA unavailable。")
        args.cuda = False
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    print("device:", device)

    """fix the random seed"""
    seed = getattr(args, "seed", 2025)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """build dataset"""
    train_cf, user_dict, n_params, norm_mat, norm_mat_var = load_data(args)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    print(train_cf.shape)
    print(train_cf)

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    """define model"""
    from modules.UnGGSL import UnGGSL


    """"init logger"""
    logger = init_logger(args)
    logger.info(f"model parameters: {args}")


    model = UnGGSL(n_params, args, norm_mat,norm_mat_var,logger).to(device)


    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_score = -float('inf')
    best_epoch = -1
    stopping_step = 0
    best_metrics = {}  # store all metrics for the best epoch
    best_save_path = None
    selection_split = 'valid' if user_dict['valid_user_set'] is not None else 'test'

    epoch_times = []
    k_values = parse_ks(args.Ks)

    print("start training ...")
    
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss_value, s = 0.0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                    user_dict['train_user_set'],
                                    s, s + args.batch_size,
                                    n_negs)

            batch_loss, _, _ = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss_value += batch_loss.item()
            s += args.batch_size
        train_e_t = time()

        epoch_time = train_e_t - train_s_t
        epoch_times.append(epoch_time)
        logger.info(f"Epoch {epoch}, Time: {epoch_time:.4f}s")

        if epoch % 5 == 0:
            """testing"""

            model.eval()
            test_s_t = time()

            # user_mean,item_mean,user_var,item_var = model.generate(split = True )
            
            test_ret = test(model, user_dict, n_params, mode='test')
            test_e_t = time()


            logger.info(f"Testing - epoch: {epoch}, testing time(s): {test_e_t - test_s_t:.4f}, Loss: {loss_value:.4f}")

            
            ndcg_str = " ; ".join([f"ndcg@{k}: {test_ret['ndcg'][i]:.4f}" for i, k in enumerate(k_values)])
            logger.info(f"\t\t{ndcg_str}")

            recall_str = " ; ".join([f"recall@{k}: {test_ret['recall'][i]:.4f}" for i, k in enumerate(k_values)])
            logger.info(f"\t\t{recall_str}")

            precision_str = " ; ".join([f"precision@{k}: {test_ret['precision'][i]:.4f}" for i, k in enumerate(k_values)])
            logger.info(f"\t\t{precision_str}")

            hit_ratio_str = " ; ".join([f"hit_ratio@{k}: {test_ret['hit_ratio'][i]:.4f}" for i, k in enumerate(k_values)])
            logger.info(f"\t\t{hit_ratio_str}")
            
            """validation"""
            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                valid_s_t = time()
                valid_ret = test(model, user_dict, n_params, mode='valid')
                valid_e_t = time()

                # output validation metrics
                logger.info(f"Validation - epoch: {epoch}, validation time(s): {valid_e_t - valid_s_t:.4f}, Loss: {loss_value:.4f}")

                ndcg_str = " ; ".join([f"ndcg@{k}: {valid_ret['ndcg'][i]:.4f}" for i, k in enumerate(k_values)])
                logger.info(f"\t\t{ndcg_str}")

                recall_str = " ; ".join([f"recall@{k}: {valid_ret['recall'][i]:.4f}" for i, k in enumerate(k_values)])
                logger.info(f"\t\t{recall_str}")

                precision_str = " ; ".join([f"precision@{k}: {valid_ret['precision'][i]:.4f}" for i, k in enumerate(k_values)])
                logger.info(f"\t\t{precision_str}")

                hit_ratio_str = " ; ".join([f"hit_ratio@{k}: {valid_ret['hit_ratio'][i]:.4f}" for i, k in enumerate(k_values)])
                logger.info(f"\t\t{hit_ratio_str}")

            valid_score = float(valid_ret['ndcg'][0])
            is_best = valid_score > best_valid_score
            if is_best:
                best_valid_score = valid_score
                best_epoch = epoch
                stopping_step = 0
                best_metrics = {
                    'epoch': epoch,
                    'valid_ndcg': valid_ret['ndcg'],
                    'valid_recall': valid_ret['recall'],
                    'valid_precision': valid_ret['precision'],
                    'valid_hit_ratio': valid_ret['hit_ratio'],
                    'recall': test_ret['recall'],
                    'ndcg': test_ret['ndcg'], 
                    'precision': test_ret['precision'],
                    'hit_ratio': test_ret['hit_ratio'],
                }

                if args.save:
                    os.makedirs(args.model_dir, exist_ok=True)
                    ump_suffix = '_noatt' if getattr(args, 'disable_ump', False) else ''
                    save_path = os.path.join(args.model_dir, f'model_dataset_{args.dataset}_noise_{args.noise_ratio}_dim{args.dim}_hops{args.context_hops}_beta{args.beta}_lr{args.lr}{ump_suffix}.ckpt')
                    best_save_path = save_path
                    logger.info(f"Saving best model at epoch {epoch}: valid ndcg@10={best_valid_score:.6f} -> {save_path}")
                    torch.save(model.state_dict(), save_path)
            else:
                stopping_step += 1
                logger.info(
                    f"No improvement at epoch {epoch}: valid ndcg@10={valid_score:.6f}, "
                    f"best={best_valid_score:.6f} at epoch {best_epoch}, stopping_step={stopping_step}/10"
                )

            if stopping_step >= 10:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        else:
            logger.info(f"Epoch {epoch}: Training Loss: {loss_value:.4f}, Time: {train_e_t - train_s_t:.4f}s")


    if len(epoch_times) > 0:
        epoch_times_arr = np.array(epoch_times)
        logger.info("############################## Epoch Time Statistics ##############################")
        logger.info(f"Average epoch time: {np.mean(epoch_times_arr):.2f}s (over {len(epoch_times_arr)} epochs)")
        logger.info(f"Shortest epoch time: {np.min(epoch_times_arr):.2f}s (epoch {np.argmin(epoch_times_arr)})")
        logger.info(f"Longest epoch time: {np.max(epoch_times_arr):.2f}s (epoch {np.argmax(epoch_times_arr)})")
    
    # Output best performance summary
    if best_metrics:
        logger.info(f'Best model selected by {selection_split} ndcg@10 at epoch {best_metrics["epoch"]}:')
        logger.info(f'  best {selection_split} ndcg@10: {best_valid_score:.4f}')
        if best_save_path:
            logger.info(f'  best model path: {best_save_path}')

        valid_metrics = {
            'ndcg': best_metrics['valid_ndcg'],
            'recall': best_metrics['valid_recall'],
            'precision': best_metrics['valid_precision'],
            'hit_ratio': best_metrics['valid_hit_ratio'],
        }
        test_metrics = {
            'ndcg': best_metrics['ndcg'],
            'recall': best_metrics['recall'],
            'precision': best_metrics['precision'],
            'hit_ratio': best_metrics['hit_ratio'],
        }

        logger.info('Best model validation performance:')
        for line in format_metrics('valid', valid_metrics, k_values):
            logger.info(line)

        logger.info('Best model test performance:')
        for line in format_metrics('test', test_metrics, k_values):
            logger.info(line)
    else:
        logger.info('No validation checkpoint was evaluated.')

if __name__ == '__main__':
    args = parse_args()
    print("Starting training...")
    train(args)
