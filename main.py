import os
import random

import torch
import numpy as np
import wandb
import argparse
from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping
import logging

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
n_users = 0
n_items = 0

best_state_dict = None  

# ============== Sweep ==============
sweep_config = {
    'method': 'grid', 
    'metric': {
        'name': 'best_ndcg@10',
        'goal': 'maximize'
    },
    'parameters': {
        'beta':{
            'values':[0.6]
        },
        'noise_ratio':{
            'values':[0.0,0.3,0.5]
        },
        'lr':{
            'values':[0.01]
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 50,
        's': 2
    }
}


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


def init_logger(args,sweep_id=None):

    ump_suffix = '_noatt' if getattr(args, 'disable_ump', False) else ''
    out_dir = os.path.join(args.log_dir, args.dataset, args.gnn + 'softplus' + ump_suffix)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sweep_suffix = f'_sweep_{sweep_id}' if sweep_id else ''
    filename = f'log_{args.dataset}_dim{args.dim}_hops{args.context_hops}_beta_{args.beta}_ratingvar_noise_{args.noise_ratio}_0205_{sweep_suffix}.txt'
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

def train(use_sweep=False):
    global n_users, n_items, K, device, args
    

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if args.cuda and not torch.cuda.is_available():
        print("[warning] CUDA unavailable。")
        args.cuda = False
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    print("device:", device)

    """fix the random seed"""
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    
    if use_sweep:
        wandb.init()
        config =wandb.config
        # use sweep hyperparameters to override args
        if hasattr(config, 'lr'):
            args.lr = config.lr
        if hasattr(config, 'context_hops'):
            args.context_hops = config.context_hops
        if hasattr(config, 'noise_ratio'):
            args.noise_ratio = config.noise_ratio
        if hasattr(config,'beta'):
            args.beta = config.beta
        if hasattr(config, 'logsigma'):
            args.logsigma = config.logsigma
        if hasattr(config, 'dim'):
            args.dim = config.dim
            
        wandb.config.update(vars(args), allow_val_change=True)
        
        sweep_id = wandb.run.id
    else:
        # nomal mode: initialize wandb manually, hyperparameters from args
        wandb.login()
        wandb.init(
            project="UnGGCN",
            name=f"{args.dataset}_hops{args.context_hops}_beta{args.beta}_noise{args.noise_ratio}_beta{args.beta}_dim{args.dim}_disable_ump{args.disable_ump}",
            config=vars(args),
            notes="UnGGCN Experiment"
        )
        sweep_id = None
        
        
    


    """build dataset"""
    train_cf, user_dict, n_params, norm_mat, norm_mat_var = load_data(args)
    train_cf_size = len(train_cf)
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

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    best_metrics = {}  # store all metrics for the best epoch

    epoch_times = []

    print("start training ...")
    
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
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

            loss += batch_loss
            s += args.batch_size
        train_e_t = time()

        epoch_time = train_e_t - train_s_t
        epoch_times.append(epoch_time)
        logger.info(f"Epoch {epoch}, Time: {epoch_time:.4f}s")

        wandb.log({
                'epoch': epoch,
                'train_loss': loss.item(),
                'train_time': train_e_t - train_s_t
            })


        if epoch % 5 == 0:
                # k values for evaluation
            k_values = eval(args.Ks) if isinstance(args.Ks, str) else args.Ks

            """testing"""

            model.eval()
            test_s_t = time()

            # user_mean,item_mean,user_var,item_var = model.generate(split = True )
            
            test_ret = test(model, user_dict, n_params, mode='test')
            test_e_t = time()


            logger.info(f"Testing - epoch: {epoch}, testing time(s): {test_e_t - test_s_t:.4f}, Loss: {loss.item():.4f}")

            
            ndcg_str = " ; ".join([f"ndcg@{k}: {test_ret['ndcg'][i]:.4f}" for i, k in enumerate(k_values)])
            logger.info(f"\t\t{ndcg_str}")

            recall_str = " ; ".join([f"recall@{k}: {test_ret['recall'][i]:.4f}" for i, k in enumerate(k_values)])
            logger.info(f"\t\t{recall_str}")

            precision_str = " ; ".join([f"precision@{k}: {test_ret['precision'][i]:.4f}" for i, k in enumerate(k_values)])
            logger.info(f"\t\t{precision_str}")

            hit_ratio_str = " ; ".join([f"hit_ratio@{k}: {test_ret['hit_ratio'][i]:.4f}" for i, k in enumerate(k_values)])
            logger.info(f"\t\t{hit_ratio_str}")
            
            # record all metrics
            test_log = {
                'epoch': epoch,
                'test_recall@10': test_ret['recall'][1],
                'test_ndcg@10': test_ret['ndcg'][1],
                'test_precision@10': test_ret['precision'][1],
                'test_hit_ratio@10': test_ret['hit_ratio'][1],
                'test_time': test_e_t - test_s_t
            }
            # record metrics for all k values
            for i, k in enumerate(k_values):
                test_log[f'test_recall@{k}'] = test_ret['recall'][i]
                test_log[f'test_ndcg@{k}'] = test_ret['ndcg'][i]
                test_log[f'test_precision@{k}'] = test_ret['precision'][i]
                test_log[f'test_hit_ratio@{k}'] = test_ret['hit_ratio'][i]

            """validation"""
            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                valid_s_t = time()
                valid_ret = test(model, user_dict, n_params, mode='valid')
                valid_e_t = time()

                
            # record validation metrics
                test_log['valid_recall@10'] = valid_ret['recall'][0]
                test_log['valid_ndcg@10'] = valid_ret['ndcg'][0]
                test_log['valid_precision@10'] = valid_ret['precision'][0]
                test_log['valid_hit_ratio@10'] = valid_ret['hit_ratio'][0]
                test_log['valid_time'] = valid_e_t - valid_s_t
                
                wandb.log(test_log)

                # output validation metrics
                logger.info(f"Validation - epoch: {epoch}, validation time(s): {valid_e_t - valid_s_t:.4f}, Loss: {loss.item():.4f}")

                ndcg_str = " ; ".join([f"ndcg@{k}: {valid_ret['ndcg'][i]:.4f}" for i, k in enumerate(k_values)])
                logger.info(f"\t\t{ndcg_str}")

                recall_str = " ; ".join([f"recall@{k}: {valid_ret['recall'][i]:.4f}" for i, k in enumerate(k_values)])
                logger.info(f"\t\t{recall_str}")

                precision_str = " ; ".join([f"precision@{k}: {valid_ret['precision'][i]:.4f}" for i, k in enumerate(k_values)])
                logger.info(f"\t\t{precision_str}")

                hit_ratio_str = " ; ".join([f"hit_ratio@{k}: {valid_ret['hit_ratio'][i]:.4f}" for i, k in enumerate(k_values)])
                logger.info(f"\t\t{hit_ratio_str}")

            """early stopping"""
            cur_best_pre_0, stopping_step, should_stop = early_stopping(logger,valid_ret['ndcg'][1], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)

            

            if valid_ret['ndcg'][1] == cur_best_pre_0:
                best_metrics = {
                    'epoch': epoch,
                    'recall': test_ret['recall'],
                    'ndcg': test_ret['ndcg'], 
                    'precision': test_ret['precision'],
                    'hit_ratio': test_ret['hit_ratio'],
                }
                

                wandb.run.summary['best_epoch'] = epoch
                wandb.run.summary['best_ndcg@5'] = test_ret['ndcg'][0]
                wandb.run.summary['best_ndcg@10'] = test_ret['ndcg'][1] 
                wandb.run.summary['best_ndcg@20'] = test_ret['ndcg'][2] if len(test_ret['ndcg']) > 2 else test_ret['ndcg'][1]
                wandb.run.summary['best_recall@5'] = test_ret['recall'][0]
                wandb.run.summary['best_recall@10'] = test_ret['recall'][1] 
                wandb.run.summary['best_recall@20'] = test_ret['recall'][2] if len(test_ret['recall']) > 2 else test_ret['recall'][1]
                
            
            if should_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            """save weight"""
            if valid_ret['ndcg'][1] == cur_best_pre_0 and args.save:
                ump_suffix = '_noatt' if getattr(args, 'disable_ump', False) else ''
                if sweep_id:
                    save_path = os.path.join(args.model_dir, f'model_dataset_{args.dataset}_noise_{args.noise_ratio}_dim{args.dim}_hops{args.context_hops}_beta{args.beta}_lr{args.lr}_sweep_{sweep_id}.ckpt')
                else:
                    save_path = os.path.join(args.model_dir, f'model_dataset_{args.dataset}_noise_{args.noise_ratio}_dim{args.dim}_hops{args.context_hops}_beta{args.beta}_lr{args.lr}{ump_suffix}.ckpt')
                torch.save(model.state_dict(), save_path)
        else:
            logger.info(f"Epoch {epoch}: Training Loss: {loss.item():.4f}, Time: {train_e_t - train_s_t:.4f}s")


    if len(epoch_times) > 0:
        epoch_times_arr = np.array(epoch_times)
        logger.info("############################## Epoch Time Statistics ##############################")
        logger.info(f"Average epoch time: {np.mean(epoch_times_arr):.2f}s (over {len(epoch_times_arr)} epochs)")
        logger.info(f"Shortest epoch time: {np.min(epoch_times_arr):.2f}s (epoch {np.argmin(epoch_times_arr)})")
        logger.info(f"Longest epoch time: {np.max(epoch_times_arr):.2f}s (epoch {np.argmax(epoch_times_arr)})")
    
    # Output best performance summary
    if best_metrics:
        logger.info(f'Best performance achieved at epoch {best_metrics["epoch"]}:')
        logger.info(f'  valid set ndcg@10: {cur_best_pre_0:.4f}')
        

        metrics_names = ['ndcg', 'recall', 'precision', 'hit_ratio']
        k_values_final = eval(args.Ks) if isinstance(args.Ks, str) else args.Ks

        for metric_name in metrics_names:
            if metric_name in best_metrics:
                metric_values = best_metrics[metric_name]
                metric_str = ', '.join([f'{metric_name}@{k}: {metric_values[i]:.4f}' 
                                    for i, k in enumerate(k_values_final) if i < len(metric_values)])
                logger.info(f'  {metric_str}')
            
            
    else:
        logger.info(f'Final valid set ndcg@10: {cur_best_pre_0:.4f}')
        
    wandb.finish()

def run_sweep(sweep_count=10):
    """
    create and run a new sweep for hyperparameter optimization
    
    args:
        sweep_count: hyperparameter combinations to run 
    """
    wandb.login()
    
    # create sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project="UnGGCN"
    )
    
    print(f"=" * 50)
    print(f"Sweep created successfully!")
    print(f"Sweep ID: {sweep_id}")
    print(f"View results: https://wandb.ai/YOUR_USERNAME/UnGGCN/sweeps/{sweep_id}")
    print(f"=" * 50)
    

    wandb.agent(
        sweep_id, 
        function=lambda: train(use_sweep=True),  # 传入 use_sweep=True
        count=sweep_count 
    )


def join_sweep(sweep_id):
    """
    add this machine to an existing sweep for parallel hyperparameter optimization
    
    args:
        sweep_id: the ID of the existing sweep to join (can be found in the URL of the sweep dashboard)
    """
    wandb.login()
    
    print(f"adding Sweep: {sweep_id}")
    
    wandb.agent(
        sweep_id,
        function=lambda: train(use_sweep=True),
        project="UnGGCN"
    )

if __name__ == '__main__':
    """
    Command Line Usage:
    
    1. Start hyperparameter sweep (runs 50 experiments by default):

       python main.py --sweep --count 10
    
    2. Join an existing sweep (parallel search across multiple machines):
       python main.py --agent abc123xyz
       
       where abc123xyz is the Sweep ID printed when the first machine created the sweep
    
    3. Normal single run (same as before):
       python main.py
    """
    
    
    from utils.parser import parse_args
    
    args = parse_args()
    
    if args.sweep:
        print("Starting hyperparameter search mode...")
        run_sweep(sweep_count=args.count)
    elif args.agent:
        print(f"Joining existing Sweep: {args.agent}")
        join_sweep(args.agent)
    else:
        print("Starting normal training mode...")
        train(use_sweep=False)
