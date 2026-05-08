from .metrics import *
from .parser import parse_ks

import torch
import numpy as np
import multiprocessing

cores = multiprocessing.cpu_count() // 2

Ks = [10, 20]
test_flag = 'part'
BATCH_SIZE = 2048
batch_test_flag = True
device = torch.device("cpu")


def get_performance(user_pos_test, r, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
        'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}


@torch.no_grad()
def test(model, user_dict, n_params, mode='test', eval_args=None):
    global Ks, test_flag, BATCH_SIZE, batch_test_flag, device
    if eval_args is not None:
        Ks = parse_ks(eval_args.Ks)
        test_flag = eval_args.test_flag
        BATCH_SIZE = eval_args.test_batch_size
        batch_test_flag = eval_args.batch_test_flag
    device = next(model.parameters()).device

    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks))}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    if mode == 'test':
        test_user_set = user_dict['test_user_set']
    else:
        test_user_set = user_dict['valid_user_set']
        if test_user_set is None:
            test_user_set = user_dict['test_user_set']

    
    
    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = (n_test_users + BATCH_SIZE - 1) // BATCH_SIZE

    count = 0

    user_mean, item_mean, user_var, item_var = model.generate()
    max_k = max(Ks)

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * BATCH_SIZE
        end = min((u_batch_id + 1) * BATCH_SIZE, n_test_users)

        user_list_batch = test_users[start:end]
        if len(user_list_batch) == 0:
            continue

        user_batch = torch.LongTensor(user_list_batch).to(device)
        u_mean = user_mean[user_batch]
        u_var = user_var[user_batch]

        rate_batch = model.rating(u_mean, u_var, item_mean, item_var)

        for row, user in enumerate(user_list_batch):
            training_items = train_user_set.get(user, [])
            if len(training_items) > 0:
                rate_batch[row, list(training_items)] = -float('inf')

        _, topk_indices = torch.topk(rate_batch, max_k, dim=1)
        topk_indices = topk_indices.cpu().numpy()

        batch_result = []
        for row, user in enumerate(user_list_batch):
            user_pos_test = test_user_set[user]
            r = np.isin(topk_indices[row], list(user_pos_test)).astype(np.int32).tolist()
            batch_result.append(get_performance(user_pos_test, r, Ks))

        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users

    assert count == n_test_users

    
    return result
