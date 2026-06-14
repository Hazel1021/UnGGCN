import numpy as np
import scipy.sparse as sp
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)


def reset_data_state():
    global n_users, n_items, train_user_set, test_user_set, valid_user_set
    n_users = 0
    n_items = 0
    train_user_set = defaultdict(list)
    test_user_set = defaultdict(list)
    valid_user_set = defaultdict(list)


def read_cf_amazon(file_name):
    return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]


def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def statistics(train_data, valid_data, test_data, noise_ratio, seed=2025):
    global n_users, n_items

    train_data = train_data.copy()
    valid_data = valid_data.copy()
    test_data = test_data.copy()

    n_users = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    if dataset != 'no':
        n_items -= n_users
        # remap [n_users, n_users+n_items] to [0, n_items]
        train_data[:, 1] -= n_users
        valid_data[:, 1] -= n_users
        test_data[:, 1] -= n_users
        
    train_data_dic_before_noise = [[] for _ in range(n_users)]

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
        train_data_dic_before_noise[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))
    clean_train_data = train_data.copy()
    print(f"adding noise with ratio {noise_ratio}")
    
    print(f"before adding noise, the train data has {len(train_data)} interactions")
    
    noise_add_inters = []
    if noise_ratio == 0.0:
        print("no noise added")
    else:
        rng = random.Random(seed)
        for user, interation in enumerate(train_data_dic_before_noise):
            n_noise = int(len(interation) * noise_ratio)
            if n_noise > 0:
                interacted = set(interation)
                noisy_items = set()
                while len(noisy_items) < n_noise and len(noisy_items) + len(interacted) < n_items:
                    item = rng.randrange(n_items)
                    if item not in interacted:
                        noisy_items.add(item)
                for item in noisy_items:
                    train_user_set[user].append(item)
                    noise_add_inters.append([user, item])
                    # train_data = np.vstack((train_data, [user, item]))
        if noise_add_inters:
            train_data = np.vstack((train_data, np.array(noise_add_inters, dtype=np.int32)))
    noisy_edges = np.array(noise_add_inters, dtype=np.int32).reshape(-1, 2)
    print(f"after adding noise, the train data has {len(train_data)} interactions")
    return train_data, clean_train_data, noisy_edges


def build_sparse_graph(data_cf):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()
    
    def _bi_norm_lap_var(adj):
        # D^{-1}AD^{-1}
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        bi_lap = d_mat_inv.dot(adj).dot(d_mat_inv)
        return bi_lap.tocoo()

    cf = data_cf.copy()
    print('n_users=%d, n_items=%d' % (n_users, n_items))
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))
    return _bi_norm_lap(mat),_bi_norm_lap_var(mat)


def load_data(model_args):
    global args, dataset
    reset_data_state()
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'
    noise_ratio = args.noise_ratio

    if dataset == 'no':
        read_cf = read_cf_yelp2018
    else:
        read_cf = read_cf_amazon

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    all_cf=np.concatenate([train_cf,test_cf],axis=0)
    print(f"train + test len: {len(all_cf)}")
    if args.dataset != 'no' :
        valid_cf = read_cf(directory + 'valid.txt')
        all_cf=np.concatenate([all_cf,valid_cf],axis=0)
        print(f"all_cf len: {len(all_cf)}")
    else:
        valid_cf = test_cf
        print("no valid set")  
        
        

    seed = getattr(args, "seed", 2025)
    trian_cf_nosie_injected, clean_train_cf, injected_noise_edges = statistics(
        train_cf, valid_cf, test_cf, noise_ratio, seed=seed
    )
    

    print('building the adj mat ...')
    norm_mat, norm_mat_var = build_sparse_graph(trian_cf_nosie_injected)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'clean_train_cf': clean_train_cf,
        'train_cf_with_noise': trian_cf_nosie_injected,
        'injected_noise_edges': injected_noise_edges,
    }
    user_dict = {
        'train_user_set': train_user_set,
        'valid_user_set': valid_user_set if args.dataset != 'no' else None,
        'test_user_set': test_user_set,
    }

    print('loading over ...')
    return trian_cf_nosie_injected, user_dict, n_params, norm_mat, norm_mat_var
