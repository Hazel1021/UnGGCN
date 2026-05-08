import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from pathlib import Path
import csv
import math
from torch.nn.utils import spectral_norm
from time import time


class UnGGSL(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, adj_mat_var,logger=None):
        super(UnGGSL, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.adj_mat_var = adj_mat_var
        self.emb_size = args_config.dim
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        self.logsigma=args_config.logsigma
       
        self._init_weight()
        self.user_mu = nn.Parameter(self.user_mu)
        self.item_mu = nn.Parameter(self.item_mu)
        self.user_logsigma = nn.Parameter(self.user_logsigma)
        self.item_logsigma = nn.Parameter(self.item_logsigma)


        self.n_negs = args_config.n_negs  
        self.K = args_config.K
        self.context_hops = args_config.context_hops
        self.lw = args_config.lw
        self.logger = logger
        self.beta = args_config.beta
        self.disable_ump = getattr(args_config, "disable_ump", False)


        # Normal-Gamma  ( ν=0, λ=1, α=2, β=2)
        self.nu_prior = 0.0
        self.lambda_prior = 1.0
        self.alpha_prior = 2.0
        self.beta_prior = 2.0
        

        self.gcn_encoder = UncertaintyGCNEncoder(
            n_layers=self.context_hops,    
            beta=self.beta,
            disable_ump=self.disable_ump,
        )

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
    def _init_weight(self):
        # init
        self.user_mu=nn.init.xavier_normal_(torch.empty(self.n_users, self.emb_size))
        self.item_mu=nn.init.xavier_normal_(torch.empty(self.n_items, self.emb_size))
        
        # init
        self.user_logsigma=nn.init.constant_(torch.empty(self.n_users, self.emb_size), self.logsigma)
        self.item_logsigma=nn.init.constant_(torch.empty(self.n_items, self.emb_size), self.logsigma)


        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.sparse_norm_adj_var = self._convert_sp_mat_to_sp_tensor(self.adj_mat_var).to(self.device)

    def encode(self, user_mu, user_logsigma, item_mu, item_logsigma):
        all_mu = torch.cat([user_mu, item_mu], 0)
        all_logsigma = torch.cat([user_logsigma, item_logsigma], 0) 
        all_var = torch.exp(2 * all_logsigma)

        
        out_mean, out_var = self.gcn_encoder(all_mu, all_var, self.sparse_norm_adj,self.sparse_norm_adj_var)


        return out_mean, out_var

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']


        all_mean, all_var = self.encode(self.user_mu, self.user_logsigma,
                                         self.item_mu, self.item_logsigma)
        user_mu = all_mean[:self.n_users]
        user_var = all_var[:self.n_users]
        item_mu = all_mean[self.n_users:]
        item_var = all_var[self.n_users:]

        # batch'mean and sigma
        user_mu = user_mu[user]
        pos_mu = item_mu[pos_item]
        user_var = user_var[user]
        pos_var = item_var[pos_item]


        neg_idx = neg_item[:, 0]
        neg_mu = item_mu[neg_idx]
        neg_var = item_var[neg_idx]

           

        return self.create_loss(user_mu, pos_mu, neg_mu, 
                                user_var, pos_var, neg_var,user, pos_item, neg_idx,self.logger)
    

    def create_loss(self, user_mu, pos_mu, neg_mu, 
                    user_var, pos_var, neg_var,user,pos_item, neg_idx,logger=None):
        """
        
        Args:
            user_mu: [batch_size, dim]
            pos_mu: [batch_size, dim] 
            neg_mu: [batch_size, dim] 
            user_var: [batch_size, dim] 
            pos_var: [batch_size, dim] 
            neg_var: [batch_size, dim] 
            neg_idx: negative sample id
        
        Returns:
            total_loss, ranking_loss, prior_loss
        """
        loss_s_t= time()
        batch_size = user_mu.size(0)
        

        mu_diff = neg_mu - pos_mu  # μ_j - μ_i: [batch_size, dim]
        E_Z = (user_mu * mu_diff).sum(dim=-1)  # [batch_size]
        
        
        # (σ_ik + σ_jk): [batch_size, dim]
        var_sum = pos_var + neg_var
        
        # (μ_uk^2 + σ_uk): [batch_size, dim]
        user_term = user_mu.pow(2) + user_var
        
        # term1 = (σ_ik + σ_jk)(μ_uk^2 + σ_uk): [batch_size, dim]
        term1 = var_sum * user_term
        
        # term2 = σ_uk(μ_jk - μ_ik)^2: [batch_size,  dim]
        term2 = user_var * mu_diff.pow(2)
        
        # Var[Z_uij] = Σ_k [...]: [batch_size]
        Var_Z = (term1 + term2).sum(dim=-1)
        Var_Z = Var_Z + 1e-8  
        

        std_Z = torch.sqrt(Var_Z)
        normalized_score =  - E_Z/ std_Z
        
        # Φ(x) = 0.5 * (1 + erf(x / sqrt(2))) 
        prob_i_greater_j = 0.5 * (1.0 + torch.erf(normalized_score / math.sqrt(2.0)))
        
        # avoid log(0) 或 log(1)
        prob_i_greater_j = torch.clamp(prob_i_greater_j, min=1e-10, max=1.0 - 1e-10)
        
        # Ranking loss = -Σ log p(i >_u j)
        ranking_loss = -torch.log(prob_i_greater_j).mean()
        


        # fetch id
        unique_u_ids=torch.unique(user)
        unique_i_ids=torch.unique(torch.cat([pos_item,neg_idx.flatten()]))
        batch_user_mu=self.user_mu[unique_u_ids]
        batch_user_var=torch.exp(2 * self.user_logsigma[unique_u_ids])
        batch_item_mu=self.item_mu[unique_i_ids]
        batch_item_var=torch.exp(2 * self.item_logsigma[unique_i_ids])
        
        batch_mu = torch.cat([batch_user_mu, batch_item_mu], 0)
        # print("batch_mu shape:", batch_mu.shape)
        batch_var = torch.cat([batch_user_var, batch_item_var], 0)
        batch_var = torch.clamp(batch_var, min=1e-6)

      
        
        # (1/2 - α) log(Σ) = -3/2 * log(Σ)
        prior_term1 = (0.5 - self.alpha_prior) * torch.log(batch_var)
        
        # -(2β + λμ^2) / (2Σ) = -(4 + μ^2) / (2Σ)
        prior_term2 = -(2.0 * self.beta_prior + self.lambda_prior * batch_mu.pow(2)) / (2.0 * batch_var)
        

        prior_loss = -(prior_term1 + prior_term2).mean()  
        
        total_loss = ranking_loss +  self.lw *prior_loss
        loss_e_t= time()
        loss_time=loss_e_t-loss_s_t

        return total_loss, ranking_loss, prior_loss


    def generate(self, split=True):

        all_mu, all_var = self.encode(self.user_mu, self.user_logsigma,
                                       self.item_mu, self.item_logsigma)
        if split:
            user_mean = all_mu[:self.n_users]
            item_mean = all_mu[self.n_users:]
            user_var = all_var[:self.n_users]
            item_var = all_var[self.n_users:]

            return user_mean, item_mean, user_var, item_var
        else:

            return all_mu , all_var


    def rating(self, user_mu, user_var, item_mu, item_var):

        """
        
        Args:
            user_mu: [n_users, dim]
            user_var: [n_users, dim]
            item_mu: [n_items, dim]
            item_var: [n_items, dim]
        
        Returns:
            score: [n_users, n_items]
        """
        E_Y = torch.matmul(user_mu, item_mu.t())
        
        # Var[Y_ui] = Σ_k [σ_uk * σ_ik + σ_uk * μ_ik^2 + μ_uk^2 * σ_ik]
        Var_Y = (
            torch.matmul(user_var, item_var.t())
            + torch.matmul(user_var, item_mu.pow(2).t())
            + torch.matmul(user_mu.pow(2), item_var.t())
        ) + 1e-8
        
         # Probit approximation: ŝ ≈ μ_s / sqrt(1 + (π/8) * σ_s²)
        score = E_Y / torch.sqrt(1.0 + (math.pi / 8.0) * Var_Y)
               

        return score  # [n_users, n_items]
    
   


class UncertaintyGraphConvLayer(nn.Module):
    
    """
    uncertainty graph convolution layer
    """
    
    def __init__(self, beta=0.5, disable_ump=False):
        """
        Args:
            beta: controls how much variance affects the attention weights
        """
        super(UncertaintyGraphConvLayer, self).__init__()
        self.beta=beta
        self.disable_ump = disable_ump

        print(f"UncertaintyGraphConvLayer initialized with beta={beta}, disable_ump={disable_ump}")


    def forward(self, mu, var, adj_norm, adj_norm_var):
        """
        Args:
            mu:  [N, dim]
            var:  [N, dim]
            adj_norm:D^{-1/2} A D^{-1/2}
            adj_norm_var:  D^{-1} A D^{-1}

        Returns:
            new_mu: [N, dim]
            new_sigma: [N, dim]
        """
        
        # weight based on variance (softplus)

        attention =  F.softplus(- var,beta = self.beta)
        # print("sigma:",sigma)
        
        

        if self.disable_ump:
            mu_weighted = mu
            sigma_weighted = var
        else:
            mu_weighted = mu * attention
            sigma_weighted = var * (attention ** 2)

        new_mu = torch.sparse.mm(adj_norm, mu_weighted)
        new_var= torch.sparse.mm(adj_norm_var, sigma_weighted)
            
        return new_mu, new_var
     

    


class UncertaintyGCNEncoder(nn.Module):
    """
    multi-layer GCN encoder that propagates both mean and variance
    """
    
    def __init__(self, n_layers=3, beta=0.5, disable_ump=False):
        """
        Args:
            n_layers: number of GCN layers (hops)
            beta: weighting factor for variance in attention
        """
        super(UncertaintyGCNEncoder, self).__init__()
        self.n_layers = n_layers
        self.beta = beta
        self.disable_ump = disable_ump


        self.convs = nn.ModuleList([
            UncertaintyGraphConvLayer(
                beta=self.beta,
                disable_ump=self.disable_ump,
            )
            for n_layer in range(n_layers)
        ])
        

    def forward(self, init_mu, init_var, adj_norm, adj_norm_var):
        """
        Args:
            init_mu: initial mean embeddings [N, dim]
            init_var: initial variances [N, dim]
            adj_norm: normalized adjacency matrix
            adj_norm_var: normalized adjacency matrix for variance propagation

        Returns:
            final_mu: [N, dim]
            final_var: [N, dim]
        """
        mu_list = [init_mu]
        var_list = [init_var]
        
        mu = init_mu
        var = init_var

        
        for layer_idx, conv in enumerate(self.convs, start=1):
            mu, var = conv(mu, var, adj_norm, adj_norm_var)
            mu_list.append(mu)
            var_list.append(var)

        
        n_layers = len(mu_list)
        mu_stack = torch.stack(mu_list, dim=0)   # [L+1, N, dim]
        var_stack = torch.stack(var_list, dim=0)  # [L+1, N, dim]
        
        final_mu = mu_stack.mean(dim=0)                    # [N, dim]
        
        final_var = var_stack.sum(dim=0) / (n_layers ** 2)  # [N, dim] 

        
        return final_mu, final_var

    

    
