import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.linalg as slin
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import os
import glob
import re
import pickle
import math
from torch.optim.adam import Adam
from pathlib import Path
import yaml
import argparse

#from train import train

# parser = argparse.ArgumentParser()
# #=====================
# #graph configurations
# #=====================
# parser.add_argument('--data_type', type=str, default= 'real_world',
#                     choices=['synthetic', 'discrete', 'real'],
#                     help='choosing which experiment to do.')
# parser.add_argument('--data_sample_size', type=int, default=5000,
#                     help='the number of samples of data')
# parser.add_argument('--data_variable_size', type=int, default=10,
#                     help='the number of variables in synthetic generated data')
# parser.add_argument('--batch_size', type=int, default = 100,# note: should be divisible by sample size, otherwise throw an error
#                     help='Number of samples per batch.')
# parser.add_argument('--no-cuda', action='store_true', default=True,
#                     help='Disables CUDA training.')
# parser.add_argument('--no-factor', action='store_true', default=False,
#                     help='Disables factor graph model.')
# parser.add_argument('--x_dims', type=int, default=1, #changed here
#                     help='The number of input dimensions: default 1.')
# parser.add_argument('--z_dims', type=int, default=1,
#                     help='The number of latent variable dimensions: default the same as variable size.')
# parser.add_argument('--encoder-hidden', type=int, default=64,
#                     help='Number of hidden units.')
# parser.add_argument('--decoder-hidden', type=int, default=64,
#                     help='Number of hidden units.')
# parser.add_argument('--encoder-dropout', type=float, default=0.0,
#                     help='Dropout rate (1 - keep probability).')
# parser.add_argument('--decoder-dropout', type=float, default=0.0,
#                     help='Dropout rate (1 - keep probability).')
#
# #=================
# #training config
# #=================
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--lr', type=float, default=3e-3,  # basline rate = 1e-3
#                     help='Initial learning rate.')
# parser.add_argument('--lr-decay', type=int, default=200,
#                     help='After how epochs to decay LR by a factor of gamma.')
# parser.add_argument('--optimizer', type = str, default = 'Adam',
#                     help = 'the choice of optimizer used')
# parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
#                     help = 'threshold for learned adjacency matrix binarization')
# parser.add_argument('--epochs', type=int, default= 300,
#                     help='Number of epochs to train.')
# parser.add_argument('--batch-size', type=int, default = 100, # note: should be divisible by sample size, otherwise throw an error
#                     help='Number of samples per batch.')
#
#
#
#
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# args.factor = not args.no_factor
# print(args)

def matrix_poly(matrix, d):
    x = torch.eye(d).double()+ torch.div(matrix, d)
    return torch.matrix_power(x, d)

def nll_gaussian(preds, target, variance, add_const=False):
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))

def kl_gaussian(preds, zsize):
    predsnew = preds.squeeze(1)
    mu = predsnew[:,0:zsize]
    log_sigma = predsnew[:,zsize:2*zsize]
    kl_div = torch.exp(2*log_sigma) - 2*log_sigma + mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)) - zsize)*0.5


def count_accuracy(G_true: nx.DiGraph,
                   G: nx.DiGraph,
                   G_und: nx.DiGraph = None) -> tuple:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        G_true: ground truth graph
        G: predicted graph
        G_und: predicted undirected edges in CPDAG, asymmetric

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    B_true = nx.to_numpy_array(G_true) != 0
    B = nx.to_numpy_array(G) != 0
    B_und = None if G_und is None else nx.to_numpy_array(G_und)
    d = B.shape[0]
    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, fpr, shd, pred_size

def kl_gaussian_sem(preds):
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)))*0.5

def preprocess_adj_new(adj):
    adj_normalized = (torch.eye(adj.shape[0]).double() - (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_new1(adj):
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double()-adj.transpose(0,1))
    return adj_normalized

#path
_DIR = os.path.dirname(os.path.abspath(__file__))
_REAL_WORLD_DIR = os.path.join(_DIR, "data/")
from sklearn.preprocessing import StandardScaler

def load_data(args, batch_size = 1000, data_type='real_world', debug=False):
    if args.data_type == 'real_world':
        df = pd.read_csv(os.path.join(_REAL_WORLD_DIR, "sachs_cd3cd28.csv"))
        label_mapping = {
            0: "Raf",
            1: "Mek",
            2: "Plcg",
            3: "PIP2",
            4: "PIP3",
            5: "Erk",
            6: "Akt",
            7: "PKA",
            8: "PKC",
            9: "P38",
            10: "Jnk",
        }
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        df.rename(columns=inverse_mapping, inplace=True)

        # ground truth graph
        graph = {
            7: [10, 9, 6, 5, 1, 0, 4],
            8: [10, 9, 1, 0],
            4: [6, 3, 2, 7],
            1: [5],
            0: [1],
            3: [8],
            2: [8, 3],
        }
        G = nx.DiGraph(graph)
        G.add_nodes_from(range(11))

    elif args.data_type == 'synthetic':
        df = pd.read_csv(os.path.join(_REAL_WORLD_DIR, "synthetic_data.csv"))
        G = nx.read_edgelist("data/ground_truth_graph.txt", create_using= nx.DiGraph, nodetype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    # 轉成 torch tensor
    X = torch.FloatTensor(X_scaled)#float32
    # X = torch.FloatTensor(df.values)
    feat_train = X
    feat_valid = X
    feat_test =  X

    # 建立 Dataset
    train_data = TensorDataset(feat_train, feat_train)
    valid_data = TensorDataset(feat_valid, feat_train)
    test_data = TensorDataset(feat_test, feat_train)

    # 建立 DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data, train_loader, valid_loader, test_loader, G#, scaler




# if __name__ == '__main__':
#     train_loader, valid_loader, test_loader, ground_truth_G = load_data(args, args.batch_size, args.data_type)
#     # print(len(train_loader.shape)) #853
#     for batch_idx, (data, relations) in enumerate(train_loader):
#         if batch_idx == 0:
#             print(f"data={data}")
#             print(f"relations={relations}")
#             relations = relations.unsqueeze(2)
#             print(f"relations={relations}")
