import time
import argparse
import pickle
import os
import datetime
import torch
import torch.optim as optim
from networkx import reconstruct_path
from torch.optim import lr_scheduler
from utils import *
from modules import *

parser = argparse.ArgumentParser()
#=====================
#graph configurations
#=====================
parser.add_argument('--data_type', type=str, default= 'real_world',
                    choices=['synthetic', 'discrete', 'real'],
                    help='choosing which experiment to do.')
parser.add_argument('--data_sample_size', type=int, default=5000,
                    help='the number of samples of data')
parser.add_argument('--data_variable_size', type=int, default=11,
                    help='the number of variables in synthetic generated data')
parser.add_argument('--batch-size', type=int, default = 100,# note: should be divisible by sample size, otherwise throw an error
                    help='Number of samples per batch.')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--x_dims', type=int, default=1, #changed here
                    help='The number of input dimensions: default 1.')
parser.add_argument('--z_dims', type=int, default=1,
                    help='The number of latent variable dimensions: default the same as variable size.')
parser.add_argument('--encoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')

#=================
#training config
#=================
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=3e-3,  # basline rate = 1e-3
                    help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--optimizer', type = str, default = 'Adam',
                    help = 'the choice of optimizer used')
parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                    help = 'threshold for learned adjacency matrix binarization')
parser.add_argument('--epochs', type=int, default= 150,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default = 100, # note: should be divisible by sample size, otherwise throw an error
                    help='Number of samples per batch.')
parser.add_argument('--gamma', type=float, default= 1.0,
                    help='LR decay factor.')
parser.add_argument('--tau_A', type = float, default=0.01,
                    help='coefficient for L-1 norm of A.')
parser.add_argument('--lambda_A',  type = float, default= 0.,
                    help='coefficient for DAG constraint h(A).')
parser.add_argument('--c_A',  type = float, default= 0.1,
                    help='coefficient for absolute value h(A).')
parser.add_argument('--h_tol', type=float, default = 1e-8,
                    help='the tolerance of error of h(A) to zero')
parser.add_argument('--suffix', type=str, default='_springs5',
                    help='Suffix for training data (e.g. "_charged".')

#==================
#loop control
#==================
parser.add_argument('--k_max_iter', type = int, default = 1e2,
                    help ='the max iteration number for searching lambda and c')

#==================
#save
#==================
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)
torch.manual_seed(args.seed)


# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    save_folder = save_folder.replace(':', '_')
    os.makedirs(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")


#compute constraint h(A) value

def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

prox_plus = torch.nn.Threshold(0.,0.)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1

#=========
#get data
#=========
train_loader, valid_loader, test_loader, ground_truth_G = load_data( args, args.batch_size, args.suffix)

#==========
#adj 可給ground truth adj
#==========
num_nodes = args.data_variable_size
# adj_A = np.zeros((num_nodes, num_nodes))
adj_A = np.random.randn(num_nodes, num_nodes) * 0.01

#========================
#encoder decoder setting
#========================
encoder = GCNEncoder(args.data_variable_size, args.x_dims, args.encoder_hidden,
                         int(args.z_dims), adj_A,
                         batch_size = args.batch_size,
                         do_prob = args.encoder_dropout, factor = args.factor).double()

decoder = GCNDecoder(args.data_variable_size * args.x_dims,
                     args.z_dims, args.x_dims, encoder,
                     data_variable_size=args.data_variable_size,
                     batch_size=args.batch_size,
                     n_hid=args.decoder_hidden,
                     do_prob=args.decoder_dropout).double()
#keep X and fc same dtype float32
encoder.float()
decoder.float()
print("Checking encoder parameters:")
for name, param in encoder.named_parameters():
    print(f"Found parameter: {name}")

print("\nChecking if adj_A is in the list:")
if "adj_A" in [name for name, _ in encoder.named_parameters()]:
    print("adj_A is correctly registered!")
else:
    print("adj_A is NOT registered as a parameter! This is the core problem.")
encoder.init_weights()
decoder.init_weights()

#=================
#setting optimizer
#=================
if args.optimizer == 'Adam':
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=args.lr)
elif args.optimizer == 'LBFGS':
    optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=args.lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)



def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr
#=========
#training
#=========
# 設定梯度裁剪的最大範數值
# max_grad_norm = 1.0

def train(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []

    encoder.train()
    decoder.train()
    scheduler.step()

    optimizer, lr = update_optimizer(optimizer, args.lr, c_A)
    for batch_index, (data, relations) in enumerate(train_loader):
        relations = relations.unsqueeze(2)
        optimizer.zero_grad()
        #data.shape = (853,11)
        data = data.unsqueeze(-1)#(100,11,1)
        enc_x, logits, origin_A, z_gap, z_positive, myA, Wa, adj_norm = encoder(data)  # logits is of size: [num_sims, z_dims]
        edges = logits
        dec_x, output= decoder(data, edges, args.data_variable_size * args.x_dims, adj_norm, Wa)
        if torch.sum(output != output):
            print('nan error in train\n')

        target = data
        preds = output
        variance = 0.

        #reconstruction accuracy loss
        loss_nll = nll_gaussian(preds, target, variance)

        #kl loss
        loss_kl = kl_gaussian_sem(logits)

        #ELBO loss
        loss = loss_kl + loss_nll

        # add A loss
        one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = args.tau_A * torch.sum(torch.abs(one_adj_A))

        #compute h(A)
        h_A = _h_A(origin_A, args.data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 0.001 * torch.trace(
            origin_A * origin_A) + sparse_loss  # +  0.01 * torch.sum(variance * variance)
        loss.backward()

        # 這裡會檢查所有參數的梯度
        all_params = list(encoder.named_parameters()) + list(decoder.named_parameters())

        # # 遍歷所有參數，檢查它們是否有梯度
        # for name, param in all_params:
        #     if param.grad is not None:
        #         # 只印出有梯度的參數，如果梯度為零可以特別標記
        #         grad_norm = param.grad.norm().item()
        #         if grad_norm > 1e-8:  # 設定一個閾值來避免印出極小的數值
        #             print(f"Epoch {epoch}, Parameter: {name}, Grad Norm: {grad_norm:.6f}")
        #     else:
        #         # 這是最關鍵的檢查：如果梯度為 None，代表計算圖斷了
        #         print(f"Epoch {epoch}, Parameter: {name}, Grad is None!")

        optimizer.step()
        myA.data = stau(myA.data, args.tau_A * lr)

        if torch.sum(origin_A != origin_A):
            print('nan error of origin A\n')

        # compute metrics
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))

        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        shd_trian.append(shd)

    print(h_A.item())
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'ELBO_loss: {:.10f}'.format(np.mean(kl_train) + np.mean(nll_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
          'time: {:.4f}s'.format(time.time() - t))

    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'ELBO_loss: {:.10f}'.format(np.mean(kl_train) + np.mean(nll_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    if 'graph' not in vars():
        print('error on assign')

    return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A

###########################
# main
###########################
t_total = time.time()
best_ELBO_loss = np.inf
best_NLL_loss = np.inf
best_MSE_loss = np.inf
best_epoch = 0
best_ELBO_graph = []
best_NLL_graph = []
best_MSE_graph = []
# optimizer step on hyparameters
c_A = args.c_A
lambda_A = args.lambda_A
h_A_new = torch.tensor(1.)
h_tol = args.h_tol
k_max_iter = int(args.k_max_iter)
h_A_old = np.inf

try:
    for step_k in range(k_max_iter):
        while c_A < 1e+20:
            for epoch in range(args.epochs):
                ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer)
                if ELBO_loss < best_ELBO_loss:
                    best_ELBO_loss = ELBO_loss
                    best_epoch = epoch
                    best_ELBO_graph = graph

                if NLL_loss < best_NLL_loss:
                    best_NLL_loss = NLL_loss
                    best_epoch = epoch
                    best_NLL_graph = graph

                if MSE_loss < best_MSE_loss:
                    best_MSE_loss = MSE_loss
                    best_epoch = epoch
                    best_MSE_graph = graph

            print("Optimization Finished!")
            print("Best Epoch: {:04d}".format(best_epoch))
            if ELBO_loss > 2 * best_ELBO_loss:
                break

            # update parameters
            A_new = origin_A.data.clone()
            h_A_new = _h_A(A_new, args.data_variable_size)
            if h_A_new.item() > 0.25 * h_A_old:
                c_A*=10
            else:
                break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()

        if h_A_new.item() <= h_tol:
            break


    if args.save_folder:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

    # test()
    print (best_ELBO_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
    print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print(best_NLL_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
    print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


    print (best_MSE_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
    print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph = origin_A.data.clone().numpy()
    graph[np.abs(graph) < 0.1] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.2] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.3] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


except KeyboardInterrupt:
    # print the best anway
    print(best_ELBO_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
    print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print(best_NLL_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
    print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print(best_MSE_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
    print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph = origin_A.data.clone().numpy()
    graph[np.abs(graph) < 0.1] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.2] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.3] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


f = open('trueG', 'w')
matG = np.matrix(nx.to_numpy_array(ground_truth_G))
for line in matG:
    np.savetxt(f, line, fmt='%.5f')
f.closed

f1 = open('predG', 'w')
matG1 = np.matrix(origin_A.data.clone().numpy())
for line in matG1:
    np.savetxt(f1, line, fmt='%.5f')
f1.closed


if log is not None:
    print(save_folder)
    log.close()
