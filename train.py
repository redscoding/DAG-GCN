import time
import argparse
import pickle
import os
import datetime
from xmlrpc.server import ServerHTMLDoc

import numpy as np
import torch
import torch.optim as optim
from networkx import reconstruct_path
from torch.optim import lr_scheduler
from utils import *
from modules import *
from results import save_diagnostic_plot

parser = argparse.ArgumentParser()
#=====================
#graph configurations
#=====================
parser.add_argument('--data_type', type=str, default= 'real_world',
                    choices=['synthetic', 'discrete', 'real','real_world'],
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
parser.add_argument('--lr-decay', type=int, default=200,#200
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--optimizer', type = str, default = 'Adam',
                    help = 'the choice of optimizer used')
parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                    help = 'threshold for learned adjacency matrix binarization')
parser.add_argument('--epochs', type=int, default= 300,#300
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default = 100, # note: should be divisible by sample size, otherwise throw an error
                    help='Number of samples per batch.')
parser.add_argument('--gamma', type=float, default= 1.0,
                    help='LR decay factor.')
parser.add_argument('--tau_A', type = float, default=0.0, #0.01學不到資訊
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
_, train_loader, valid_loader, test_loader, ground_truth_G = load_data( args, args.batch_size, args.suffix)

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
#DAG-GNN
 #=================
# encoder = Encoder(args.data_variable_size * args.x_dims, args.x_dims, args.encoder_hidden,
#                          int(args.z_dims), adj_A,
#                          batch_size = args.batch_size,
#                          do_prob = args.encoder_dropout, factor = args.factor).double()
# decoder = Decoder(args.data_variable_size * args.x_dims,
#                          args.z_dims, args.x_dims, encoder,
#                          data_variable_size = args.data_variable_size,
#                          batch_size = args.batch_size,
#                          n_hid=args.decoder_hidden,
#                          do_prob=args.decoder_dropout).double()
# encoder.float()
# decoder.float()
# encoder.init_weights()
# decoder.init_weights()
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

def train(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer, train_loader):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []
    tpr_train = []

    #紀錄矩陣消失問題
    epoch_loss_recon = 0.0
    epoch_loss_dag_h_A = 0.0
    epoch_loss_sparse_total = 0.0
    epoch_A_magnitude = 0.0
    epoch_A_grad_magnitude = 0.0

    num_batches = len(train_loader)


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
        # enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data)  # logits is of size: [num_sims, z_dims]
        # edges = logits
        #
        # dec_x, output, adj_A_tilt_decoder = decoder(data, edges, args.data_variable_size * args.x_dims, origin_A, adj_A_tilt_encoder, Wa)
        if torch.sum(output != output):
            print('nan error in train\n')

        target = data
        preds = output
        variance = 0.

        #reconstruction accuracy loss
        loss_nll = nll_gaussian(preds, target, variance)

        #1029修改稀疏損失
        sparse_loss_l1 = args.tau_A * torch.sum(torch.abs(origin_A))
        sparse_loss_l2_trace = 0.001 * torch.trace(origin_A * origin_A)
        loss_sparse_total = sparse_loss_l1 + sparse_loss_l2_trace

        #單純紀錄DAG loss h(A)
        h_A = _h_A(origin_A, args.data_variable_size)
        loss_dag_augmented = lambda_A * h_A + 0.5 * c_A * h_A * h_A

        #kl loss
        loss_kl = kl_gaussian_sem(logits)
        # loss_kl  =  kl_gaussian(logits, args.z_dims)

        #ELBO loss
        loss = loss_kl + loss_nll

        # add A loss
        one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = args.tau_A * torch.sum(torch.abs(one_adj_A))

        loss = loss_kl + loss_nll + loss_dag_augmented + loss_sparse_total

        loss.backward()

        #1029修改 監控日誌 鄰接矩陣與梯度issues
        #累加A的大小
        epoch_A_magnitude += torch.mean(torch.abs(origin_A.data)).item()
        #累加A的梯度大小
        if origin_A.grad is not None:
                epoch_A_grad_magnitude += torch.mean(torch.abs(origin_A.grad.data)).item()

        #累加L_Recon
        epoch_loss_recon += loss_nll.item()

        #累加 h_A (DAG損失原始值)
        epoch_loss_dag_h_A += h_A.item()

        #累加L_sparse (L1+ L2-trace)
        epoch_loss_sparse_total += loss_sparse_total.item()

        #日誌結束

        # 這裡會檢查所有參數的梯度
        all_params = list(encoder.named_parameters()) + list(decoder.named_parameters())



        optimizer.step()
        myA.data = stau(myA.data, args.tau_A * lr)

        if torch.sum(origin_A != origin_A):
            print('nan error of origin A\n')

        # compute metrics
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0

        print("Epoch", epoch, "edges after threshold:", np.count_nonzero(graph))

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))

        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        shd_trian.append(shd)
        tpr_train.append(tpr)

        # if shd < best_shd:
        #     best_shd = shd
        #     best_SHD_graph = graph.copy()
        #     print("update shd at epoch", epoch)
        #
        #
        # if (tpr > best_tpr) and (shd <= 36):
        #     best_tpr = tpr
        #     best_tpr_graph = graph.copy()
        #     print("update tpr at epoch", epoch)

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
          'tpr_train: {:.10f}'.format(np.mean(tpr_train)),
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
              'tpr_train: {:.10f}'.format(np.mean(tpr_train)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    if 'graph' not in vars():
        print('error on assign')



    #計算監控指標
    avg_A_mag = epoch_A_magnitude / num_batches
    avg_A_grad_mag = epoch_A_grad_magnitude / num_batches
    avg_loss_recon = epoch_loss_recon / num_batches
    avg_loss_dag_h_A = epoch_loss_dag_h_A / num_batches
    avg_loss_sparse = epoch_loss_sparse_total / num_batches

    return avg_A_mag, avg_A_grad_mag, avg_loss_recon, avg_loss_dag_h_A, avg_loss_sparse, np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A, shd, tpr#, best_SHD_graph, best_tpr_graph

###########################
# main
###########################

train_data,_,_,_,_ = load_data( args, args.batch_size, args.suffix)

t_total = time.time()

#日誌紀錄
log_epoch = []
log_A_magnitude = []
log_A_grad_magnitude = []
log_loss_recon = []
log_loss_dag_h_A = []
log_loss_sparse = []
# optimizer step on hyparameters
c_A = args.c_A
lambda_A = args.lambda_A
h_A_new = torch.tensor(1.)
h_tol = args.h_tol
k_max_iter = int(args.k_max_iter)
h_A_old = np.inf

#bootstrap settings
k_boot_iter = 3
all_result_graph = []

try:
    for b in range(k_boot_iter):
        print(f"--- [Bootstrap Iteration {b+1} / {k_boot_iter}] ---")

        N_total = len(train_data)
        # print(f"N = {N_total}") 853
        #重抽樣 從n個索引抽出一個
        boot_indices = torch.randint(0,N_total,(N_total,))

        #根據索引重建新的資料集
        boot_data = train_data[boot_indices]
        # print(f"boot_data = {boot_data}")

        #建立新的dataset & dataloader
        boot_dataset = torch.utils.data.TensorDataset(*boot_data)
        train_loader = torch.utils.data.DataLoader(
            boot_dataset,
            batch_size = args.batch_size,
            shuffle = True
        )
        encoder = GCNEncoder(args.data_variable_size, args.x_dims, args.encoder_hidden,
                             int(args.z_dims), adj_A,
                             batch_size=args.batch_size,
                             do_prob=args.encoder_dropout, factor=args.factor).double()

        decoder = GCNDecoder(args.data_variable_size * args.x_dims,
                             args.z_dims, args.x_dims, encoder,
                             data_variable_size=args.data_variable_size,
                             batch_size=args.batch_size,
                             n_hid=args.decoder_hidden,
                             do_prob=args.decoder_dropout).double()
        # keep X and fc same dtype float32
        encoder.float()
        decoder.float()
        encoder.init_weights()
        decoder.init_weights()
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

        best_ELBO_loss = np.inf
        best_NLL_loss = np.inf
        best_MSE_loss = np.inf
        best_shd = np.inf
        best_tpr = -1
        best_epoch = 0
        best_ELBO_graph = None
        best_NLL_graph = None
        best_MSE_graph = None
        best_SHD_graph = None
        best_tpr_graph = None

        for step_k in range(k_max_iter):
            while c_A < 1e+20:
                for epoch in range(args.epochs):
                    #  return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A, shd, tpr
                    avg_A_mag, avg_A_grad_mag, avg_loss_recon, avg_loss_dag_h_A, avg_loss_sparse, ELBO_loss, NLL_loss, MSE_loss, graph, origin_A, shd_new, tpr_new= train(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer, train_loader = train_loader)

                    log_epoch.append(epoch)
                    log_A_magnitude.append(avg_A_mag)
                    log_A_grad_magnitude.append(avg_A_grad_mag)
                    log_loss_recon.append(avg_loss_recon)
                    log_loss_dag_h_A.append(avg_loss_dag_h_A)
                    log_loss_sparse.append(avg_loss_sparse)

                    print(
                        f"[DEBUG] epoch {epoch}, shd_new={shd_new}, best_shd={best_shd}, tpr_new={tpr_new}, best_tpr={best_tpr}")

                    if ELBO_loss <= best_ELBO_loss :
                        print(f"[DEBUG] update elbo at epoch {epoch}, graph nnz={graph.sum()}")
                        best_ELBO_loss = ELBO_loss
                        best_epoch = epoch
                        if np.count_nonzero(graph)>0:
                            best_ELBO_graph = graph.copy()

                    if NLL_loss <= best_NLL_loss :
                        print('update nll')
                        best_NLL_loss = NLL_loss
                        best_epoch = epoch
                        if np.count_nonzero(graph)>0:
                            best_NLL_graph = graph.copy()
                    if MSE_loss <= best_MSE_loss:
                        print('update mse')
                        best_MSE_loss = MSE_loss
                        best_epoch = epoch
                        if np.count_nonzero(graph)>0:
                            best_MSE_graph = graph.copy()

                    if shd_new <= best_shd:
                        best_shd = shd_new
                        if np.count_nonzero(graph)>0:
                            best_SHD_graph = graph.copy()

                    if (tpr_new >= best_tpr) and (shd_new <= 36):
                        best_tpr = tpr_new
                        best_tpr_graph = graph.copy()


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

        if best_SHD_graph is not None:
                all_result_graph.append(best_SHD_graph.copy())

    valid_graphs = [graph for graph in all_result_graph if graph is not None]

    if not valid_graphs:
        print("警告：所有的 bootstrap 運行都失敗了。")
        num_nodes = 11  # <--- 請確認
        edge_frequency_matrix = np.zeros((num_nodes, num_nodes))
    else:
        # 3. 只對成功的結果取平均
        success_rate = len(valid_graphs) / len(all_result_graph)
        print(f"Bootstrap 成功率: {success_rate * 100:.2f}% ({len(valid_graphs)} / {len(all_result_graph)})")

        # 這裡 valid_graphs 是一個只包含 numpy 陣列的串列，可以安全地取平均
        edge_frequency_matrix = np.mean(valid_graphs, axis=0)

        final_aggregated_graph = edge_frequency_matrix.copy()

        threshold = 0.5
        final_aggregated_graph[final_aggregated_graph >= threshold] = 1
        final_aggregated_graph[final_aggregated_graph < threshold] = 0

        print("\n--- [Bootstrap 聚合圖 (Threshold=0.5) 最終結果] ---")
        print(final_aggregated_graph)
        print("Ground Truth:")
        print(nx.to_numpy_array(ground_truth_G))

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(final_aggregated_graph))
        print('Bootstrap Aggregated Graph Accuracy:')
        print(f'SHD: {shd}, TPR: {tpr}, FDR: {fdr}, FPR: {fpr}, NNZ: {nnz}')
        print("--------------------------------------------------\n")


    if args.save_folder:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

    # test()
    # print (best_ELBO_graph)
    # print(nx.to_numpy_array(ground_truth_G))
    # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
    # print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
    #
    # print(best_NLL_graph)
    # print(nx.to_numpy_array(ground_truth_G))
    # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
    # print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
    #
    #
    # print (best_MSE_graph)
    # print(nx.to_numpy_array(ground_truth_G))
    # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
    # print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
    #
    # print(best_SHD_graph)
    # print(nx.to_numpy_array(ground_truth_G))
    # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_SHD_graph))
    # print('Best SHD Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
    #
    # print(best_tpr_graph)
    # print(nx.to_numpy_array(ground_truth_G))
    # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_tpr_graph))
    # print('Best tpr Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
    #
    # graph = origin_A.data.clone().numpy()
    # graph[np.abs(graph) < 0.1] = 0
    # # print(graph)
    # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    # print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
    #
    # graph[np.abs(graph) < 0.2] = 0
    # # print(graph)
    # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    # print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
    #
    # graph[np.abs(graph) < 0.3] = 0
    # # print(graph)
    # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    # print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


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
