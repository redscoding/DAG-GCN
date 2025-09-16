import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fontTools.feaLib.ast import GlyphClassName
from torch.autograd import Variable
from utils import preprocess_adj_new, preprocess_adj_new1
#mlp->gcn->batch_norm->leaky relu->gcn->m_Z s_Z ->Z->gcn->batch_norm->leaky relu->gcn->batch_norm->leaky relu->mlp->m_X s_X->~X
###########################
# DAG-GNN
###########################
"""
encoder = Encoder(args.data_variable_size * args.x_dims, args.x_dims, args.encoder_hidden,
                         int(args.z_dims), adj_A,
                         batch_size = args.batch_size,
                         do_prob = args.encoder_dropout, factor = args.factor).double()
"""
class Encoder(nn.Module):
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol=0.1):
        super(Encoder, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, inputs):
        # print(inputs.shape)
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)

        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj_new(adj_A1).float()

        adj_A = torch.eye(adj_A1.size()[0]).float()
        # print("Weight dtype:", self.fc1.weight.dtype)

        H1 = F.relu((self.fc1(inputs)))
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa

class Decoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(Decoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,inputs, input_z, n_in_node, origin_A, adj_A_tilt, Wa):

        #adj_A_new1 = (I-A^T)^(-1)
        origin_A = origin_A.float()
        input_z = input_z.float()
        Wa = Wa.float()
        adj_A_new1 = preprocess_adj_new1(origin_A).float()  # 這裡就改成 float
        # adj_A_new1 = torch.linalg.inv(adj_A_new1)

        mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa  # 此時全部都是 float

        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out, adj_A_tilt
###########################
#DAG-GCN
###########################
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        """
        x: 節點特徵矩陣 (n_nodes x in_features)
        adj: 鄰接矩陣 (n_nodes x n_nodes)，這裡是學習到的 A
        """

        # 如果需要正規化，可以加上 D^{-1/2} A D^{-1/2}
        adj_norm = self.normalize_adj(adj)
        # GCN 聚合公式
        out = torch.matmul(adj_norm, x)   # A * X
        out = self.linear(out)            # (AX)W

        return out

    def normalize_adj(self, adj):
        """對 A 做對稱正規化: D^{-1/2} A D^{-1/2}"""
        rowsum = adj.sum(1) + 1e-8  # 避免除以 0
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

class GCNEncoder(nn.Module):
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol=0.1):
        super(GCNEncoder, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.factor = factor
        self.n_xdims = n_xdims
        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_hid, bias = True)
        self.fc3 = nn.Linear(n_hid, n_hid, bias = True)
        self.fc4 = nn.Linear(n_hid, n_hid, bias = True)

        self.gcn = GCNLayer(n_hid, n_hid)
        self.batchnorm = nn.BatchNorm1d(n_in)

        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He-initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a= 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        # print(inputs.shape) #(100,11,1)
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error in GCNencoder \n')

        # to amplify the value of A and accelerate convergence.
        adj_A = torch.sinh(3.*self.adj_A)

        # adj_Aforz = I-A^T
        # adj_Aforz = preprocess_adj_new(adj_A1).float()
        # print(f'adj_Aforz={adj_Aforz.shape}') #11,11

        #adj_A = torch.eye(self.adj_A.size()[0]).float()######
        # print("Weight dtype:", self.fc1.weight.dtype)

        H1 = F.leaky_relu((self.fc1(inputs))) #100,11,1 1*64 -> 11 * 64
        x = (self.fc2(H1)) #11 * 64  64*64 ->11 * 64
        gcn_x = self.gcn(x, adj_A)
        # gcn_x = self.batchnorm(gcn_x)
        # print(f"gcn_x={gcn_x.shape}") #100,11,64

        H2 = F.leaky_relu((self.fc3(gcn_x)))# fc3 fc4 = 64,64
        gcn_x2 =self.gcn(self.fc4(H2), adj_A)

        logits = gcn_x2+self.Wa

        return x, logits, adj_A, self.z, self.z_positive, self.adj_A, self.Wa

class GCNDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(GCNDecoder, self).__init__()
        self.gcn = GCNLayer(n_hid, n_hid)
        self.batchnorm = nn.BatchNorm1d(n_in)
        self.out_fc1 = nn.Linear(n_hid, n_hid, bias = True) #64,64
        self.out_fc2 = nn.Linear(n_hid, n_hid, bias = True)
        self.out_fc3 = nn.Linear(n_hid, n_hid, bias = True)
        self.out_fc4 = nn.Linear(n_hid, n_hid, bias = True)
        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He-initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a = 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self,inputs, input_z, n_in_node, origin_A, Wa):

        #adj_A_new1 = (I-A^T)^(-1)
        origin_A = origin_A.float()
        input_z = input_z.float()
        # print(f"input_z= {input_z.shape}")#100,11,64
        Wa = Wa.float()
        # adj_A_new1 = preprocess_adj_new1(origin_A).float()  # 這裡就改成 float
        # adj_A_new1 = torch.linalg.inv(adj_A_new1)

        gcn_x1 = self.gcn(input_z, origin_A)
        # gcn_x1 = self.batchnorm(gcn_x1)

        H1 = F.leaky_relu((self.out_fc1(gcn_x1)))
        gcn_x2 = self.gcn(self.out_fc2(H1),origin_A)
        gcn_x2 = self.batchnorm(gcn_x2)
        H2 = F.leaky_relu(self.out_fc3(gcn_x2))
        H3 = self.out_fc4(H2)

        mat_z = H3+Wa

        H4 = F.leaky_relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H4)

        return mat_z, out