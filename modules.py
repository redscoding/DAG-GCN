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
        adj_A1 = torch.tanh(3.*self.adj_A)

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
        # GCN 聚合公式
        out = torch.matmul(adj, x)   # A * X
        out = self.linear(out)            # (AX)W

        return out


class GCNEncoder(nn.Module):
    def __init__(self, num_n, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol=0.1):
        super(GCNEncoder, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.factor = factor
        self.n_hid = n_hid
        self.num_n = num_n
        self.n_xdims = n_xdims
        self.batch_size = batch_size
        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_hid, bias = True)
        self.fc3 = nn.Linear(n_hid, n_hid, bias = True)
        self.fc4 = nn.Linear(n_hid, n_hid, bias = True)

        self.gcn1 = GCNLayer(n_hid, n_hid)
        self.gcn2 = GCNLayer(n_hid, n_hid)
        self.batchnorm1 = nn.BatchNorm1d(n_hid)
        self.batchnorm2 = nn.BatchNorm1d(n_hid)

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

    def normalize_adj(self, adj):
        """對 A 做對稱正規化: D^{-1/2} A D^{-1/2}"""
        rowsum = adj.sum(1) + 1e-8  # 避免除以 0
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def forward(self, inputs):
        # print(inputs.shape) #(100,11,1)
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error in GCNencoder \n')

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh( 3. *self.adj_A)
        adj_A_pos =F.relu(adj_A1)
        adj_norm = self.normalize_adj(adj_A_pos)

        # MLP
        x = F.leaky_relu((self.fc1(inputs))) #100,11,1 1*64 -> 11 * 64
        # x = (self.fc2(H1)) #11 * 64  64*64 ->11 * 64 x.shape = 100,11,64

        #GCN
        gcn_x = self.gcn1(x, adj_norm)
        # print(f'gcn_x={gcn_x.shape}')
        B, N, H = gcn_x.shape  # 動態抓
        gcn_x = gcn_x.view(B * N, H)
        gcn_x = self.batchnorm1(gcn_x)
        gcn_x = gcn_x.view(B, N, H)
        # print("gcn_x.numel() =", gcn_x.numel())

        H2 = F.leaky_relu(gcn_x)# fc3 fc4 = 64,64

        # GCN
        gcn_x2 =self.gcn2(self.fc4(H2), adj_norm)
        #for testing X
        B, N, H = gcn_x2.shape  # 動態抓
        gcn_x2 = gcn_x2.view(B * N, H)
        gcn_x2 = self.batchnorm2(gcn_x2)
        gcn_x2 = gcn_x2.view(B, N, H)

        logits = gcn_x2+self.Wa

        return x, logits, adj_A1, self.z, self.z_positive, self.adj_A, self.Wa, adj_norm

class GCNDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(GCNDecoder, self).__init__()
        self.gcn1 = GCNLayer(n_hid, n_hid)
        self.gcn2 = GCNLayer(n_hid, n_hid)
        self.batchnorm1 = nn.BatchNorm1d(n_hid)
        self.batchnorm2 = nn.BatchNorm1d(n_hid)
        self.out_fc1 = nn.Linear(n_hid, n_hid, bias = True) #64,64
        self.out_fc2 = nn.Linear(n_hid, n_hid, bias = True)
        self.out_fc3 = nn.Linear(n_hid, n_hid, bias = True)
        self.out_fc4 = nn.Linear(n_hid, n_hid, bias = True)
        self.out_fc5 = nn.Linear(n_hid, 1, bias = True)
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


    def forward(self,inputs, input_z, n_in_node, adj_norm, Wa):

        #adj_A_new1 = (I-A^T)^(-1)
        adj_norm = adj_norm.float()
        input_z = input_z.float()
        # print(f"input_z= {input_z.shape}")#100,11,64
        Wa = Wa.float()
        # adj_A_new1 = preprocess_adj_new1(origin_A).float()  # 這裡就改成 float
        # adj_A_new1 = torch.linalg.inv(adj_A_new1)

        gcn_x1 = self.gcn1(input_z, adj_norm)#100,11,64

        B, N, H = gcn_x1.shape  # 動態抓
        gcn_x1 = gcn_x1.view(B * N, H)
        gcn_x1 = self.batchnorm1(gcn_x1)
        gcn_x1 = gcn_x1.view(B, N, H)


        H1 = F.leaky_relu(gcn_x1)

        gcn_x2 = self.gcn2(self.out_fc1(H1),adj_norm)#100,11,64

        B, N, H = gcn_x2.shape  # 動態抓
        gcn_x2 = gcn_x2.view(B * N, H)
        gcn_x2 = self.batchnorm2(gcn_x2)
        gcn_x2 = gcn_x2.view(B, N, H)


        H2 = F.leaky_relu(gcn_x2)
        H3 = self.out_fc3(self.out_fc2(H2))

        mat_z = H3+Wa

        H4 = F.leaky_relu(self.out_fc4((mat_z)))
        out = self.out_fc5(H4)

        return mat_z, out