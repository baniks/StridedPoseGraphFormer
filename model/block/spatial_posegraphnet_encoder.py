from __future__ import absolute_import

import torch.nn as nn
import torch
import math
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GConv(Module):
    def __init__(self, in_features, out_features, adj, bias=True, num_groups=1):
        """
        Args:
            A: (num_groups, num_nodes, num_nodes), normalized adjacent matrix.
        """
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj

        self.num_groups = num_groups
        self.num_joints =  adj.shape[2] if len(adj.shape) > 3 else adj.shape[1]
        self.weight = nn.Linear(in_features, self.num_groups * out_features, bias=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight.weight)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        """
        Args:
            x: (batch_size, num_nodes, in_channels)
        Retrun:
            out: (batch_size, num_nodes, out_channels)
        """

        x = self.weight(input) # BS x J X (num_groups*out_channels)
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.num_joints, self.num_groups, self.out_features) # BS x J x G x C_out

        x = x.transpose(-3,-2) # BS x G x J x C_out
        x = torch.matmul(adj, x)
        x = torch.sum(x, dim=-3) # BS x J x C_out

        if self.bias is not None:
            x = x + self.bias

        return x


class GCNResidualBlockBN_New(nn.Module):
    """
    graph linear block with BN at joint feature level with new adjaceny
    """
    def __init__(self, nfeat, nout, dropout, num_joints, adj, bias=True, num_groups=1):
        super(GCNResidualBlockBN_New, self).__init__()

        self.nout = nout
        self.num_joints = num_joints
        self.adj = adj
        self.num_groups = num_groups

        self.gc1 = GConv(nfeat, nout, self.adj, bias, self.num_groups)
        self.gc2 = GConv(nout, nout, self.adj, bias, self.num_groups)
        self.dropout = dropout

        # norm applied at individual joints' feature level
        self.bn1 = nn.BatchNorm1d(num_joints * nout)
        self.bn2 = nn.BatchNorm1d(num_joints * nout)

        if (nfeat == nout):
            self.residual_flag = "same"
        else:
            self.residual_flag = "diff"
            self.residual_gc = GConv(nfeat, nout, self.adj)
            self.residual_bn =nn.BatchNorm1d(num_joints*nout)

    def forward(self, x, adj):

        batch_size = x.size(0)

        if self.residual_flag == "same":
            residual = x
        else:
            residual = self.residual_gc(x, adj)
            residual = residual.view(batch_size, self.num_joints*self.nout)
            residual = self.residual_bn(residual)
            residual = residual.view(batch_size, self.num_joints, self.nout)

        out = self.gc1(x, adj)
        out = out.view(batch_size, self.num_joints*self.nout)

        out = self.bn1(out)
        out = out.view(batch_size, self.num_joints, self.nout)
        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)

        out = self.gc2(out, adj)
        out = out.view(batch_size, self.num_joints*self.nout)
        out = self.bn2(out)
        out = out.view(batch_size, self.num_joints, self.nout)

        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = out + residual
        return out


class GCNResidualBN_adjL(nn.Module):
    """
    PoseGraphNet: graph linear with BN at joint feature level with Learnable adj
    """

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L


    def laplacian_batch(self, A_hat):
        if len(A_hat.shape) == 3:
            batch, N = A_hat.shape[:2]
            D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
            L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        elif len(A_hat.shape) > 3:
            batch, group, N = A_hat.shape[:3]
            D_hat = (torch.sum(A_hat, -2) + 1e-5) ** (-0.5)
            L = D_hat.view(batch, group, N, 1) * A_hat * D_hat.view(batch, group, 1, N)
        return L


    def normalize_undigraph_batch(self, A_hat):

        if len(A_hat.shape) == 3:
            batch_size, num_node = A_hat.shape[:2]
            A = A_hat[0]
            Dl = torch.sum(A, 0)
            Dn = torch.zeros((num_node, num_node)).cuda()
            for i in range(num_node):
                if Dl[i] > 0:
                    Dn[i, i] = Dl[i]**(-0.5)
            DAD = torch.mm(torch.mm(Dn, A), Dn)
            normed_adj = torch.zeros(batch_size, num_node, num_node).cuda()
            normed_adj[:] = DAD

        elif len(A_hat.shape) > 3:

            batch_size, num_group, num_node = A_hat.shape[:3]
            normed_adj = torch.zeros(batch_size, num_group, num_node, num_node).cuda()

            for grp_id in range(num_group):
                grp_adj = A_hat[0, grp_id]
                Dl = torch.sum(grp_adj, 0)
                Dn = torch.zeros((num_node, num_node)).cuda()
                for i in range(num_node):
                    if Dl[i] > 0:
                        Dn[i, i] = Dl[i]**(-0.5)
                normed_adj[:, grp_id] = torch.mm(torch.mm(Dn, grp_adj), Dn)

        return normed_adj


    def __init__(self, nfeat, nhid, nclass, dropout, num_joints, adj, num_groups):
        """
        adj: unnormalized. with self loop. 1 x num_nodes x num_nodes
        """
        super(GCNResidualBN_adjL, self).__init__()
        if adj.shape[0] > 1 and len(adj.shape) == 3:
            adj = adj.unsqueeze(0).repeat(1, 1, 1, 1)

        self.adj = Parameter(adj)
        self.adj_norm = None
        self.num_groups = num_groups
        self.child = [
            (0,1), (0,4), (1,2), (2,3), (4,5), (5,6),
            (0,7), (7,8), (8,9), (9,10),
            (8, 11), (11, 12), (12, 13),
            (8, 14), (14, 15), (15, 16)
        ]
        self.parent = [
            (1,0), (2,1), (3,2), (4,0), (5,4), (6,5),
            (7,0), (8,7), (9,8), (10,9),
            (11,8), (12,11), (13,12),
            (14,8), (15,14), (16,15)
        ]

        self.gc1 = GConv(nfeat, nhid[0], self.adj, bias=True, num_groups=self.num_groups)
        self.gcBlock2 = GCNResidualBlockBN_New(nhid[0], nhid[1], dropout, num_joints, self.adj, bias=True, num_groups=self.num_groups)
        self.gcBlock3 = GCNResidualBlockBN_New(nhid[1], nhid[2], dropout, num_joints, self.adj, bias=True, num_groups=self.num_groups)
        self.gcBlock4 = GCNResidualBlockBN_New(nhid[2], nhid[3], dropout, num_joints, self.adj, bias=True, num_groups=self.num_groups)
        self.gc4 = GConv(nhid[3], nclass, self.adj, bias=True, num_groups=self.num_groups) # uncomment for 3 blocks

        self.dropout = dropout

        self.nhid = nhid
        self.num_joints = num_joints
        self.bn1 = nn.BatchNorm1d(num_joints * nhid[0])
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x : bs x J x 2
        # adj : 1 x J x J

        batch_size = x.size(0)

        # recalculate normalized adj
        self.adj_norm = self.normalize_undigraph_batch(self.adj) # --------- commented for torchscript
        # self.adj_norm = self.adj

        out = self.gc1(x, self.adj_norm)

        out = out.view(batch_size, self.num_joints*self.nhid[0])
        out = self.bn1(out)
        out = out.view(batch_size, self.num_joints, self.nhid[0])
        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)

        out = self.gcBlock2(out, self.adj_norm)
        out = self.gcBlock3(out, self.adj_norm)
        out = self.gcBlock4(out, self.adj_norm)
        out = self.gc4(out, self.adj_norm)

        return out


class SpatialPoseGraphNet(GCNResidualBN_adjL):

    def __init__(self, nfeat, nhid, nclass, dropout, num_joints, adj, num_groups, n_embedding, pretrain=False):
        """
        adj: unnormalized. with self loop. 1 x num_nodes x num_nodes
        """
        super(SpatialPoseGraphNet, self).__init__(nfeat, nhid, nclass, dropout, num_joints, adj, num_groups)
        self.gc4 = GConv(nhid[3], n_embedding, self.adj, bias=True, num_groups=self.num_groups)
        self.pretrain = pretrain
        if self.pretrain:
            self.gc5 = GConv(n_embedding, nclass, self.adj, bias=True, num_groups=self.num_groups)


    def forward(self, x):
        # x : bs x J x 2
        # adj : G x J x J

        batch_size = x.size(0)

        # recalculate normalized adj
        self.adj_norm = self.normalize_undigraph_batch(self.adj) # --------- commented for torchscript
        # self.adj_norm = self.adj

        out = self.gc1(x, self.adj_norm)

        out = out.view(batch_size, self.num_joints*self.nhid[0])
        out = self.bn1(out)
        out = out.view(batch_size, self.num_joints, self.nhid[0])
        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)

        out = self.gcBlock2(out, self.adj_norm)
        out = self.gcBlock3(out, self.adj_norm)
        out = self.gcBlock4(out, self.adj_norm)
        out = self.gc4(out, self.adj_norm)
        if self.pretrain:
            out = self.gc5(out, self.adj_norm)


        return out
