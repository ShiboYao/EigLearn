import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, k, dropout):
        super(GCN, self).__init__()
        self.gc1_weight = Parameter(torch.FloatTensor(nfeat,nhid))
        self.gc1_bias = Parameter(torch.FloatTensor(nhid))
        self.gc2_weight = Parameter(torch.FloatTensor(nhid,nclass))
        self.gc2_bias = Parameter(torch.FloatTensor(nclass))
        self.reset_parameters()
        self.delta = Parameter(torch.zeros(k))
        self.dropout = dropout

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.gc1_weight.size(1))
        self.gc1_weight.data.uniform_(-stdv, stdv)
        self.gc1_bias.data.fill_(0.0)
        stdv = 1. / math.sqrt(self.gc2_weight.size(1))
        self.gc2_weight.data.uniform_(-stdv, stdv)
        self.gc2_bias.data.fill_(0.0)

    def sparse_dropout(self, x):
        if not self.training:
            return x
        mask = ((torch.rand(x._values().size()) +
                 (1-self.dropout)).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / (1-self.dropout))
        return torch.sparse.FloatTensor(rc, val, x.shape)

    def forward(self, x, adj, eigvec_mat): #columns are the eigenvecs
        x = self.sparse_dropout(x)
        support = torch.mm(x, self.gc1_weight)
        tilda = torch.mm(eigvec_mat.T, support)
        r = torch.mm(eigvec_mat, torch.mm(torch.diag(self.delta), tilda))
        x = torch.spmm(adj, support) + r + self.gc1_bias
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        support = torch.mm(x, self.gc2_weight)
        tilda = torch.mm(eigvec_mat.T, support)
        r = torch.mm(eigvec_mat, torch.mm(torch.diag(self.delta), tilda))
        x = torch.spmm(adj, support) + r + self.gc2_bias
        return F.log_softmax(x, dim=1)
