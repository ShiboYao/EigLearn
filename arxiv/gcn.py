import math
import torch
import torch.sparse as sp
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
            k, eigvec_mat, dropout):
        super(GCN, self).__init__()
        self.gc1_weight = Parameter(torch.FloatTensor(in_channels,hidden_channels))
        self.gc1_bias = Parameter(torch.FloatTensor(hidden_channels))
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.gc2_weight = Parameter(torch.FloatTensor(hidden_channels,hidden_channels))
        self.gc2_bias = Parameter(torch.FloatTensor(hidden_channels))
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.gc3_weight = Parameter(torch.FloatTensor(hidden_channels,out_channels))
        self.gc3_bias = Parameter(torch.FloatTensor(out_channels))
        self.delta = Parameter(torch.zeros(k))
        self.reset_parameters()
        self.eigvec_mat = eigvec_mat
        self.dropout = dropout

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.gc1_weight.size(-2) + self.gc1_weight.size(-1)))
        self.gc1_weight.data.uniform_(-stdv, stdv)
        self.gc1_bias.data.fill_(0.0)
        self.bn1.weight.data.fill_(1.0)
        self.bn1.bias.data.fill_(0.0)
        stdv = math.sqrt(6.0 / (self.gc2_weight.size(-2) + self.gc2_weight.size(-1)))
        self.gc2_weight.data.uniform_(-stdv, stdv)
        self.gc2_bias.data.fill_(0.0)
        self.bn2.weight.data.fill_(1.0)
        self.bn2.bias.data.fill_(0.0)        
        stdv = math.sqrt(6.0 / (self.gc3_weight.size(-2) + self.gc3_weight.size(-1)))
        self.gc3_weight.data.uniform_(-stdv, stdv)
        self.gc3_bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.gc1_weight)
        tilda = torch.mm(self.eigvec_mat.T, support)
        r = torch.mm(self.eigvec_mat, torch.mm(torch.diag(self.delta), tilda))
        x = torch.spmm(adj, support) + r + self.gc1_bias
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        support = torch.mm(x, self.gc2_weight)
        tilda = torch.mm(self.eigvec_mat.T, support)
        r = torch.mm(self.eigvec_mat, torch.mm(torch.diag(self.delta), tilda))
        x = torch.spmm(adj, support) + r + self.gc2_bias
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        support = torch.mm(x, self.gc3_weight)
        tilda = torch.mm(self.eigvec_mat.T, support)
        r = torch.mm(self.eigvec_mat, torch.mm(torch.diag(self.delta), tilda))
        x = torch.spmm(adj, support) + r + self.gc3_bias
        return F.log_softmax(x, dim=1)
