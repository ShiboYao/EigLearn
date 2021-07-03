import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_sparse import fill_diag, sum as sparsesum, mul
import torch_geometric.transforms as T
from gcn import GCN

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger


def sym_normalize_adj(adj):
    """symmetrically normalize adjacency matrix"""
    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(np.maximum(degree, np.finfo(float).eps), -0.5)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def row_normalize(adj):
    """row normalize"""
    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1)).flatten()
    d_mat_inv = sp.diags(1./np.maximum(degree, np.finfo(float).eps))
    return d_mat_inv.dot(adj).tocoo()


def preprocess_high_order_adj(adj, order, eps):
    """A higher-order polynomial with sparsification"""
    adj = row_normalize(adj)
    adj_sum = adj
    cur_adj = adj
    for i in range(1, order):
        cur_adj = cur_adj.dot(adj)
        adj_sum += cur_adj
    adj_sum /= order

    adj_sum.setdiag(0)
    adj_sum.data[adj_sum.data<eps] = 0
    adj_sum.eliminate_zeros()

    adj_sum += sp.eye(adj.shape[0])
    return sym_normalize_adj(adj_sum + adj_sum.T)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--mat', type=str, default='sym')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_s', type=float, default=0.002)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--epochs_s', type=int, default=50)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    adj_t = data.adj_t.to_symmetric()
    adj_t = adj_t.to_scipy('coo')
    if args.mat=='sym':
        adj_t = sym_normalize_adj(adj_t + sp.eye(adj_t.shape[0]))
    else:
        adj_t = preprocess_high_order_adj(adj_t,3,1e-4)
    data.adj_t = sparse_mx_to_torch_sparse_tensor(adj_t)
    data = data.to(device)

    adj = adj_t
    eigval, eigvec_mat = eigsh(adj, k=args.k, tol=1e-8, which='LM')
    eigvec_mat = torch.FloatTensor(eigvec_mat).cuda()

    split_idx = dataset.get_idx_split()
    train_num = split_idx['train'].shape[0]
    valid_num = split_idx['valid'].shape[0]
    test_num = split_idx['test'].shape[0]
    idx = torch.randperm(train_num + valid_num + test_num)
    split_idx['train'] = idx[:train_num]
    split_idx['valid'] = idx[train_num:(train_num+valid_num)]
    split_idx['test'] = idx[-test_num:]
    train_idx = split_idx['train'].to(device)

    model = GCN(data.num_features, args.hidden_channels,
                dataset.num_classes, args.k, eigvec_mat,
                args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger1 = Logger(args.runs, args)
    logger2 = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = optim.Adam([
                                {'params':model.gc1_weight},
                                {'params':model.gc1_bias},
                                {'params':model.bn1.weight},
                                {'params':model.bn1.bias},
                                {'params':model.gc2_weight},
                                {'params':model.gc2_bias},
                                {'params':model.bn2.weight},
                                {'params':model.bn2.bias},
                                {'params':model.gc3_weight},
                                {'params':model.gc3_bias},
                                ],
                                lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger1.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
        logger1.print_statistics(run)

        optimizer = optim.Adam([
                                {'params':model.delta},
                                ],
                                lr=args.lr_s)
        for epoch in range(1, 1 + args.epochs_s):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger2.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger2.print_statistics(run)
    logger1.print_statistics()
    logger2.print_statistics()


if __name__ == "__main__":
    main()
