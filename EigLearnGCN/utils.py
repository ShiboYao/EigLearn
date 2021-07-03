import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, eigs
import networkx as nx
import torch
import pickle as pkl
import sys


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset_str, mat, k, state):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #adj += adj.T #make it symetrical
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    features = row_normalize(features)
    
    if mat=="sym":
        adj = sym_normalize_adj(adj + sp.eye(adj.shape[0]))
    elif mat=="poly":
        adj = preprocess_high_order_adj(adj,5,1e-4)
    else:
        print("Use either sym or poly")
        exit(1)

    eigval, eigvec_mat = eigsh(adj+adj.T, k=k, tol=1e-8, which='LM')
    #eigval, eigvec_mat = eigs(adj, k=k, tol=1e-8, which='LM')
    #eigval = eigval.real
    #eigvec_mat = eigvec_mat.real
    #print(eigvec_mat.T.dot(eigvec_mat))
    '''
    with open(dataset_str+str(k)+'.pkl', 'rb') as handle:
        a = pkl.load(handle)
    eigvec_mat = a[0]
    eigval = a[1]
    
    a = [eigvec_mat,eigval]
    with open(dataset_str+str(k)+'.pkl', 'wb') as handle:
        pkl.dump(a, handle, protocol=pkl.HIGHEST_PROTOCOL)
    '''
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx = list(idx_train) + list(idx_val) + idx_test
    
    if state is not None:
        state.shuffle(idx)
    else:
        np.random.shuffle(idx)
    
    idx_val = idx[:500]
    if dataset_str == 'cora':
        idx_train = idx[500:500+140]
        idx_test = idx[500+140:]
    elif dataset_str == 'citeseer':
        idx_train = idx[500:500+120]
        idx_test = idx[500+120:]
    elif dataset_str == 'pubmed':
        idx_train = idx[500:500+60]
        idx_test = idx[500+60:]
    
    features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    eigvec_mat = torch.FloatTensor(eigvec_mat)
    eigval = torch.FloatTensor(eigval)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, eigval, eigvec_mat, features, labels, idx_train, idx_val, idx_test


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


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels) * 100


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
