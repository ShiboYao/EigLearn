from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

from collections import deque

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed_data', type=int, default=0, 
                    help='Random seed for data split.')
parser.add_argument('--seed_train', type=int, default=0, 
                    help='Random seed for training.')
parser.add_argument('--epochs_w', type=int, default=200,
                    help='Number of epochs to train w.')
parser.add_argument('--epochs_s', type=int, default=500,
                    help='Number of epochs to train spectra.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--lr_s', type=float, default=0.002,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--weight_decay_s', type=float, default=4e-3,
                    help='Weight decay for learning pertubation.')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--k', type=int, default=40,
                    help='Number of eigenvalues to perturb.')
parser.add_argument('--data', type=str, default='cora',
                    help='Which dataset: cora, citeseer, pubmed.')
parser.add_argument('--mat', type=str, default='sym',
                    help='Graph filter matrix.')
parser.add_argument('--perturb', type=float, default=100.,
                    help='perturbation range of eigenvalues')
parser.add_argument('--verbose', type=int, default=0,
                    help='Whether to print out training steps.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
state = None

if args.seed_data > 0:
    state = np.random.RandomState(args.seed_data)
if args.seed_train > 0:
    torch.manual_seed(args.seed_train)
    if args.cuda:
        torch.cuda.manual_seed(args.seed_train)

# Load data
adj, eigval, eigvec_mat, features, labels, idx_train, idx_val, idx_test = load_data(args.data, args.mat, args.k, state)

# compute upper and lower bound for eigenvalues
eigval = torch.abs(eigval)
upper = eigval * args.perturb
lower = -upper # feel free to investigate the perturbation :-)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            k=args.k,
            dropout=args.dropout)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    eigvec_mat = eigvec_mat.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    upper = upper.cuda()
    lower = lower.cuda()

def train_w(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, eigvec_mat)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj, eigvec_mat)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    if args.verbose == 1:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.2f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))
    
    return loss_val


def train_spectra(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, eigvec_mat)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
   
    optimizer.step()
    
    #model.delta.data = torch.min(model.delta.data, upper)
    #model.delta.data = torch.max(model.delta.data, lower)
    
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj, eigvec_mat)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    if args.verbose == 1:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.2f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))
    
    return loss_val

    
def test():
    model.eval()
    output = model(features, adj, eigvec_mat)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.2f}".format(acc_test.item()))


# Learn dense weight matrices
print("***** train w *****")   
optimizer = optim.Adam([
                        {'params':model.gc1_weight},
                        {'params':model.gc1_bias},
                        {'params':model.gc2_weight},
                        {'params':model.gc2_bias}
                       ], 
                       lr=args.lr, weight_decay=args.weight_decay)

t_total = time.time()
for epoch in range(args.epochs_w):
    err = train_w(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()


# Learn Eigen Perturbation
print("***** train spectra *****")
optimizer = optim.Adam([
                        {'params':model.delta}
                       ],
                       lr=args.lr_s, weight_decay=args.weight_decay_s)

val_err = deque([1e8]*3)
t_total = time.time()
for epoch in range(args.epochs_s):
    err = train_spectra(epoch)
    if err >= sum(val_err) / len(val_err) * .9999:
        print("Early stop at epoch ", epoch)
        break
    val_err.popleft()
    val_err.append(err)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
