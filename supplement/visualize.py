import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("sens_reg/sens_reg.csv", header=None, sep=' ')
df = df.iloc[:,range(0,df.shape[1],2)]
data = ["Cora", "CiteSeer", "PubMed"]
k_num = [5,10,15,20,30,40,50,100,150,200,500,1000]
weight_decay = ["4e-3", "4e-4", "4e-5", "0"]

loss_pre = df.iloc[range(0,len(df),4)]
loss_post = df.iloc[range(1,len(df),4)]
acc_pre = df.iloc[range(2,len(df),4)]
acc_post = df.iloc[range(3,len(df),4)]

loss_reduce = loss_pre.values - loss_post.values
acc_improve = acc_post.values - acc_pre.values

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18,10), sharex=True)

for i,a in enumerate(ax[0]):
    a.set_title(data[i], fontsize=16)
plt.setp(ax, xticks=range(len(k_num)), xticklabels=k_num)

ax[0][0].set_ylabel("Testing Accuracy Improvement", fontsize=16)
for i,col in enumerate(ax[0]):
    for j in range(4):
        col.plot(range(len(k_num)), acc_improve[:,4*i+j], label=weight_decay[j], linestyle='dashed', linewidth=.5)
        col.scatter(range(len(k_num)), acc_improve[:,4*i+j], label=weight_decay[j], marker='o')

ax[1][0].set_ylabel("Testing Loss Reduction", fontsize=16)
ax[1][0].set_xlabel("k: number of eigenvalue to perturb", fontsize=12)
for i,col in enumerate(ax[1]):
    for j in range(4):
        col.plot(range(len(k_num)), loss_reduce[:,4*i+j], label=weight_decay[j], linestyle='dashed', linewidth=.5)
        col.scatter(range(len(k_num)), loss_reduce[:,4*i+j], label=weight_decay[j], marker='o')
    
 
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines[4:], labels[4:], loc = 'center', fontsize=14)

fig.savefig("sens_reg.pdf")


df = pd.read_csv("sens_lr/sens_lr.csv", header=None, sep=' ')
df = df.iloc[:,range(0,df.shape[1],2)]
data = ["Cora", "CiteSeer", "PubMed"]
k_num = [5,10,15,20,30,40,50,100,150,200,500,1000]
lr = ["2e-3", "5e-3", "1e-2", "2e-2"]

loss_pre = df.iloc[range(0,len(df),4)]
loss_post = df.iloc[range(1,len(df),4)]
acc_pre = df.iloc[range(2,len(df),4)]
acc_post = df.iloc[range(3,len(df),4)]

loss_reduce = loss_pre.values - loss_post.values
acc_improve = acc_post.values - acc_pre.values

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18,10), sharex=True)

for i,a in enumerate(ax[0]):
    a.set_title(data[i], fontsize=16)
plt.setp(ax, xticks=range(len(k_num)), xticklabels=k_num)

ax[0][0].set_ylabel("Testing Accuracy Improvement", fontsize=16)
for i,col in enumerate(ax[0]):
    for j in range(4):
        col.plot(range(len(k_num)), acc_improve[:,4*i+j], label=lr[j], linestyle='dashed', linewidth=.5)
        col.scatter(range(len(k_num)), acc_improve[:,4*i+j], label=lr[j], marker='o')

ax[1][0].set_ylabel("Testing Loss Reduction", fontsize=16)
ax[1][0].set_xlabel("k: number of eigenvalue to perturb", fontsize=12)
for i,col in enumerate(ax[1]):
    for j in range(4):
        col.plot(range(len(k_num)), loss_reduce[:,4*i+j], label=lr[j], linestyle='dashed', linewidth=.5)
        col.scatter(range(len(k_num)), loss_reduce[:,4*i+j], label=lr[j], marker='o')
    
 
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines[4:], labels[4:], loc = 'center', fontsize=14)

fig.savefig("sens_lr.pdf")
