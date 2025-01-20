import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric_temporal.nn import ASTGCN
import numpy as np
import scipy.sparse as sp
import csv
import torch_sparse

file_path = 'juli_50.csv'
def read_adjacency_matrix_from_csv(file_path):
    adjacency_matrix = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            adjacency_matrix.append(list(map(int, row)))
    return adjacency_matrix

A = read_adjacency_matrix_from_csv(file_path)
A1=torch.tensor(A,dtype=torch.float32)



batch_size=32
window_size=40
node_num=50
in_features=1
out_features=64
#A1=A1.expand(batch_size,window_size,node_num,node_num).cuda()


A1=sp.coo_matrix(A1)
values = A1.data
indices = np.vstack((A1.row, A1.col))
A1=torch.LongTensor(indices)
A1=A1.cuda()


x=torch.randn(batch_size,window_size,node_num,in_features)
x=x.permute(0,2,3,1).cuda()

#print(x.shape)
edge_index=A1.cuda()
model=ASTGCN(nb_block=1,in_channels=1,K=3, nb_chev_filter=64, nb_time_filter=64, time_strides=2, num_for_predict=1,len_input=40,num_of_vertices=node_num).cuda()
y=model(x,edge_index)
print(y.shape)