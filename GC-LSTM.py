import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

best_load=35

def f1(x,w):
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            if w[i][j]>0:
                if x[i]<best_load and x[j]>best_load:
                    #print(i,j,w[i][j])
                    available=(best_load-x[i])*w[i][j]
                    x[i]=x[i]+available
                    x[j]=x[j]-available
                    #print(available,i,j)  
    return x

def f2(nodes,a):
  #nodes=(batch_size, node_num, 1)
  #a=(batch_size, node_num, node_num)
  #print(nodes.shape)
  for z in range(nodes.shape[0]):
    for i in range(nodes.shape[1]):
        for j in range(nodes.shape[1]):
            if a[z,i,j]>0:
                #print(nodes[z,i].shape)
                if nodes[z,i]<best_load and nodes[z,j]>best_load:
                    available=(best_load-nodes[z,i])*a[z,i,j]
                    nodes[z,i]=nodes[z,i]+available
                    nodes[z,j]=nodes[z,j]-available
    return nodes




#a=torch.ones(node_num,node_num)
def onebian(a):
    for z in range(a.shape[0]):
        for i in range(a.shape[1]):
                for j in range(i,a.shape[1]):
                    if a[z,i][j]>=a[z,j][i]:
                        a[z,j][i]=0
                    if a[z,i][j]<a[z,j][i]:
                        a[z,i][j]=0
                
    return a

def compare_and_update(a, i, j):
    # Create a mask for the comparison condition
    mask = a[:, i, j] >= a[:, j, i]

    # Update the elements using torch.where
    a[:, i, j] = torch.where(mask, a[:, i, j], 0)
    a[:, j, i] = torch.where(mask, 0, a[:, j, i])

    return a

class GraphConvolution(nn.Module):

    def __init__(self, window_size, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weights = nn.Parameter(
            torch.Tensor(window_size,in_features, out_features)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.weights)

    def forward(self, adjacency, nodes):
        """
        :param adjacency: FloatTensor (batch_size, window_size, node_num, node_num)
        :param nodes: FloatTensor (batch_size, window_size, node_num, in_features)
        :return output: FloatTensor (batch_size, window_size, node_num, out_features)
        """
        batch_size = adjacency.size(0)
        window_size, in_features, out_features = self.weights.size()
        weights = self.weights.unsqueeze(0).expand(batch_size, window_size, in_features, out_features)
        output = adjacency.matmul(nodes).matmul(weights)
        #print(output.size())
        return output

class Generator(nn.Module):

    def __init__(self, window_size, node_num, in_features, out_features, lstm_features):
        super(Generator, self).__init__()
        self.window_size = window_size
        self.node_num = node_num
        self.in_features = in_features
        self.out_features = out_features
        self.gcn = GraphConvolution(window_size, in_features, out_features)
        self.lstm = nn.LSTM(
            input_size=out_features * node_num,
            hidden_size=lstm_features,
            num_layers=1,
            batch_first=True
        )
        self.f1=nn.Linear(lstm_features, node_num * out_features)
        self.f2=nn.Linear(2*out_features,1)

    def forward(self, in_shots,in_node,a):
        """
        :param in_shots: FloatTensor (batch_size, window_size, node_num, node_num)
        :return out_shot: FloatTensor (batch_size, node_num * node_num)
        """
        batch_size, window_size, node_num = in_shots.size()[0: 3]
        eye = torch.eye(node_num).cuda().unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
        in_shots = in_shots + eye
        diag = in_shots.sum(dim=-1, keepdim=True).pow(-0.5).expand(in_shots.size()) * eye
        adjacency = diag.matmul(in_shots).matmul(diag)
        #nodes = torch.rand(batch_size, window_size, node_num, self.in_features).cuda()
        nodes=in_node
        gcn_output = self.gcn(adjacency, nodes)
        #print(gcn_output.size())
        gcn_output = gcn_output.view(batch_size, window_size, -1)
        #print(gcn_output.size())
        _, (hn, _) = self.lstm(gcn_output)
        hn = hn.permute(1, 0, 2).contiguous().view(batch_size, -1)
        output = self.f1(hn)
        #print(output.size())
        output=output.view(batch_size,self.node_num,self.out_features)
        #print(output.size())
        #print(output[:,1,1].size())
        #print(self.f2(torch.cat((output[:,1,:],output[:,1,:]),dim=1)).size())
        a=torch.ones(batch_size,node_num,node_num).cuda()
        for i in range(node_num):
            for j in range(node_num):
               #a[][i][j] =float(self.f2(torch.cat((output[:,i,:],output[:,j,:]),dim=1)))
                #a[:,i,j] =float(self.f2(torch.cat((output[:,i,:],output[:,j,:]),dim=1)))
                aa  =self.f2(torch.cat((output[:,i,:],output[:,j,:]),dim=1)).squeeze(1)
                a[:,i,j] =aa
        #print(a)
        #a=torch.tensor(a).cuda()
        for i in range(node_num):
            for j in range(node_num):
                a=compare_and_update(a,i,j)
        #print(a.shape)
        a=F.normalize(a,p=1,dim=2)
        
        return a

import yaml
import os
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
from utils import LPDataset

config = yaml.safe_load(open('config.yml'))

node_num = config['node_num']
window_size = config['window_size']

base_path = os.path.join('./data/', config['dataset'])
train_save_path = os.path.join(base_path, 'train1.npy')

train_data = LPDataset(train_save_path, window_size)
sample_data = LPDataset(train_save_path, window_size)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=config['batch_size'],
    shuffle=True,
    pin_memory=True
)
sample_loader = DataLoader(
    dataset=sample_data,
    batch_size=config['batch_size'],
    shuffle=False,
    pin_memory=True
)

generator = Generator(
    window_size=window_size,
    node_num=node_num,
    in_features=config['in_features'],
    out_features=config['out_features'],
    lstm_features=config['lstm_features']
)
generator = generator.cuda()
pretrain_optimizer = optim.RMSprop(generator.parameters(), lr=config['pretrain_learning_rate'])
a=torch.ones(config['batch_size'],node_num,node_num).cuda()
for epoch in range(config['pretrain_epoches']):
    for i, data in enumerate(train_loader):
        pretrain_optimizer.zero_grad()
        in_data,out_data=data
        in_data=torch.split(in_data,[node_num,1],dim=3)
        in_data1=in_data[0].cuda()
        in_node=in_data[1].cuda()
        
        out_data=torch.split(out_data,[node_num,1],dim=2)
        out_data1=out_data[0].cuda()
        out_node=out_data[1].cuda()

        A = generator(in_data1,in_node,a)
        y=f2(out_node,A)
        loss = F.mse_loss(y, best_load*torch.ones(config['batch_size'],node_num,1).cuda()).requires_grad_(True)
        
        loss.backward(retain_graph=True)
        pretrain_optimizer.step()
        print(loss)
        
