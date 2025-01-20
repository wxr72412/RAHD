import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

best_load=35

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


def compare_and_update(a, i, j):
    # Create a mask for the comparison condition
    mask = a[:, i, j] >= a[:, j, i]

    # Update the elements using torch.where
    a[:, i, j] = torch.where(mask, a[:, i, j], 0)
    a[:, j, i] = torch.where(mask, 0, a[:, j, i])

    return a


class Generator(nn.Module):

    def __init__(self, window_size, node_num, in_features, out_features, lstm_features):
        super(Generator, self).__init__()
        self.window_size = window_size
        self.node_num = node_num
        self.in_features = in_features
        
        
        self.lstm = nn.LSTM(
            input_size=node_num,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.f1=nn.Linear(256, node_num * node_num)


    def forward(self, in_shots,in_node):
        #in_shots=(batch_size, window_size, node_num, in_features)
        #in_node=(batch_size, window_size, node_num, 1)
        batch_size = in_shots.size(0)
        in_node=in_node.squeeze(3)
        _, (hn, _) = self.lstm(in_node)
        hn = hn.permute(1, 0, 2).contiguous().view(batch_size, -1)
        #print(hn.shape)
        output = self.f1(hn)
        a=output.view(batch_size,self.node_num,self.node_num)
        
        for i in range(node_num):
            for j in range(node_num):
                a=compare_and_update(a,i,j)
        #print(a.shape)
        #a=F.normalize(a,p=1,dim=2)
        
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

        A = generator(in_data1,in_node)
        A= torch.mul(A,out_data1)
        A=F.normalize(A,p=1,dim=2)
        y=f2(out_node,A)
        loss = F.mse_loss(y, best_load*torch.ones(config['batch_size'],node_num,1).cuda()).requires_grad_(True)
        
        loss.backward(retain_graph=True)
        pretrain_optimizer.step()
        print(loss)
        
