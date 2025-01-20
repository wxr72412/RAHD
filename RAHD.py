import yaml
import os
import torch
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
from utils import LPDataset
from torch.nn import init
import math
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
from utils import MSE, EdgeWiseKL, MissRate
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


class MyConv(nn.Module):
    def __init__(self, embedding_dim, num_filter, output_dim,filter_sizes,kernel_size, skip, drop):
        super(MyConv, self).__init__()
        self.skip = skip
        self.kernel_size = kernel_size
        #textcnn
        #filter_sizes = [2, 3, 4]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filter,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        #skipcnn
        #kernel_size = [3, 5]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=(kernel_size[0],embedding_dim),dilation=(skip,1),stride=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=(kernel_size[1],embedding_dim),dilation=(skip,1),stride=1)

        self.fc = nn.Linear((len(filter_sizes)+len(kernel_size)) * num_filter, output_dim)


        self.dropout = nn.Dropout(drop)
    def forward(self, x):
        x =x.cuda()
        x=x.unsqueeze(1)
        x0 = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pool_x0 = [F.avg_pool1d(i, i.shape[2]).squeeze(2) for i in x0]


        #x1 = in_data(x, self.skip, self.kernel_size[0]).cuda()
        #x1 = x1.unsqueeze(1)
        x1=self.conv1(x)
        x1=F.relu(x1).squeeze(3)
        pool_x1 = F.avg_pool1d(x1, x1.shape[2]).squeeze(2)

        #x2 = in_data(x, self.skip, self.kernel_size[1]).cuda()
        #x2 = x2.unsqueeze(1)
        x2=self.conv2(x)
        x2=F.relu(x2).squeeze(3)
        pool_x2 = F.avg_pool1d(x2, x2.shape[2]).squeeze(2)
        #print(pool_x2.shape)
        x3 = torch.cat(pool_x0 + [pool_x1, pool_x2], 1)
        #print(x3.shape)
        #x3 = F.relu(x3)
        x3 = self.fc(x3)
        #x3 = self.dropout(x3)
        return x3


class Generator(nn.Module):

    def __init__(self, window_size, node_num, in_features, out_features):
        super(Generator, self).__init__()
        self.window_size = window_size
        self.node_num = node_num
        self.in_features = in_features
        self.out_features = out_features
        self.gcn = GraphConvolution(window_size, in_features, out_features)
        self.cnn = MyConv(embedding_dim=node_num*out_features, num_filter=100, output_dim=node_num*node_num,filter_sizes=[2,3,5],kernel_size=[2,3], skip=6, drop=0.2)


    def forward(self, in_shots):
        """
        :param in_shots: FloatTensor (batch_size, window_size, node_num, node_num)
        :return out_shot: FloatTensor (batch_size, node_num * node_num)
        """
        batch_size, window_size, node_num = in_shots.size()[0: 3]
        #print(in_shots.size())
        eye = torch.eye(node_num).cuda().unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
        in_shots = in_shots + eye
        diag = in_shots.sum(dim=-1, keepdim=True).pow(-0.5).expand(in_shots.size()) * eye
        adjacency = diag.matmul(in_shots).matmul(diag).cuda()
        nodes = torch.rand(batch_size, window_size, node_num, self.in_features).cuda()
        gcn_output = self.gcn(adjacency, nodes)
        #print(gcn_output.size())
        gcn_output = gcn_output.view(batch_size, window_size, -1).cuda()
        #print(gcn_output.size())
        out = self.cnn(gcn_output)
        return out


config = yaml.safe_load(open('config.yml'))

node_num = config['node_num']
window_size = config['window_size']

base_path = os.path.join('./data/', config['dataset'])
train_save_path = os.path.join(base_path, 'train.npy')

train_data = LPDataset(train_save_path, window_size)
#print('train data size: {}'.format(len(train_data)))

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
)

mse = nn.MSELoss(reduction='sum')
#mse = utils.MSELoss()
generator=generator.cuda()

pretrain_optimizer = optim.RMSprop(generator.parameters(), lr=config['pretrain_learning_rate'])
generator_optimizer = optim.RMSprop(generator.parameters(), lr=config['g_learning_rate'])


loss_list=[]

mloss = 1000000
for epoch in range(config['pretrain_epoches']):
    for i, data in enumerate(train_loader):
        
        pretrain_optimizer.zero_grad()
        in_shots, out_shot = data
        in_shots, out_shot = in_shots.cuda(), out_shot.cuda()
        
        outpred = generator(in_shots)
        out_shot = out_shot.view(outpred.shape[0], -1)
        loss = mse(outpred, out_shot)
        #print(loss)
        loss.backward()
        pretrain_optimizer.step()
        nn.utils.clip_grad_norm_(generator.parameters(), config['gradient_clip'])
        pretrain_optimizer.step()
        print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, loss.item()))
        loss_list.append(loss.item())
    
        #print(outpred.shape)


torch.save(generator, 'mscnn1.pkl')

cnn = torch.load('mscnn1.pkl').cuda()

test_save_path = os.path.join(base_path, 'test.npy')
test_data = LPDataset(test_save_path, window_size)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=config['batch_size'],
    shuffle=True,
    pin_memory=True
)
print('test data size: {}'.format(len(test_data)))
total_samples = 0
total_mse = 0
total_kl = 0
total_missrate = 0

for i, data in enumerate(test_loader):
    in_shots, out_shot = data
    in_shots, out_shot = in_shots.cuda(), out_shot.cuda()
    predicted_shot = cnn(in_shots)
    predicted_shot = predicted_shot.view(-1, config['node_num'], config['node_num'])
    predicted_shot = (predicted_shot + predicted_shot.transpose(1, 2)) / 2
    for j in range(config['node_num']):
        predicted_shot[:, j, j] = 0
    mask = predicted_shot >= config['epsilon']
    predicted_shot = predicted_shot * mask.float()
    batch_size = in_shots.size(0)
    total_samples += batch_size
    total_mse += batch_size * MSE(predicted_shot, out_shot)*10000
    total_kl += batch_size * EdgeWiseKL(predicted_shot, out_shot)*10000
    total_missrate += batch_size * MissRate(predicted_shot, out_shot)

print('MSE: %.8f' % (total_mse / total_samples))
print('edge wise KL: %.8f' % (total_kl / total_samples))
print('miss rate: %.4f' % (total_missrate / total_samples))