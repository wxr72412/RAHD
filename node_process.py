import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import os

def get_snapshot(path, node_num, max_thres):
    with open(path, 'r') as file:
        snapshot = np.zeros(shape=(node_num, node_num), dtype=np.float32)
        #print(snapshot.shape)
        for line in file:
            line = line.strip().split(' ')
            node1 = int(line[0])
            node2 = int(line[1])
            edge = float(line[2])
            edge = min(edge, max_thres)
            snapshot[node1, node2] = edge
            snapshot[node2, node1] = edge
    snapshot /= max_thres
    return snapshot

class LPDataset(Dataset):

    def __init__(self, path, window_size):
        super(LPDataset, self).__init__()
        self.data = torch.from_numpy(np.load(path))
        self.window_size = window_size
        self.num = self.data.size(0) - window_size

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data[item: item + self.window_size], self.data[item + self.window_size]
    
config = yaml.safe_load(open('config.yml', 'r'))

# build path

base_path = os.path.join('data\\', config['dataset'])
raw_base_path = os.path.join(base_path, 'raw')
train_save_path = os.path.join(base_path, 'train.npy')
test_save_path = os.path.join(base_path, 'test.npy')