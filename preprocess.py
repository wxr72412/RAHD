import yaml
import os
import numpy as np
from utils import get_snapshot

# load config

config = yaml.safe_load(open('config.yml', 'r'))

# build path

base_path = os.path.join('data\\', config['dataset'])
raw_base_path = os.path.join(base_path, 'raw')
train_save_path = os.path.join(base_path, 'train1.npy')
test_save_path = os.path.join(base_path, 'test1.npy')

# load data

num = len(os.listdir(raw_base_path))
#print(num)
data = np.zeros(shape=(num, config['node_num'], config['node_num']+1), dtype=np.float32)
#print(data.shape)
for i in range(num):
    path = os.path.join(raw_base_path, 'edge_list_' + str(i) + '.txt')
    path1='node\\'+str(i)+'.csv'
    if not os.path.exists(path):
        with open(path, 'w') as file:
            print('create file: ' + path)
    adj = get_snapshot(path, config['node_num'], config['max_thres'])
    with open(path1,'r') as f:
        array1=f.read()
        #array1=np.array(array1)
        array1=list(map(int,array1.split()))
        node=np.array(array1)
        node=node.reshape(-1,1)
    data[i]=np.hstack((adj,node))

total_num = num - config['window_size']
test_num = int(config['test_rate'] * total_num)
train_num = total_num - test_num
print(train_num)
train_data = data[0: train_num + config['window_size']]
test_data = data[train_num: num]

# save data

np.save(train_save_path, train_data)
np.save(test_save_path, test_data)
