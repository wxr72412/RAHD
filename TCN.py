import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ResidualBlock(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        res = res[:, :, -x.shape[2]:]
        return x + res

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, n_blocks=2, n_layers=5):
        super(TCN, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.layers = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(2 ** i, out_channels, out_channels, kernel_size, padding=2 ** i),
                ResidualBlock(2 ** i, out_channels, out_channels, kernel_size, padding=2 ** i)
            )
            for i in range(n_layers)
        ])

        self.last_layer = nn.Linear(out_channels, 1)

    def forward(self, x):
        #x = x.transpose(1, 2)
        x = self.first_layer(x)
        #print(x.shape)
        for layer in self.layers:
            x = layer(x)
        #print(x.shape)
        x = x.transpose(1, 2)
        x = self.last_layer(x[:, -1])
        return x

# 构建模型
tcn_model = TCN(in_channels=10, out_channels=64, kernel_size=3, dropout=0.1, n_blocks=2, n_layers=5)

# 加载数据
data = torch.randn(32, 10, 32000)
model = TCN(10, 64, 3, 0.1, 2, 5)
y = model(data)
print(y.shape)
