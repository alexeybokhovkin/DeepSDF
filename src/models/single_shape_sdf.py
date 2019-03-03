import torch
import torch.nn.functional as F
from torch import nn



class SDFNet(nn.Module):
    def __init__(self, inner_dim=512, num_layers=8, dropout_rate=0.3):
        super(SDFNet, self).__init__()
        self.layers = nn.ModuleList()
        
        input_dim = 3
        
        for i in range(num_layers-1):
            self.layers.append(nn.Linear(input_dim, inner_dim))
            input_dim = inner_dim
        self.output = nn.Linear(inner_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x), inplace=True)
            x = self.dropout(x)
        
        x = self.tanh(self.output(x))
        return x 