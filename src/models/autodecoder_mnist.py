import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class AD(nn.Module):
    def __init__(self, image_size=784, z_dim=latent_size, data_shape=60000):
        super(AD, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True), nn.Linear(512, 28 * 28), nn.Tanh())
        
        self.latent_vectors = nn.Parameter(torch.FloatTensor(data_shape, z_dim))
        
        init.xavier_normal(self.latent_vectors)
    
    def forward(self, ind):
        x = self.latent_vectors[ind]
        return self.decoder(x)
    
    def predict(self, x):
        return self.decoder(x)