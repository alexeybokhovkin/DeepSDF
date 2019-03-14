import torch
import torch.nn as nn
import torch.nn.functional as F


class AD_SDF(nn.Module):
    def __init__(self, image_size=784, z_dim=256, data_shape=200):
        super(AD, self).__init__()
        self.decoder_stage1 = nn.Sequential(
            nn.Linear(z_dim+3, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 253))
        
        self.decoder_stage2 == nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Tanh())
        
        self.latent_vectors = nn.Parameter(torch.FloatTensor(data_shape, z_dim))
        
        init.xavier_normal(self.latent_vectors)
    
    def forward(self, ind, x):
        code = self.latent_vectors[ind]
        data = torch.cat((code, data), dim=1)
        decoder_stage1_out = self.decoder_stage1(data)
        data = torch.cat((decoder1_out, data), dim=1)
        decoder_stage2_out = self.decoder_stage2(data)
        return decoder_stage2_out