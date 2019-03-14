import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class AD_SDF(nn.Module):
    def __init__(self, z_dim=256, data_shape=200):
        super(AD_SDF, self).__init__()
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
        
        self.decoder_stage2 = nn.Sequential(
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
        code = self.latent_vectors[ind].repeat(x.shape[0], 1)
        data = torch.cat((code, x), dim=1)
        decoder_stage1_out = self.decoder_stage1(data)
        data = torch.cat((decoder_stage1_out, data), dim=1)
        decoder_stage2_out = self.decoder_stage2(data)
        return decoder_stage2_out
    
    def codes(self):
        return self.latent_vectors