import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from torch.utils.data.dataset import Dataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
image_size = 784
h_dim = 400
num_epochs = 15
batch_size = 128
learning_rate = 1e-4
latent_size = 2

class DatasetMNIST(Dataset):

    def __init__(self, root_dir, latent_size, transform=None):
        self.data = torchvision.datasets.MNIST(root=root_dir,
                                             train=True,
                                             download=True).train_data.float()/256.0
        self.image_size = self.data.shape[1] * self.data.shape[2]
        self.transform = transform
        rnd = np.random.randint(low=0, high=256, size=(self.data.shape[0], latent_size), dtype='uint8').astype(np.float32)
        self.latent_vectors = torch.tensor(rnd, requires_grad=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].reshape((self.image_size, 1))
        latent = self.latent_vectors[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, latent

dataset = DatasetMNIST(root_dir='../../data', latent_size=latent_size)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)


# AD model
class AD(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=latent_size):
        super(AD, self).__init__()
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
    
    def forward(self, x):
        h = F.relu(self.fc4(x))
        return torch.sigmoid(self.fc5(h))

model = AD().to(device)

optimizer = torch.optim.Adam([dataset.latent_vectors,] + list(model.parameters()), lr=learning_rate)


# Start training
for epoch in range(num_epochs):
    for i, (x, latent) in enumerate(data_loader):
        # Forward pass
        latent = latent.to(device).view(-1, latent_size)
        x_reconst = model(latent).unsqueeze(2)

        loss = F.binary_cross_entropy(x_reconst, x, size_average=False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, len(data_loader), loss.item()))
    
    with torch.no_grad():
        # Visualize 2D latent space

        step = 1.0
        steps = 20
        size = 28

        out = torch.zeros(size=(steps * size, steps * size))

        for l1 in range(0, steps):
            for l2 in range(0, steps):
                x = torch.tensor([l1 * step, l2 * step])
                out_ = model(x)
                out[l1 * size:(l1 + 1) * size, l2 * size:(l2 + 1) * size] = out_.view(size, size)
        save_image(out, os.path.join(sample_dir, 'latent_space-{}.png'.format(epoch + 1)))