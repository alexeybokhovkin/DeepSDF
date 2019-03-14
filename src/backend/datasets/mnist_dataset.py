from torch.utils.data.dataset import Dataset
import torchvision

class DatasetMNIST(Dataset):

    def __init__(self, root_dir, latent_size, transform=None):
        mnist = torchvision.datasets.MNIST(root=root_dir, train=True, download=True)
        self.data = mnist.train_data.float()/255.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]

        return image.flatten(), index