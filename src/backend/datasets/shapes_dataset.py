from torch.utils.data.dataset import Dataset
import torchvision
import numpy as np

class DatasetShapes(Dataset):
    
    def __init__(self, csv_file, index, transform=None):
        '''
        csv_file: path to .csv with dataset specification
        index: index of shape to process
        '''
        self.samples = pd.read_csv(csv_file)
        self.index = index 
        self.point_cloud = np.load(self.samples.iloc[index, 0])
        self.sdf = np.load(self.samples.iloc[index, 1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.point_cloud[index], self.sdf[index]