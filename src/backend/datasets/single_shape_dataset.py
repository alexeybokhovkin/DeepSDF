from torch.utils.data import Dataset


class SDFItemDataset(Dataset):
    def __init__(self, point_cloud, sdf):
        '''
        point_cloud: xyz numpy array of shape (n_points, 3)
        sdf: sdf values of shape (n_points, 1)
        '''
        self.point_cloud = point_cloud
        self.sdf = sdf

    def __getitem__(self, index):
        return self.point_cloud[index], self.sdf[index]

    def __len__(self):
        return self.point_cloud.shape[0]