import os
import numpy as np
import torch
from torch.utils.data import Dataset
from indoor3d_util import args

class S3DIS(Dataset):
    def __init__(self, split='train', data_root=args['NEW_DATA_PATH'], test_area='5', fea_dim=6):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.test_area = test_area
        if self.split == 'train':
            self.files = [file for file in os.listdir(self.data_root) if file[5]!=self.test_area]
        else:
            self.files = [file for file in os.listdir(self.data_root) if file[5]==self.test_area]

    def __getitem__(self, item):
        file = np.load(os.path.join(args['NEW_DATA_PATH'], self.files[item]))
        return file[:,:6],file[:,6],file[:,7]

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    s3dis = S3DIS()
    a,b,c = (s3dis[0])