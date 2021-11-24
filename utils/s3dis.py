import os
import numpy as np
import torch
import pickle
import time
from torch.utils.data import Dataset
from utils.indoor3d_util import args
import glob
from utils.tools import DataProcessing


class S3DIS(Dataset):
    def __init__(self, split='train', data_root=args['NEW_DATA_PATH'], test_area='5'):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.test_area = test_area
        self.label_to_names = {0: 'ceiling',
                             1: 'floor',
                             2: 'wall',
                             3: 'beam',
                             4: 'column',
                             5: 'window',
                             6: 'door',
                             7: 'table',
                             8: 'chair',
                             9: 'sofa',
                             10: 'bookcase',
                             11: 'board',
                             12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.val_split = 'Area_' + str(test_area)
        self.all_files = glob.glob(os.path.join(self.data_root, '*.npy'))
        print(self.all_files)
        self.val_proj = []
        self.val_semlabels = []
        self.val_inslabels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'train': [], 'test': []}
        self.input_colors = {'train': [], 'test': []}
        self.input_semlabels = {'train': [], 'test': []}
        self.input_inslabels = {'train': [], 'test': []}
        self.input_names = {'train': [], 'test': []}
        self.load_sub_sampled_clouds()

    def load_sub_sampled_clouds(self):
        for i, file_path in enumerate(self.all_files): ##~/DataSet/origin_ply/
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'test'
            else:
                cloud_split = 'train'
            # Name of the input files
            kd_tree_file = os.path.join(self.data_root, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_np_file = os.path.join(self.data_root, '{:s}.npy'.format(cloud_name))
            data = np.load(sub_np_file)
            sub_colors = data[:,3:6]
            sub_semlabels = data[:,6]
            sub_inslabels = data[:,7]
            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_semlabels[cloud_split] += [sub_semlabels]
            self.input_inslabels[cloud_split] += [sub_inslabels]
            self.input_names[cloud_split] += [cloud_name]
            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))
        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = os.path.join(self.data_root, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, semlabels, inslabels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_semlabels += [semlabels]
                self.val_inslabels += [inslabels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))


    def __getitem__(self, item):
        self.possibility[self.split] = []
        self.min_possibility[self.split] = []
        # Random initialize
        for i, tree in enumerate(self.input_colors[self.split]):
            self.possibility[self.split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[self.split] += [float(np.min(self.possibility[self.split][-1]))]

        cloud_idx = int(np.argmin(self.min_possibility[self.split]))
        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.argmin(self.possibility[self.split][cloud_idx])
        print(cloud_idx,point_ind)
        # Get all points within the cloud from tree structure
        points = np.array(self.input_trees[self.split][cloud_idx].data, copy=False)
        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)
        # Add noise to the center point
        noise = np.random.normal(scale=args['noise_init'] / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)
        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < args['num_points']:
            # Query all points within the cloud
            queried_idx = self.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.input_trees[self.split][cloud_idx].query(pick_point, k=args['num_points'])[1][0]

        # Shuffle index
        queried_idx = DataProcessing.shuffle_idx(queried_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_colors = self.input_colors[self.split][cloud_idx][queried_idx]
        queried_pc_semlabels = self.input_semlabels[self.split][cloud_idx][queried_idx]
        queried_pc_inslabels = self.input_inslabels[self.split][cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[self.split][cloud_idx][queried_idx] += delta
        self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

        # up_sampled with replacement
        if len(points) < args['num_points']:
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_semlabels, queried_pc_inslabels = \
                DataProcessing.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_semlabels, queried_pc_inslabels, queried_idx, args['num_points'])

        return queried_pc_xyz.astype(np.float32),queried_pc_colors.astype(np.float32),queried_pc_semlabels,queried_pc_inslabels

    def __len__(self):
        return args['train_steps'] * args['batch_size'] if self.split == 'train' \
            else args['val_steps'] * args['val_batch_size']


if __name__ == '__main__':
    s3dis = S3DIS()
    for i in range(len(s3dis)):
        a,b,c,d = s3dis[i]
        print(c,d)
        print(len(c),len(d))