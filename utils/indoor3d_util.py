"""
    Modified from: https://github.com/charlesq34/pointnet/blob/master/sem_seg/indoor3d_util.py
"""
import h5py
import numpy as np
import glob
import os
import paramiko
import gc
import sys


DATA_PATH = '../Stanford3dDataset_v1.2_Aligned_Version/'
g_classes = [x.rstrip() for x in open(os.path.join('../meta/s3dis_names.txt'))]
# g_class2label --->  {'ceiling': 0, 'floor': 1, 'wall': 2, 'beam': 3, '
# column': 4, 'window': 5, 'door': 6, 'chair': 7,  'table': 8,
# 'bookcase': 9, 'sofa': 10, 'board': 11, 'clutter': 12}
g_class2label = {cls: i for i, cls in enumerate(g_classes)}


def collect_point_label(anno_path, out_filename, file_format='txt'):
    """ Convert original dataset files to data_label file (each line is XYZRGBLG).
        We aggregated all the points from each instance in the room.
    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBLG)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    points_list = []
    instanceid = 0
    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in g_classes:  # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f)
        labels = np.ones((points.shape[0], 1)) * g_class2label[cls]
        instancelabels = np.ones((points.shape[0], 1)) * instanceid
        instanceid += 1
        points_list.append(np.concatenate([points, labels, instancelabels], 1))  # Nx8

    data_label = np.concatenate(points_list, 0)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min

    if file_format == 'txt':
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %d %d %d %d\n' % \
                       (data_label[i, 0], data_label[i, 1], data_label[i, 2],
                        data_label[i, 3], data_label[i, 4], data_label[i, 5],
                        data_label[i, 6]))
        fout.close()
    elif file_format == 'numpy':
        np.save(out_filename, data_label)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % file_format)
        exit()