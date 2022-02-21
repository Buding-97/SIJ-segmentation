"""
    Modified from: https://github.com/charlesq34/pointnet/blob/master/sem_seg/indoor3d_util.py
"""
import numpy as np
import glob
import os
import yaml
from utils.tools import DataProcessing
from sklearn.neighbors import KDTree
import pickle


# g_class2label ---> {'ceiling': 0, 'floor': 1, 'wall': 2, 'beam': 3, 'column': 4, 'window': 5, 'door': 6, 'chair': 7,
# 'table': 8, 'bookcase': 9, 'sofa': 10, 'board': 11, 'clutter': 12}
with open('../comfig.yaml','r',encoding='utf-8') as f:
    args = yaml.load(f,Loader=yaml.FullLoader)
g_classes = [x.rstrip() for x in open(os.path.join('../meta/s3dis_names.txt'))]
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
    dict_ins2sem = {}
    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in g_classes:  # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f)
        labels = np.ones((points.shape[0], 1)) * g_class2label[cls]
        instancelabels = np.ones((points.shape[0], 1)) * instanceid
        dict_ins2sem[instanceid] = g_class2label[cls]
        instanceid += 1
        points_list.append(np.concatenate([points, labels, instancelabels], 1)) #point*3 color*3 label instancelabel = 8
    data_label = np.concatenate(points_list, 0)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min
    sub_xyz,sub_colors,sub_inslabels = DataProcessing.grid_sub_sampling(points=data_label[:,0:3].astype(np.float32),
                                                                     features=data_label[:,3:6].astype(np.uint8),
                                                                     labels=data_label[:,-1].astype(np.int32),
                                                                     grid_size=0.04)
    sub_colors = sub_colors / 255.0
    sub_semlabels = np.array([dict_ins2sem[i] for i in np.squeeze(sub_inslabels)],dtype=np.int32)


    if file_format == 'txt':
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %f %f %f %d %d\n' % \
                       (sub_xyz[i, 0], sub_xyz[i, 1], sub_xyz[i, 2],
                        sub_colors[i, 0], sub_colors[i, 0], sub_colors[i, 0],
                        sub_semlabels[i], sub_inslabels[i]))
        fout.close()
    elif file_format == 'numpy':
        np.save(out_filename + '.npy', np.concatenate((sub_xyz,sub_colors,sub_semlabels[:,np.newaxis],sub_inslabels),axis=1))
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % file_format)
        exit()

    search_tree = KDTree(sub_xyz)
    with open(out_filename + '_KDTree.pkl', 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(data_label[:,0:3].astype(np.float32), return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    with open(out_filename + '_proj.pkl', 'wb') as f:
        pickle.dump([proj_idx, data_label[:,-2].astype(np.int32), data_label[:,-1].astype(np.int32)], f)