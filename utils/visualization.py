import numpy as np
from utils.tools import write_ply


def raw_plyfile():
    pointcloud = np.load('data/stanford_indoor3d_ins.sem/Area_1_conferenceRoom_1.npy')
    color = (pointcloud[:, 3:6] * 255).astype(np.uint8)
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']
    write_ply('raw.ply', [pointcloud[:, 0:3], color], field_names)


def sem_plyfile():
    pointcloud = np.load('../data/stanford_indoor3d_ins.sem/Area_1_conferenceRoom_1.npy')
    semvalue = np.unique(pointcloud[:,6])
    color = np.random.randint(255, size=(len(semvalue),3), dtype=np.uint8)
    colors = np.empty((pointcloud.shape[0],3),dtype=np.uint8)
    for num,i in enumerate(semvalue):
        colors[pointcloud[:,6] == i] = color[num]
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']
    write_ply('sem.ply', [pointcloud[:, 0:3], colors], field_names)


def ins_plyfile():
    pointcloud = np.load('../data/stanford_indoor3d_ins.sem/Area_1_conferenceRoom_1.npy')
    insvalue = np.unique(pointcloud[:,7])
    color = np.random.randint(255, size=(len(insvalue),3), dtype=np.uint8)
    colors = np.empty((pointcloud.shape[0],3),dtype=np.uint8)
    for num,i in enumerate(insvalue):
        colors[pointcloud[:,7] == i] = color[num]
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']
    write_ply('ins.ply', [pointcloud[:, 0:3], colors], field_names)


def main():
    ins_plyfile()


if __name__ == '__main__':
    main()