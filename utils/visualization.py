import numpy as np
from utils.tools import write_ply


def raw_plyfile():
    pointcloud = np.load('data/stanford_indoor3d_ins.sem/Area_1_conferenceRoom_1.npy')
    color = (pointcloud[:, 3:6] * 255).astype(np.uint8)
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']
    write_ply('w.ply', [pointcloud[:, 0:3], color], field_names)


def sem_plyfile():
    pointcloud = np.load('data/stanford_indoor3d_ins.sem/Area_1_conferenceRoom_1.npy')
    color = (pointcloud[:, 3:6] * 255).astype(np.uint8)
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']
    write_ply('w.ply', [pointcloud[:, 0:3], color], field_names)


def ins_plyfile():
    pointcloud = np.load('data/stanford_indoor3d_ins.sem/Area_1_conferenceRoom_1.npy')
    color = (pointcloud[:, 3:6] * 255).astype(np.uint8)
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']
    write_ply('w.ply', [pointcloud[:, 0:3], color], field_names)


def main():
    raw_plyfile()


if __name__ == '__main__':
    main()