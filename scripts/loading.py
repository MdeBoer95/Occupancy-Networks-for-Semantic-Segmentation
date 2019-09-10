from medpy.io import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import numpy as np
import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
from im2mesh.onet.generation import Generator3D
import numpy as np



data_img = '/visinf/projects_students/VCLabOccNet/Smiths_LKA_Weapons/ctix-lka-20190503/12-54-500-12/BAGGAGE_20180913_160122_126581.mha'
data_img_label = '/visinf/projects_students/VCLabOccNet/Smiths_LKA_Weapons/ctix-lka-20190503/12-54-500-12/BAGGAGE_20180913_160122_126581_label_auto_1.mha'
data_prep_path = '/visinf/projects_students/VCLabOccNet/test/BAGGAGE_20180913_160122_126581.npz'
dataset = ct.CTImagesDataset(data_prep_path)

# Loader for v_dataset
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=10, num_workers=4, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
X1 = []
Y1 = []
Z1 = []

def add_occupied_coords(label):
    label_arr = label[0]
    label_offsets = label[1].offset
    z_spacing = label[1].spacing[2]

    for x in range(label_arr.shape[0]):
        for y in range(label_arr.shape[1]):
            for z in range(label_arr.shape[2]):
                if label_arr[x, y, z] == 1:
                    X1.append(int(label_offsets[0] + x))
                    Y1.append(int(data_arr.shape[1] - (-label_offsets[1] + y)))
                    Z1.append(int(label_offsets[2]/z_spacing + z))


for label in labels:
    add_occupied_coords(label)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(0, 620)
ax.set_ylim3d(0, 420)
ax.set_zlim3d(0, 512)
print(data_arr.shape[2])



# rotate the axes and update
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# rotate the axes and update
for angle in range(0, 360, 20):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
