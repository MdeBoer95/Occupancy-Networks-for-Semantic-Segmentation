import os
import itertools
import random

import numpy as np
from medpy.io import load
from torch.utils.data import Dataset, DataLoader
import torchvision
import im2mesh.data.ct_transforms as ct_transforms
import time

LABEL_SUFFIX = "_label_"  # followed by a number and the file format
MHA_FORMAT = ".mha"


class CTImagesDataset(Dataset):
    def __init__(self, root_dir, sampled_points=1024):
        """
        Args:
            root_dir (string): Name of directory with the preprocessed subdirectories for each image.
            point_samples (int): Number of sampled points for each image
        """
        self.root_dir = root_dir
        self.ctsamplesfiles = [x for x in os.listdir(root_dir)]  # numpy files with preprocessed samples
        self.sampled_points = sampled_points

    def __len__(self):
        return len(self.ctsamplesfiles)

    def __getitem__(self, idx):
        npzfile = np.load(os.path.join(self.root_dir, self.ctsamplesfiles[idx]))

        # draw a subsample of the points
        points = npzfile['points']
        occ = npzfile['points_occ']
        point_indices = np.random.randint(points.shape[0], size=self.sampled_points)
        points = points[point_indices, :]
        occ = occ[point_indices]


        sample = {'points': points, 'points.occ': occ, 'inputs': npzfile['inputs']}#, 'labels': npzfile['labels']}
        return sample


if __name__ == '__main__':
    start = time.time()
    data = CTImagesDataset("/visinf/projects_students/VCLabOccNet/test")
    counter = 0
    for datax in data:
        counter += 1
        print(counter)
    end = time.time()
    print('Runtime for', counter, "samples:", end-start)
