import os
import itertools
import random

import numpy as np
from medpy.io import load
from torch.utils.data import Dataset, DataLoader
import torchvision
import im2mesh.data.ct_transforms as ct_transforms
import time
import torch
import operator

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
        def label_to_image_size(boundingbox, size):
            """
            Pad labels to uniform size
            :param boundingbox: A 3D numpy array with values 0 and 1.
            :param size: The label will get padded to that size
            :return: A tuple with (padded label, shape of label before padding)
            """
            shape = boundingbox.shape
            label_torch = torch.from_numpy(boundingbox)
            padding = torch.nn.ConstantPad3d(
                (0, size[2] - shape[2], 0, size[1] - shape[1], 0, size[0] - shape[0]), 0
            )
            pad = padding(label_torch).numpy()
            assert (np.array_equal(boundingbox, pad[:shape[0], :shape[1], :shape[2]]))
            return pad, shape

        def sample_points(list_points, list_occ, sample_number, share_occ):
            """
            Sample points from the file
            :param share_occ: share of points that will be occupied
            :param sample_number: number of points that will be sampled
            :param list_points: list of points
            :param list_occ: list of occupied values
            :return: tuple of sublist of list_points and list_occ
            """
            assert(len(list_occ) == len(list_points))
            result = []
            occ_list = []
            non_occ_list = []
            points_occ_list = list(zip(list_points, list_occ))
            for coordinate, occupancy in points_occ_list:
                if occupancy == 1.0:
                    occ_list.append((coordinate, occupancy))
                elif occupancy == 0.0:
                    non_occ_list.append((coordinate, occupancy))
                else:
                    raise ValueError
            occ_number = int(sample_number * share_occ)
            occ_indices = np.random.randint(len(occ_list), size=occ_number)
            non_occ_number = self.sampled_points - occ_number
            non_occ_indices = np.random.randint(len(non_occ_list), size=non_occ_number)
            for i in occ_indices:
                result.append(occ_list[i])
            for j in non_occ_indices:
                result.append(non_occ_list[j])
            result = list(zip(*result))
            assert(len)
            return list(result[0]), list(result[1])

        npzfile = np.load(os.path.join(self.root_dir, self.ctsamplesfiles[idx]))
        # load image
        inputs = npzfile['inputs']
        # draw a subsample of the points

        points = npzfile['points']
        occ = npzfile['points_occ']
        points, occ = sample_points(points, occ, self.sampled_points, 0.5)
        # load labels only during testing/validation
        label = npzfile['labels'][0]
        label_offset = label[0]
        label_box = label[1]
        padded_label, label_shape = label_to_image_size(label_box, [640, 448, 512])

        sample =\
            {
                'points': np.asarray(points),
                'points.occ': np.asarray(occ), 'inputs': inputs,
                'label_offset': np.asarray(label_offset),
                'padded_label': padded_label,
                'label_shape': np.asarray(label_shape)
            }
        # print("label shape: ", label_shape)
        return sample


if __name__ == '__main__':
    start = time.time()
    data = CTImagesDataset("/visinf/projects_students/VCLabOccNet/test")
    counter = 0
    for datax in data:
        counter += 1
        print(counter)
    end = time.time()
    print('Runtime for', counter, "samples:", end - start)
