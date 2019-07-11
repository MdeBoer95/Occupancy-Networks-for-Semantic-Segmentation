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
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Name of directory with the subdirectories for each image.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        # Only get name, if directory
        self.sub_dirs = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
        # store the path for each image and it's labels in a list [ [imagepath, [labelpath1, labelpath2, ...]] ]
        allfiles = []

        for sub_dir in self.sub_dirs and len(allfiles) < 21:
            sub_dir_files = os.listdir(os.path.join(self.root_dir, sub_dir))
            for filename in sub_dir_files:
                if filename.endswith(MHA_FORMAT) and LABEL_SUFFIX not in filename:
                    # Image paths
                    image_filepath = os.path.join(self.root_dir, sub_dir, filename)
                    # Label paths
                    label_filepaths = [os.path.join(self.root_dir, sub_dir, labelname)
                                        for labelname in sub_dir_files
                                            if LABEL_SUFFIX in labelname and labelname.endswith(MHA_FORMAT)
                                                and filename[0:-4] in labelname]
                    # Append paths from found images with corresponding labels
                    allfiles.append([image_filepath, label_filepaths])

        self.allfiles = allfiles


    def __len__(self):
        return len(self.allfiles)

    def __getitem__(self, idx):
        image = load(self.allfiles[idx][0])[0]  # only take the image data, not the header
        image_shape = image.shape

        mha_labels = []
        for labelpath in self.allfiles[idx][1]:
            mha_labels.append(load(os.path.join(labelpath)))
        labels = self._load_label_masks(mha_labels, image_shape)

        transformations = [ct_transforms.ReplicateBorderPadding3D((640, 448, 512)),
                           ct_transforms.NaiveRescale((32, 32, 32))]
        if image_shape[2] > 512:
            # crop center 512 of z dim
            transformations.insert(0, ct_transforms.CropZCenter3D(512))
        # compose all transformations to one
        composed_transform = torchvision.transforms.Compose(transformations)

        image = composed_transform(image)
        labels = [composed_transform(label) for label in labels]

        points, points_occ = self._sample_points_inside_boundingboxes(labels, 1024)

        sample = {'points': points, 'points.occ': points_occ, 'inputs': image}
        return sample


    def _load_label_masks(self, mha_label_list, mask_size):
        expanded_masks = []
        for mha_label in mha_label_list:
            label_box = mha_label[0]
            label_offsets = mha_label[1].offset
            label_mask = np.zeros(mask_size, dtype=np.int)
            # expand the labels to the full size (same size as the image)
            label_mask[int(label_offsets[0]):int(label_offsets[0] + label_box.shape[0]),
            int(-label_offsets[1]):int(-label_offsets[1] + label_box.shape[1]),
            int(label_offsets[2]):int(label_offsets[2] + label_box.shape[2])] = label_box
            expanded_masks.append(label_mask)
        return expanded_masks


    def _merge_labelmasks(self, labelmasks):
        mask_size = labelmasks[0].shape
        merged_masks = np.zeros(mask_size, dtype=np.int)
        for mask in labelmasks:
            merged_masks = np.bitwise_or(merged_masks, mask)
        return merged_masks


    def _sample_points_inside_boundingboxes(self, label_masks, num_points):
        """
        Sample a given number of points from the bounding box of an object.
        :param label_mask: list of label masks for the objects inside the ct image
        :param num_points: the number of points to draw in total
        :return: a list of x,y,z coordinate pairs with length num_points
        """
        # find x,y,z coords that are inside the bounding box. We look for 'planes' in x,y,z direction that contain
        # not only 0s.
        num_labels = len(label_masks)
        points_per_label = num_points // num_labels  # points per label
        rest = num_points - points_per_label * num_labels  # if not possible to distribute equally, draw the remaining
        # ones from the first bounding box
        points_per_label = [points_per_label for _ in range(num_labels)]

        points_per_label[0] += rest

        for i, label_mask in enumerate(label_masks):
            x_idxs = [x for x in range(label_mask.shape[0]) if np.sum(label_mask[x, :, :]) != 0]
            y_idxs = [y for y in range(label_mask.shape[1]) if np.sum(label_mask[:, y, :]) != 0]
            z_idxs = [z for z in range(label_mask.shape[2]) if np.sum(label_mask[:, :, z]) != 0]

            # gather all possible combinations and sample num_points points
            points_xyz = [x_idxs, y_idxs, z_idxs]
            bounding_box_points = list(itertools.product(*points_xyz))
            num_samples = points_per_label[i]
            if num_samples > len(bounding_box_points):
                num_samples = len(bounding_box_points)
            sampled_points = random.sample(bounding_box_points, num_samples)
            # get label for each point
            labels = []
            for sample in sampled_points:
                labels.append(label_mask[sample])
        return np.array(sampled_points), np.array(labels)


if __name__ == '__main__':
    start = time.time()
    data = CTImagesDataset("/visinf/projects_students/VCLabOccNet/Smiths_LKA_Weapons/ctix-lka-20190503/")
    counter = 0
    for datax in data:
        counter += 1
        print(counter)
        if counter >= 20:
            break
    end = time.time()
    print('Runtime:', end-start)
