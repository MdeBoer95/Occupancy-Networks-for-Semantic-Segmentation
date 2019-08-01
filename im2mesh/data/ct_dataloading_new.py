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
        for sub_dir in self.sub_dirs:
            sub_dir_files = os.listdir(os.path.join(self.root_dir, sub_dir))
            #Only for testing: remove this 'if' later
            if(len(allfiles) > 200):
                break
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
        image = load(self.allfiles[idx][0])[0].astype('float32')  # only take the image data, not the header
        image_shape = image.shape

        mha_labels = []
        for labelpath in self.allfiles[idx][1]:
            mha_labels.append(load(os.path.join(labelpath)))
        # To change: offset of labels
        # labels = self._load_label_masks(mha_labels, image_shape)

        transformations = [ct_transforms.ReplicateBorderPadding3D((640, 448, 512)),
                           ct_transforms.NaiveRescale((32, 32, 32))]
        if image_shape[2] > 512:
            # crop center 512 of z dim
            transformations.insert(0, ct_transforms.CropZCenter3D(512))
        # compose all transformations to one
        image_transform = torchvision.transforms.Compose(transformations)
        # Change here
        # label_transform = torchvision.transforms.Compose(transformations[0:-1])  # do not downscale the labels

        image = image_transform(image)
        labels = determine_bounding_boxes(image_shape, mha_labels)

        points, points_occ = self._sample_points_inside_boundingboxes(labels, 1024)


        sample = {'points': points.astype('float32'), 'points.occ': points_occ.astype('float32'), 'inputs': image}
        return sample

    # Determine bounding boxes
    def determine_bounding_boxes(image_shape, label_list):
        # Copy labels
        labels = label_list.copy()
        # Change offsets and label boxes, if necessary

        if shape[2]> 512:
            # Image will be cropped from center
            image_z_center = shape[2] // 2
            # Z-offset changes by the following amount
            image_z_offset = (shape[2] - 512) // 2
            
            # List for labels that will be removed
            remove = []
            for label in labels:
                # Divide by voxelspacing to get index
                voxel_spacing = label[1].voxel_spacing[2]
                z_offset = int(round(label[1].offset[2] / voxel_spacing))
                z_label_size = label[0].shape[2]
                z_begin = image_z_center - 256
                z_end = image_z_center + 256
                # First case: label is completely inside the cropped image
                if (z_offset > z_begin and (z_offset + z_label_size) < z_end):
                    # Offset needs to change:
                    label[1].offset[2] = (z_offset + image_z_offset) * voxel_spacing
                # Remove all other labels
                else:                    
                    remove.append(label)
                
            # Remove labels from list
            if len(remove) > 0:
                labels = [x for x in labels if x not in remove]
        else:
            # Image will be padded from both sides
            change = (512 - shape[2]) // 2
            # Add change to offsets
            for label in labels:
                voxel_spacing = label[1].voxel_spacing[2]
                label[1].offset[2] = ((label[1].offset[2] / voxel_spacing) + change) * voxel_spacing

        return labels 
   

    # label_masks
    def _sample_points_inside_boundingboxes(self, labels, num_points):
        """
        Sample a given number of points from the bounding box of an object.
        :param labels: list of labels with new offset for the objects inside the ct image
        :param num_points: the number of points to draw in total
        :return: a list of x,y,z coordinate pairs with length num_points
        """

        num_labels = len(labels)
        points_per_label = num_points//num_labels  # points per label
        rest = num_points - points_per_label * num_labels  # if not possible to distribute equally, draw the remaining
                                                           # ones from the first bounding box
        points_per_label = [points_per_label for _ in range(num_labels)]

        points_per_label[0] += rest

        def sample_points(num_points, limit_tuple):
            """
            sample random real values in the ranges given by the 'limit-tuple'
            :param num_points: number of point to sample
            :param limit_tuple: 6 - tuple of limits for all dimensions (x_min, x_max, y_min, y_max, z_min, z_max)
            :return:
            """
            x_koords = np.round(np.random.uniform(low=limit_tuple[0], high=limit_tuple[1], size=(num_points, 1)), 3)
            y_koords = np.round(np.random.uniform(low=limit_tuple[2], high=limit_tuple[3], size=(num_points, 1)), 3)
            z_koords = np.round(np.random.uniform(low=limit_tuple[4], high=limit_tuple[5], size=(num_points, 1)), 3)
            points_xyz = np.hstack((x_koords, y_koords, z_koords))
            return points_xyz

        def lookup_occ(label, points):
            """
            look up occupancy value at nearest neighbour for each given point
            :return: occupancy values for points
            """
            nearest_points = np.round(points).astype(int)
            # we need
            # Label offset:
            x_offset = label[1].offset[0] 
            y_offset = label[1].offset[1] 
            z_offset = int(round(label[1].offset[2] / voxel_spacing))  
            occ_val = label[nearest_points[:,0], nearest_points[:, 1], nearest_points[:,2]]
            return occ_val
        # change
        def recover_boundingbox(label):
            """
            Get dimensions of bounding box from label
            :return: 6 tuple with start and end of the bounding box in each dimension
            """
            shape = label.shape
            x_offset = label[1].offset[0] 
            y_offset = label[1].offset[1] 
            z_offset = int(round(label[1].offset[2] / voxel_spacing)) 

            x_idxs = [x for x in range(shape[0]) if np.sum(label[x, :, :]) != 0]
            x_low, x_high = min(x_idxs), max(x_idxs) + 1
            y_idxs = [y for y in range(shape[1]) if np.sum(label[:, y, :]) != 0]
            y_low, y_high = min(y_idxs), max(y_idxs) + 1
            z_idxs = [z for z in range(shape[2]) if np.sum(label[:, :, z]) != 0]
            z_low, z_high = min(z_idxs), max(z_idxs) + 1

            return x_low + x_offset, x_high + x_offset, y_low + y_offset, y_high + y_offset, z_low + z_offset, z_high + z_offset

        # sample points from each bounding box
        all_points = np.array([np.inf, np.inf, np.inf]).reshape(1,3)  # remove dummy entry later
        all_occ = np.array([np.inf]).reshape(1,)  # remove dummy entry later
        # sample points and occ values for each bounding box/ label
        for i, label in enumerate(labels):
            bounding_box_limits = recover_boundingbox(label)
            points = sample_points(points_per_label[i], bounding_box_limits)
            occ = lookup_occ(label, points)
            all_points = np.vstack((all_points, points))
            all_occ = np.append(all_occ, occ)

        all_points = all_points[1:, :]
        all_occ = all_occ[1:]
        return all_points, all_occ


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
