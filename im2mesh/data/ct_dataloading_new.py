import os
import itertools
import random

import numpy as np
from medpy.io import load
from torch.utils.data import Dataset, DataLoader
import torchvision
import im2mesh.data.ct_transforms as ct_transforms
import time
import warnings

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
        image = image_transform(image)
        #
        labels, discarded = self.determine_offsets(image_shape, mha_labels, 512)
        points, points_occ = self._sample_points_inside_boundingboxes(labels, 1024)


        sample = {'points': points.astype('float32'), 'points.occ': points_occ.astype('float32'), 'inputs': image}
        return sample
	


    # Determine bounding boxes
    def determine_offsets(self, shape, label_list, z):
        """
        Determine the offset for each label so we can use these for indexing.
        The voxel spacing will not be taken into account for the new offset, but can be applied again via the label!
        Remove labels from list, if they are partially or completely out of the new cropped image
        :param image_shape: the shape of the image (620, 420, z), determines if image is being cropped or padded
        :param label_list: A list of labels for the corresponding image
        :param z: Number that indicates
        :return: list of tuples with tuples being (new offset, label), number of discarded labels
        """
        offsets_and_labels = []
        discarded = 0
        # x and y are fixed to 620, 420. Remember y, because offset y is inverted
        y_max = 420

        # Padding changes from (620, 420, z) to (640, 448, 512):
        x_pad = 10
        y_pad = 14

        # If z-dim exceeds z, image will be cropped
        if shape[2]> z:
            # Calculate the borders of the z_dim
            z_diff = (shape[2] - z) // 2
            begin = z_diff
            end = z_diff + z

            # List for labels that will be removed
            # remove = []
            for label in label_list:

                # Voxelspacing for z_dim
                voxel_spacing = label[1].get_voxel_spacing()[2]
                z_offset = label[1].offset[2] / voxel_spacing
		        # Implement warning
                if (z_offset) % 1 > 0:
                    warnings.warn("Voxel spacing is not correct, off by: "+ str(z_offset % 1))
                z_offset = int(round(z_offset))

                #If label is completely inside the cropped image
                if z_offset > begin and (z_offset + label[0].shape[2]) < end:
                    # Offset change:
                    # Y is inverted !!!
                    offset = [label[1].offset[0] + x_pad, label[1].offset[1] - y_pad, z_offset + z_diff]
                    offsets_and_labels.append((offset, label))
                # Else: label will be not be appended, counter for discarded labels increases
                else:
                    discarded = discarded + 1
        # z will be padded from both sides
        else:
            # Calculate padding for z
            z_pad = (z - shape[2]) // 2
            # Add change to offsets
            for label in label_list:
                # Voxelspacing for z_dim
                voxel_spacing = label[1].get_voxel_spacing()[2]
                # Offset changes:
                z_offset = int(round(label[1].offset[2] / voxel_spacing))
                # Y is inverted !!!
                offset = [label[1].offset[0] + x_pad, label[1].offset[1] - y_pad, z_offset + z_pad]
                offsets_and_labels.append((offset, label))

        return offsets_and_labels, discarded


    # label_masks
    def _sample_points_inside_boundingboxes(self, labels, num_points):
        """
        Sample a given number of points from the bounding box of an object.
        :param labels: list of tuples (new offset, label) for labels inside the ct image
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
            x_coords = np.round(np.random.uniform(low=limit_tuple[0], high=limit_tuple[1], size=(num_points, 1)), 3)
            y_coords = np.round(np.random.uniform(low=limit_tuple[2], high=limit_tuple[3], size=(num_points, 1)), 3)
            z_coords = np.round(np.random.uniform(low=limit_tuple[4], high=limit_tuple[5], size=(num_points, 1)), 3)
            points_xyz = np.hstack((x_coords, y_coords, z_coords))
            return points_xyz

        def lookup_occ(label, points):
            """
            look up occupancy value at nearest neighbour for each given point
            :param label: tuple of (new offset, label)
            :param points: list of points to look up inside the label
            :return: occupancy values for points
            """

	        # Y is inverted !!!
            # Label offset:
            offset = label[0]
            print("Offset: ", offset)
            # Label shape
            shape = label[1][0].shape
            print("Shape: ", shape)
            # Array with one offset for each point in points
            offset_array = np.empty(points.shape)
            offset_array[:] = np.array(offset)
            # Subtract y_shape from y_offset
            offset_array[:, 1] = offset_array[:, 1] - shape[1]
            # List of nearest points, subtract offset from points to lookup point in label
            nearest_points = np.round(points).astype(int)
            # Get label indices for each given point
            nearest_points = np.round(nearest_points - offset_array).astype(int)
            print(nearest_points)
            # Look up occupancy values of points
            return label[1][0][nearest_points[:,0], nearest_points[:, 1], nearest_points[:,2]]

        # change
        def bounding_box_limit(label):
            """
            Get limits of bounding box of the label inside the image
            :param label: tuple of (new offset, label)
            :return: 6 tuple with start and end of the bounding box in each dimension
            """
            shape = label[1][0].shape
            offset = label[0]
            x_low = offset[0]
            x_high = offset[0] + shape[0]
            # Y is inverted !!!
            y_low = offset[1]
            y_high = offset[1] - shape[1]
            z_low = offset[2]
            z_high = offset[2]+ shape[2]

            return (x_low, x_high, y_low, y_high, z_low, z_high)

        # sample points from each bounding box
        all_points = np.array([np.inf, np.inf, np.inf]).reshape(1,3)  # remove dummy entry later
        all_occ = np.array([np.inf]).reshape(1,)  # remove dummy entry later
        # sample points and occ values for each bounding box/ label
        for i, label in enumerate(labels):
            limit = bounding_box_limit(label)
            points = sample_points(points_per_label[i], limit)
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
