import os

import numpy as np
from medpy.io import load
from torch.utils.data import Dataset, DataLoader
import torchvision
from im2mesh.data import ct_transforms

LABEL_SUFFIX = "_label_"  # followed by a number and the file format
MHA_FORMAT = ".mha"


class CTImagesDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with the subdirectories for each image.
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
        label = load_label_mask(mha_labels, image_shape)

        transformations = [ct_transforms.ReplicateBorderPadding3D((image_shape[0], 448, image_shape[2])),
                           ct_transforms.ToTensor(),
                           ct_transforms.Rescale((32, 32, 32))]
        # if z dim is too small, pad to 512
        if image_shape[2] <= 512:
            transformations.insert(0, ct_transforms.ReplicateBorderPadding3D((image_shape[0], image_shape[1], 512)))
        # otherwise crop center 512 of z dim
        else:
            transformations.insert(0, ct_transforms.CropZCenter3D(512))

        # compose all transformations to one
        composed_transform = torchvision.transforms.Compose(transformations)
        image = composed_transform(image)
        label = composed_transform(label)

        sample = {'image': image, 'label': label}
        return sample


def load_label_mask(mha_label_list, mask_size):
    merged_masks = np.zeros(mask_size, dtype=np.int)
    for mha_label in mha_label_list:
        label_box = mha_label[0]
        label_offsets = mha_label[1].offset
        label_mask = np.zeros(mask_size, dtype=np.int)
        # expand the labels to the full size (same size as the image)
        label_mask[int(label_offsets[0]):int(label_offsets[0] + label_box.shape[0]),
        int(-label_offsets[1]):int(-label_offsets[1] + label_box.shape[1]),
        int(label_offsets[2]):int(label_offsets[2] + label_box.shape[2])] = label_box

        merged_masks = np.bitwise_or(merged_masks, label_mask)
    return merged_masks


if __name__ == '__main__':
    data = CTImagesDataset("/visinf/project_students/VCLabOccNet/Smiths_LKA_Weapons/ctix-lka-20190503/")
    counter = 0
    for datax in data:
        counter += 1
        print(counter)
    print("end")
