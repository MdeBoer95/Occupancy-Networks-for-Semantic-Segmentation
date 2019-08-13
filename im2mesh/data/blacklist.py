from medpy.io import load
import itertools
import numpy as np
import os
import pickle
import random
import time
import warnings

# Script to blacklist images from database, that have only cut labels, or no labels at all
start = time.time()
print("Started reading files")
blacklist = []

# Get all image and label names
LABEL_SUFFIX = "_label_"  # followed by a number and the file format
MHA_FORMAT = ".mha"
ROOT = "/visinf/projects_students/VCLabOccNet/Smiths_LKA_Weapons/ctix-lka-20190503/"
OUT_FILE = "/visinf/projects_students/VCLabOccNet/Smiths_LKA_Weapons/ctix-lka-20190503/blacklist.pkl"
# out_dir = "out/semseg/onet"
# Only get name, if directory
sub_dirs = [x for x in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, x))]
# store the path for each image and it's labels in a list [ [imagepath, [labelpath1, labelpath2, ...]] ]
all_files = []
for sub_dir in sub_dirs:
    files = os.listdir(os.path.join(ROOT, sub_dir))
    for filename in files:
        if filename.endswith(MHA_FORMAT) and LABEL_SUFFIX not in filename:
            # Image paths
            image_filepath = os.path.join(ROOT, sub_dir, filename)
            # Label paths
            label_filepaths = [os.path.join(ROOT, sub_dir, labelname)
                                for labelname in sub_dir_files
                                    if LABEL_SUFFIX in labelname and labelname.endswith(MHA_FORMAT)
                                        and filename[0:-4] in labelname]
            # Append paths from found images with corresponding labels
            all_files.append([image_filepath, label_filepaths])
print("Read files: ", time.time() - start)
print("Started blacklisting images")

z = 512
# Check labels, if in cropped image
for paths in all_files:
	image = load(paths[0])[0].astype('float32')
    # Check, if image will be cropped:
	labels = []
	useful_labels = []
	if image.shape[2] < 512:
        # Load labels

		for label_path in paths[1]:
            labels.append(load(os.path.join(label_path)))
        # Calculate the borders of the z_dim
        z_diff = (shape[2] - z) // 2
        begin = z_diff
        end = z_diff + z

        for label in labels:
            # Voxelspacing for z_dim
            voxel_spacing = label[1].get_voxel_spacing()[2]
            z_offset = label[1].offset[2] / voxel_spacing
		    # Warning
            if (z_offset) % 1 > 0:
                print("Voxel spacing is not correct, off by: ", z_offset % 1)
                z_offset = int(round(z_offset))

            #If label is completely inside the cropped image
            if z_offset > begin and (z_offset + label[0].shape[2]) < end:

				useful_labels.append(label)
		if len(useful_labels) == 0:
			blacklist.append(paths[0])
# Pickle blacklist
outfile = open(OUT_FILE, 'wb')
pickle.dump(blacklist, outfile)
outfile.close()
print("Number of blacklisted images: ", len(blacklist))
for image in blacklist:
	print(image)
print("Blacklisted all images: ", time.time() - start)
