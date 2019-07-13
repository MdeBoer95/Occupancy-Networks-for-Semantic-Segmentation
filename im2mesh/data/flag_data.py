import os
import fnmatch
from medpy.io import load
import itk
import inspect
import time
import numpy as np

# Script to rename filenames, if their size is too big (3rd dimension > 1000)

start = time.time()
# Descend to dataset
root = '/visinf/projects_students/VCLabOccNet/Smiths_LKA_Weapons/ctix-lka-20190503/'
os.chdir(root)

# Iterate through directories and files of dataset
for dir in [x for x in os.listdir('.') if os.path.isdir(os.path.join('.', x))]:
        # For each file in the subdirectories of the data set
        for file in os.listdir(dir):
                # Fetch dimensions of image
                if fnmatch.fnmatch(file, '*.mha') and not fnmatch.fnmatch(file, '*label*'):
                        path = root + '/' + dir + '/' + file
                        data, header = load(path)
                        # Rename, if too big
                        if data.shape[2] > 1000:
                                #os.rename(path, )
                                print(path)
                                print(path[:-4])
                        break
        break
end = time.time()
print('runtime:',end - start)
