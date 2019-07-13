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
flag = True

# Iterate through directories and files of dataset
for dir in [x for x in os.listdir('.') if os.path.isdir(os.path.join('.', x))]:
        # For each file in the subdirectories of the data set
        for file in os.listdir(dir):
                # Fetch dimensions of image
                if fnmatch.fnmatch(file, '*.mha') and not fnmatch.fnmatch(file, '*label*') and flag:
                        path = root + '/' + dir + '/' + file
                        ending = '/visinf/projects_students/VCLabOccNet/Smiths_LKA_Weapons/ctix-lka-20190503//12-45-550-12/BAGGAGE_20180913_102913_126581.mha'
                        data, header = load(path)
                        # Rename, if too big and not flagged already
                        if path == ending or path == (ending[:-4]+'_big'+ ending[-4:]):
                                os.rename(path, path[:-8]+'_big'+ path[-4:])
                                flag = False
                                print('interrupt please')
                        if data.shape[2] > 1000 and not fnmatch.fnmatch(file, '*_big*') and flag:
                                os.rename(path, path[:-8]+'_big'+ path[-4:])
                                print('too big: ',path)

end = time.time()
print('runtime:',end - start)
