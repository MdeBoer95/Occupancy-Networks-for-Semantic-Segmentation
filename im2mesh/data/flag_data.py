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
os.chdir('/visinf/projects_students/VCLabOccNet/Smiths_LKA_Weapons/ctix-lka-20190503/')

# Iterate through directories and files of dataset
for dir in [x for x in os.listdir('.') if os.path.isdir(os.path.join('.', x))]:
        # For each file in the subdirectories of the data set
        for file in os.listdir(dir):
                # Fetch metadata of image
                if fnmatch.fnmatch(file, '*.mha') and not fnmatch.fnmatch(file, '*label*'):
                        meta = load(file)
                        print(file)
                        break
        break



# Save data to csv file
#np.savetxt("data_info.csv", np.column_stack((x_dim,y_dim,z_dim)), delimiter=",", fmt='%s', header = 'data')
end = time.time()
print('runtime:',end - start)
#print('max x:', max(x_dim))
#print('max y:', max(y_dim))
#print('max z:', max(z_dim))
#print('number of images:', len(x_dim))
print(nr_pic)
