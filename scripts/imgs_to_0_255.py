import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread, imshow, imsave


"""

    python imgs_to_0_255.py --in_path /mnt/c/Users/haddo/DL_stack/Halimeda/inference_test/weighted_merge/ \
                        --shape 1024 \
                        --sp /mnt/c/Users/haddo/DL_stack/Halimeda/inference_test/weighted_merge/

"""

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', help='Path to the run folder', type=str)
parser.add_argument('--sp', help='save_path', type=str)
parser.add_argument('--shape', help='img_shape', type=int)
parsed_args = parser.parse_args()

in_path = parsed_args.in_path
sp = parsed_args.sp
shape = parsed_args.shape

IMG_WIDTH = shape
IMG_HEIGHT = shape

grey_list = sorted(os.listdir(in_path))

print("grey_list",grey_list)

problematic_files=[]

for n, id_ in enumerate(grey_list):
    path = os.path.join(in_path, id_)
    img = imread(path, as_gray = True)
    # img=cv2.imread(path,2)
    # print("img is:",img.flatten())
    # print("set is:",set(img.flatten()))
    if 255 not in set(img.flatten()):
        print("Something might be wrong with img ",id_)
        print("set is:",set(img.flatten()))
        print("-------------------------")
        print("np unique is:",np.unique(img.flatten()))
        print("BE CAREFUUUUUL")
        # img_new = img*255
        # imsave(os.path.join(sp, id_), img_new)
        problematic_files.append(id_)
    else:
        print("this one should be ok!!")
        print("set is:",set(img.flatten()))


print("Problematic_files are",problematic_files)
print(" the number of Problematic_files is",len(problematic_files))

    # img_new = img*255
    # imsave(os.path.join(sp, id_), img_new)
