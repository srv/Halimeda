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

python imgs_to_0_255.py --in_path /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/gt_test/SS/ \
                    --shape 1024 \
                    --sp /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/gt_test/SS/

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

grey = np.zeros((len(grey_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for n, id_ in enumerate(grey_list):
    path = os.path.join(in_path, id_)
    img = imread(path, as_gray = True)
    print("img is:",img.flatten())
    print("set is:",set(img.flatten()))
    img_new = img*255
    imsave(os.path.join(sp, id_), img_new)
