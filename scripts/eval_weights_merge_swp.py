

import os
import cv2
import sys
import copy
import imageio
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import trapz
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm



# evaluating_instruction="python /mnt/c/Users/haddo/DL_stack/SS_Halimeda/scripts/evaluation.py --run_path {} \
#                     --mask_path  /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/gt_test/all/ \
#                     --shape 1024"

evaluating_instruction="python /mnt/c/Users/haddo/DL_stack/object_detection_utils/metrics/eval_ss2.py \
                    --pred_path /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_DATASET/INFERENCE_best_models/w_merge_swp/{}/ \
                    --gt_im_path  /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_DATASET/INFERENCE_best_models/gt_all_val/  \
                    --run_name w_merge_{} --save_path /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_DATASET/INFERENCE_best_models/w_merge_swp/{}/ --shape 1024"

input_dir="/mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_DATASET/INFERENCE_best_models/w_merge_swp/"

folders=[0.75,0.8,0.85,0.9,0.95,1]

for folder in folders:
    inference_folder=os.path.join(input_dir,str(folder))
    print("Evaluating folder: ",inference_folder)
    I=evaluating_instruction.format(str(folder),str(folder),str(folder))
    print(I)
    os.system(I)