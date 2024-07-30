

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


evaluating_instruction = "python /home/slimbook/Escritorio/Halimeda/semantic_segmentation/SS_Halimeda/scripts/evaluation.py \
                    --run_name eval_weights --pred_path /home/slimbook/Escritorio/Halimeda/merge/w_merge_yolov8/{}/ \
                    --out_path /home/slimbook/Escritorio/Halimeda/merge/w_merge_yolov8/{}/ --gt_path /home/slimbook/Escritorio/Halimeda/dataset/all_val/gt --shape 1024"

input_dir = "/home/slimbook/Escritorio/Halimeda/merge/w_merge_yolov8/"

folders = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

for folder in folders:
    inference_folder = os.path.join(input_dir, str(folder))
    print("Evaluating folder: ", inference_folder)
    I = evaluating_instruction.format(str(folder), str(folder))
    print(I)
    os.system(I)
