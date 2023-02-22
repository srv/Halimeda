

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



evaluating_instruction="python /mnt/c/Users/haddo/DL_stack/SS_Halimeda/scripts/evaluation.py --run_path {} \
                    --mask_path  /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/gt_test/all/ \
                    --shape 1024"

input_dir="/mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/inference_test/combined_weights_swp"

for folder in os.listdir(input_dir):
    inference_folder=os.path.join(input_dir,folder)
    print("Evaluating folder: ",inference_folder)
    I=evaluating_instruction.format(inference_folder)
    os.system(I)