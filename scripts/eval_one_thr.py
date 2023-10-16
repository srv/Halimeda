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
from natsort import natsorted


"""

This code comes from SS_Halimeda -> check if needed

"CALL: python3 evaluation.py --pred_path /mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/hyp_high_lr2_a/inference_test/coverage_pred \
      --gt_path /mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/hyp_high_lr2_a/inference_test/coverage_gt --shape 1024 \
        --run_name yolo_XL_hyp_high_lr2_a --save_path /mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/hyp_high_lr2_a/inference_test "


python eval_one_thr.py --run_name SS_new_SS_test  --pred_path /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_TEST/SS/inference/ \
                    --out_path /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_TEST/SS/  \
                    --gt_path /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_TEST/SS/gt --shape 1024  

                    

python eval_one_thr.py --run_name eval_low9_da_test_thr_81  --pred_path /mnt/c/Users/haddo/yolov5/projects/halimeda/NEW_DATASET/low9_da/2/inference_test/coverage/ \
                                                    --out_path /mnt/c/Users/haddo/yolov5/projects/halimeda/NEW_DATASET/low9_da --thr 81  \
                                                    --gt_path /mnt/c/Users/haddo/yolov5/datasets/halimeda/NEW_DATASET/labels/test_coverage/ --shape 1024  


python eval_one_thr.py --run_name w_merge_test  --pred_path /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_DATASET/INFERENCE_best_models/w_merge_test/ \
                        --out_path /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_DATASET/INFERENCE_best_models/w_merge_test --thr 0  \
                        --gt_path /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_DATASET/INFERENCE_best_models/gt_all_test/ --shape 1024  

python eval_one_thr.py --run_name AUC_merge_test_n_SS_val  --pred_path /mnt/c/Users/haddo/DL_stack/Halimeda/AUC_merge/inference_all_test_n_only_SS/ \
                        --out_path /mnt/c/Users/haddo/DL_stack/Halimeda/AUC_merge/inference_all_test_n_only_SS --thr 127  \
                        --gt_path /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/all_test/gt/ --shape 1024  

                    

python eval_one_thr.py --run_name w_merge_test_0_7  --pred_path /mnt/c/Users/haddo/DL_stack/Halimeda/weighted_merge/best_0.7/ \
                        --out_path /mnt/c/Users/haddo/DL_stack/Halimeda/weighted_merge --thr 50  \
                        --gt_path /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/all_test/gt/ --shape 1024 


 """
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', help='Path to the run folder', type=str)
parser.add_argument('--pred_path', help='Path to the pred folder', type=str)
parser.add_argument('--out_path', help='Path to the out folder', type=str)
parser.add_argument('--gt_path', help='Path to the gt folder', type=str)
parser.add_argument('--shape', help='img_shape', type=int)
parser.add_argument('--thr', default=35, type=int)
parsed_args = parser.parse_args()

run_name = parsed_args.run_name
out_path = parsed_args.out_path
pred_path = parsed_args.pred_path
gt_path = parsed_args.gt_path
shape = parsed_args.shape
thr = parsed_args.thr

# grey_list = sorted(os.listdir(pred_path))

image_extensions = ['.jpg', '.jpeg', '.png']  


grey_list=natsorted( [file for file in os.listdir(pred_path) if os.path.isfile(os.path.join(pred_path,file)) and file.lower().endswith(tuple(image_extensions))])
img = imread(os.path.join(pred_path, grey_list[0]))

grey = np.zeros((len(grey_list), shape, shape), dtype=np.uint8)


for n, id_ in enumerate(grey_list):
    path = os.path.join(pred_path, id_)
    img = imread(path, as_gray = True)
    img = resize(img, (shape, shape), mode='constant', preserve_range=True)
    grey[n] = img

# gt_list = sorted(os.listdir(gt_path))
# gt_list=sorted([file for file in os.listdir(gt_path) if os.path.isfile(os.path.join(gt_path,file))])
gt_list=natsorted([file for file in os.listdir(gt_path) if os.path.isfile(os.path.join(gt_path,file)) and file.lower().endswith(tuple(image_extensions))])

img = imread(os.path.join(pred_path, grey_list[0]))
gt = np.zeros((len(gt_list), shape, shape), dtype=np.uint8)
for n, id_ in enumerate(gt_list):
    path = os.path.join(gt_path, id_)
    img = imread(path,as_gray = True)
    img = resize(img, (shape, shape), mode='constant', preserve_range=True)
    gt[n] = img

grey_flat = grey.flatten()
gt_flat = gt.flatten()
gt_flat = np.where(gt_flat>127, 1, 0)
zeros = np.count_nonzero(gt_flat == 0)
ones = np.count_nonzero(gt_flat == 1)

bw_flat = np.where(grey_flat>thr, 1, 0)
TN, FP, FN, TP = metrics.confusion_matrix(gt_flat,bw_flat).ravel()

recall = TP/(TP+FN)
precision = TP/(TP+FP)
fallout = FP/(FP+TN)
accuracy = (TP+TN)/(TP+FP+FN+TN)
f1 = 2*((precision*recall)/(precision+recall))


try:
    os.mkdir(out_path)
except:
    print("")

data = {'Run': [run_name], 'thr': [thr], 'acc': [accuracy], 'prec': [precision], 'rec': [recall], 'fall': [fallout], 'f1': [f1]}

df = pd.DataFrame(data)
print(df)

df.to_excel(os.path.join(out_path,'metrics.xlsx'))
# df.to_excel(os.path.join(out_path,run_name+'.xlsx'))



 

