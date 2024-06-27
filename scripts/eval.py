import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from natsort import natsorted

'''
CALL:
python3 eval.py --run_name 000033 --path_pred ../runs/1/000033/inference --path_out ../runs/1/000033/ --path_gt ../data/test/mask --name xxx
'''
def conditional_div(a, b):
    return a / b if b else 0

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', help='Path to the run folder', type=str)
parser.add_argument('--path_pred', help='Path to the pred folder', type=str)
parser.add_argument('--path_out', help='Path to the out folder', type=str)
parser.add_argument('--path_gt', help='Path to the gt folder', type=str)
parser.add_argument('--name', help='eval name', type=str)
parsed_args = parser.parse_args()

run_name = parsed_args.run_name
path_out = parsed_args.path_out
path_pred = parsed_args.path_pred
path_gt = parsed_args.path_gt
name = parsed_args.name

pred_list = natsorted(os.listdir(path_pred))
gt_list = natsorted(os.listdir(path_gt))

confMatrix=np.zeros((2,2))

precision_list = list()
recall_list = list()
fallout_list = list()
f1_list =  list()
accuracy_list =  list()

preds = list()
gts = list()

print("evaluating: " + path_out)

for i in range(len(pred_list)):

    pred_file = os.path.join(path_pred, pred_list[i])
    gt_file = os.path.join(path_gt, gt_list[i])

    pred = cv2.imread(pred_file,2)
    gt = cv2.imread(gt_file,2)

    preds.append(pred)
    gts.append(gt)

preds_flat = list()
gts_flat = list()


for i in range(len(pred_list)): 

    pred = preds[i]
    gt = gts[i]

    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    preds_flat.append(pred_flat)
    gts_flat.append(gt_flat)

preds_flat_concat = np.hstack(preds_flat)
gts_flat_concat = np.hstack(gts_flat)

preds_flat_concat = preds_flat_concat/255
gts_flat_concat = gts_flat_concat/255

gts_flat_concat = np.where(gts_flat_concat>0.5, 1, 0)

for thr in tqdm(range(255)):

    thr = thr/255

    preds_flat_concat_bw =  np.where(preds_flat_concat>thr, 1, 0)

    TN, FP, FN, TP = metrics.confusion_matrix(gts_flat_concat, preds_flat_concat_bw).ravel()

    precision = conditional_div(TP, FP + TP)
    recall = conditional_div(TP, TP + FN)
    fallout = conditional_div(FP, FP + TN)
    accuracy = conditional_div(TP + TN, TP + FP + TN + FN)
    f1score = 2 * conditional_div(recall * precision, recall + precision)

    precision_list.append(precision)
    recall_list.append(recall)
    fallout_list.append(fallout)
    accuracy_list.append(accuracy)
    f1_list.append(f1score)

thr_best = np.nanargmax(f1_list)

prec_best = precision_list[thr_best]
rec_best = recall_list[thr_best]
fallout_best = fallout_list[thr_best]
acc_best = accuracy_list[thr_best]
f1_best = f1_list[thr_best]

save_path = os.path.join(path_out, "metrics_" + name)

try:
    os.mkdir(save_path)
except:
    print("")


data = {'Run': [run_name], 'thr': [thr_best], 'acc': [acc_best], 'prec': [prec_best], 'rec': [rec_best], 'fall': [fallout_best], 'f1': [f1_best]} #, 'auc': [roc_auc]}

df = pd.DataFrame(data)
print(df)

df.to_excel(os.path.join(save_path,'metrics.xlsx'))


