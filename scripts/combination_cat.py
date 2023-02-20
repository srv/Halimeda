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
from scipy.integrate import simpson
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument('--path_od', help='path to the od output folder.', type=str)  
parser.add_argument('--path_ss', help='path to the ss output folder.', type=str)
parser.add_argument('--path_ss_gt', help='path to the ss gt folder.', type=str)
parser.add_argument('--path_out', help='path to the ss gt folder.', type=str)
parser.add_argument('--ss_thr', help='semantic segmentation gray scale thr.', type=int, default=127)
parsed_args = parser.parse_args(sys.argv[1:])

ss_thr = parsed_args.ss_thr
path_ss = parsed_args.path_ss
path_od = parsed_args.path_od
path_out = parsed_args.path_out
path_ss_gt = parsed_args.path_ss_gt


"""
CALL: 
python get_n.py --path_od path/to/od/preds \
                --path_ss path/to/ss/preds \
                --path_ss_gt path/to/ss_gts \
                --path_out /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model_cat --ss_thr 100

"""


def main():

    list_od = natsorted(os.listdir(path_od))
    list_ss = natsorted(os.listdir(path_ss))
    list_ss_gt = natsorted(os.listdir(path_ss_gt))

    info_blobs_list = list()

    if  len(list_ss_gt) != len(list_ss) or len(list_od) != len(list_ss):
        print("NOT SAME LENGTH!")
        exit()

    for idx in range(len(list_od)):

        print("working on:" + list_ss_gt[idx])
        print("working on:" + list_ss[idx])
        print("working on:" + list_od[idx])
        
        # LOAD PREDS
        file_path_od = os.path.join(path_od,list_od[idx])
        file_path_ss_gt = os.path.join(path_ss_gt,list_ss_gt[idx])
        file_path_ss = os.path.join(path_ss,list_ss[idx])
        
        # LOAD PREDS
        image_ss_gray = cv2.imread(file_path_ss, cv2.IMREAD_GRAYSCALE)  # read ss image
        image_ss_gt = cv2.imread(file_path_ss_gt, cv2.IMREAD_GRAYSCALE)  # read gt image
        instances_od = getInstances(file_path_od, image_ss_gray.shape)  # read od predictions

        # BINARIZE SEMANTIC RPEDS
        image_ss_bw = cv2.threshold(image_ss_gray, ss_thr, 255, cv2.THRESH_BINARY)[1]
        image_ss_gt = cv2.threshold(image_ss_gt, 127, 255, cv2.THRESH_BINARY)[1]
        image_ss_gt = np.asarray(image_ss_gt)
        # MORPHOLOGICAL OPERATIONS
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(image_ss_bw, kernel, iterations=2) 
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        # BLOB DETECTION
        print("detecting blobs")
        n_labels, label_map, values, centroid = cv2.connectedComponentsWithStats(dilation,4,cv2.CV_32S)

        # FILTERING
        print("filtering blobs")
        for i in range(1, n_labels):
            area = values[i, cv2.CC_STAT_AREA]  
            if area<200:
                label_map[np.where(label_map==i)]=0

        # VALIDATE BLOBS
        print("validating blobs")
        
        blob_set = set(label_map.flat)
        blob_set.pop()

        for idx, i in enumerate(blob_set): # TODO CHANGE A QUE I SOLO SEA LOS NUMEROS DE LSO BLOBS QUE HAN SOBREVIVIDO AL FILTERING
            print("working on blob " + str(idx+1) + "/" + str(len(blob_set)))
            blob_map = np.zeros(image_ss_gray.shape, dtype="uint8")
            blob_map[np.where(label_map==i)]=1
            cov_list, n_list, sum_list = get_validation(blob_map, instances_od)

            auc = trapz(cov_list, dx=1)   # https://i.ytimg.com/vi/9wz7djdke-U/maxresdefault.jpg

            real = check_blob(blob_map, image_ss_gt)
            info_blob = [auc, n_list[0], real]
            info_blobs_list.append(info_blob)

    info_blobs_np = np.asarray(info_blobs_list)

    sn=path_out+'/info_blobs.npy'

    np.save(sn, info_blobs_np)

    print("auc: ",info_blobs_np[0][:])
    print("n_list: ",info_blobs_np[0][1])
    print("info_blobs dim: ",info_blobs_np.size())

    # Això és l'input!
    # SVMs
    # Performing GS to tune parameters for best SVM fit 

    

    C_range = [1, 10, 100]
    gamma_range = [ 1e-1,1,1e2]
    kernel=['poly','rbf','linear','sigmoid']

    params_grid = [{'kernel': ['rbf'], 'gamma': gamma_range,
                        'C': C_range},{'kernel': ['poly'],'degree':[7], 'C': C_range}]

    svm_model = GridSearchCV(SVC(), params_grid)
    svm_model.fit(X_train_scaled, y_true_tr)

    print("The best parameters are %s with a score of %0.2f"
        % (svm_model.best_params_, svm_model.best_score_))


if __name__ == "__main__":
    main()
