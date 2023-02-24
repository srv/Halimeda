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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument('--path_od', help='path to the od instances out folder.', type=str)  
parser.add_argument('--path_ss', help='path to the ss output folder.', type=str)
parser.add_argument('--path_ss_gt', help='path to the ss gt folder.', type=str)
parser.add_argument('--path_out', help='path to the ss gt folder.', type=str)
parser.add_argument('--ss_thr', help='semantic segmentation gray scale thr.', type=int, default=82)
parsed_args = parser.parse_args(sys.argv[1:])

ss_thr = parsed_args.ss_thr
path_ss = parsed_args.path_ss
path_od = parsed_args.path_od
path_out = parsed_args.path_out
path_ss_gt = parsed_args.path_ss_gt


"""
CALL: 
python combination_cat.py --path_od /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/cat_test/inference_OD/ \
                        --path_ss /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/cat_test/inference_SS/ \
                        --path_ss_gt /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/cat_test/gt \
                        --path_out /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/cat_test/out --ss_thr 100


python combine_cat_from_npy.py --path_od /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/inference_test/inference_OD/all_test/inference/labels \
                        --path_ss /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/inference_test/inference_SS/all_test \
                        --path_ss_gt /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/gt_test/all \
                        --path_out /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/cat_test/out --ss_thr 100

"""


def main():

    list_od = natsorted(os.listdir(path_od))
    list_ss = natsorted(os.listdir(path_ss))
    list_ss_gt = natsorted(os.listdir(path_ss_gt))


    if  len(list_ss_gt) != len(list_ss) or len(list_od) != len(list_ss):
        print("NOT SAME LENGTH!")
        exit()

    # Això és l'input!
    
    sp_svm=path_out+'/info_blobs_val.npy'

    info_blobs_np = np.load(sp_svm)

    auc=info_blobs_np[:,0]
    n_list= info_blobs_np[:,1]
    real= info_blobs_np[:,2]
    

    plt.figure()
    plt.scatter(auc, n_list, s=5, c=real,alpha=0.5)
    # plt.ylim(0, 400)
    plt.xlabel('auc')
    plt.ylabel('n boxes')
    plt.legend()
    plt.savefig(path_out+"/AUC_n_list.png")

    print("info_blobs: ",info_blobs_np)
    print("auc: ",info_blobs_np[:,0])
    print("n_list: ",info_blobs_np[:,1])
    print("real: ",info_blobs_np[:,2])
    print("info_blobs dim: ",info_blobs_np.shape)

    true_blobs=np.asarray([blob for blob in info_blobs_np if blob[2]==True])
    false_blobs=np.asarray([blob for blob in info_blobs_np if blob[2]==False])
    print("True blobs: ",true_blobs.shape)
    print("False blobs: ",false_blobs.shape)
    print("ALL blobs: ",info_blobs_np.shape)

    plt.figure()
    plt.scatter(true_blobs[:,0], true_blobs[:,1], s=1, c="green",alpha=0.5,label=("True"),marker='o')
    plt.xlabel('auc')
    plt.ylabel('n boxes')
    plt.legend()
    plt.savefig(path_out+"/True_blobs.png")
    plt.figure()
    plt.scatter(false_blobs[:,0], false_blobs[:,1], s=1, c="red",alpha=0.5,label=("False"),marker='x')
    plt.xlabel('auc')   
    plt.ylabel('n boxes')
    plt.legend()
    plt.savefig(path_out+"/False_blobs.png")

    # SVMs
    # Performing GS to tune parameters for best SVM fit 

    C_range = [1, 10, 100]
    gamma_range = [ 1e-1,1,1e2]
    kernel=['poly','rbf','linear','sigmoid']

    params_grid = [{'kernel': ['rbf','linear'], 'gamma': gamma_range,'C': C_range},{'kernel': ['poly'],'degree':[3], 'C': C_range}]
    #The best parameters are {'C': 100, 'gamma': 0.1, 'kernel': 'linear'} with a score of 0.83
    
    # # params_grid = [{'kernel': kernel, 'gamma': gamma_range,
    # #                     'C': C_range,'degree':[2,3,4,5], 'C': C_range}]

    # x_data=info_blobs_np[:,0:2]
    # print("x_data",x_data)
    # y_data=info_blobs_np[:,2]
    # print("y_data",y_data)

    # svm_model = GridSearchCV(SVC(), params_grid)
    # svm_model.fit(x_data, y_data)

    # print("The best parameters are %s with a score of %0.2f"
    #     % (svm_model.best_params_, svm_model.best_score_))
    
    # print(svm_model.best_params_.keys())
    # print(svm_model.best_params_.values())
    
    # df = pd.DataFrame(svm_model.best_params_,index=[0])
    # print(df)
    # df["score"]=svm_model.best_score_

    # df.to_excel(os.path.join(path_out,'results_svm.xlsx'))


def zero_division(n, d):
    return n / d if d else 0

def check_blob(blob, gt):
    check = False
    sum = 0
    ones = np.where(blob == 1)
    size = ones[0].size
    for i in range(size):
        if gt[ones[0][i], ones[1][i]] == 255:
            sum = sum + 1
    if sum/size >= 0.5:
        check  =True

    return check


def get_validation(blob, instances):

    # GET INSTANCES THAT OVERLAP WITH BLOB
    inst_blob = list()
    for inst in instances:
        found = False
        box = getBoxFromInst(inst)
        (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        for j in range(top, bottom):
            if found == False:
                for k in range(left,right):
                    if blob[j, k] == 1:
                        inst_blob.append(inst)
                        found = True
                        break
            else:
                break

    cov_list = list()
    n_list = list()
    sum_list = list()

    for thr in tqdm(range(0,100,1)):
        inst_thr= list()
        for inst in inst_blob:
            if inst[1]>(thr/100):
                inst_thr.append(inst)

        n, sum = get_aa(inst_thr)
        n_list.append(n)
        sum_list.append(sum)

        cov = get_coverage(blob,inst_thr)
        cov_list.append(cov)

    return cov_list, n_list, sum_list


def get_coverage(map, instances):
    size = 0
    count = 0

    map2 = copy.deepcopy(map)
    size = np.count_nonzero(map2 != 0)

    for inst in instances:
        box = getBoxFromInst(inst)
        (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        for j in range(top, bottom):
            for k in range(left, right):
                if map2[j,k] ==1:
                    count = count+1
                    map2[j,k] = 0
                    
    coverage = count/size

    return coverage


def get_aa(instances):
    sum = 0
    n = len(instances)
    for inst in instances:
        sum = sum + inst[1]
    return n, sum
    

def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def getInstances(file, shape):
    instances = list()
    fh1 = open(file, "r")
    for line in fh1:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(" ")
        idClass = (splitLine[0])  # class
        if len(splitLine) == 5:
            x = float(splitLine[1])
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            inst = (idClass, x, y, w, h)  
            bbox=yolo_to_xml_bbox([x, y, w, h], shape[1], shape[0])
            inst = (idClass, bbox[0], bbox[1], bbox[2], bbox[3])
        elif len(splitLine) == 6:
            confidence = float(splitLine[1])
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            inst = (idClass, confidence, x, y, w, h)   
            bbox=yolo_to_xml_bbox([x, y, w, h], shape[1], shape[0])
            inst = (idClass, confidence, bbox[0], bbox[1], bbox[2], bbox[3])
        instances.append(inst)
    fh1.close()
    return instances


def getBoxFromInst(inst):
    if len(inst) == 5:
        box = (inst[1], inst[2], inst[3], inst[4])
    elif len(inst) == 6:
        box = (inst[2], inst[3], inst[4], inst[5])
    return box


def nms(instances, thr):

    for i1, pred1 in enumerate(instances):
        intersectionArea1 = list()
        intersectionArea2 = list()
        delete = list()
        box1 = getBoxFromInst(pred1)
        area1 = getBoxArea(box1)

        for i2, pred2 in enumerate(instances):

            box2 = getBoxFromInst(pred2)
            area2 = getBoxArea(box2)

            inter = getIntersectionArea(box1, box2)
            ioa1 = inter/area1
            ioa2 = inter/area2

            intersectionArea1.append(ioa1)
            intersectionArea2.append(ioa2)

        for i3 in range(len(intersectionArea1)):
            if (intersectionArea1[i3] > thr or intersectionArea2[i3] > thr) and i3 != i1:
                delete.append(i3)
        for index in sorted(delete, reverse=True):
            del instances[index]

    return instances


def getIntersectionArea(boxA, boxB):
    if boxesIntersect(boxA, boxB) is False:
        return 0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA) * (yB - yA)


def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True


def getBoxArea(box):
    area = (box[2]-box[0])*(box[3]-box[1])
    return area


if __name__ == "__main__":
    main()
