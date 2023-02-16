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

parser = argparse.ArgumentParser()
parser.add_argument('--path_od', help='path to the od output folder.', type=str)  
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
python get_n.py --path_od path/to/od/preds --path_ss path/to/ss/preds --path_ss_gt path/to/ss_gts --path_out path/to/out --ss_thr 100

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

    metrics_list_list = list()

    for i in range(10):

        print("evaluating aucs from " + str(i*10) + " - " + str((i+1)*10))
        info_blobs_range_list = list()
        for j in range(len(info_blobs_list)):
            if info_blobs_np[j][0] > (i)*10 and info_blobs_np[j][0] < (i+1)*10:
                info_blobs_range_list.append(info_blobs_np[j])
            
        info_blobs_range_np = np.asarray(info_blobs_range_list)
        print(info_blobs_range_np)
        metrics_list = list()
        for n in tqdm(range(500)):
            info_blobs_range_n_over_list = list()
            info_blobs_range_n_below_list = list()
            for j in range(len(info_blobs_range_list)):
                if info_blobs_range_np[j,1] > n:
                    info_blobs_range_n_over_list.append(info_blobs_range_np[j])
                else:
                    info_blobs_range_n_below_list.append(info_blobs_range_np[j])

            info_blobs_range_n_over_np = np.asarray(info_blobs_range_n_over_list)
            info_blobs_range_n_below_np = np.asarray(info_blobs_range_n_below_list)
            
            size_over = len(info_blobs_range_n_over_list)
            tp = 0
            fp = 0
            if size_over >0:
                tp = np.count_nonzero(info_blobs_range_n_over_np[:,2])
                fp = size_over-tp

            size_below = len(info_blobs_range_n_below_np)
            fn = 0
            tn = 0
            if size_below>0:
                fn = np.count_nonzero(info_blobs_range_n_below_np[:,2])
                tn = size_below-fn

            acc = 0
            prec = 0
            rec = 0
            fall = 0
            f1 = 0

            acc = zero_division(tp+tn, tp+tn+fp+fn)
            prec = zero_division(tp,tp+fp)
            rec = zero_division(tp,tp+fn)
            fall = zero_division(fp,fp+tn)
            f1 = 2*(zero_division(prec*rec,prec+rec))
            n_blobs = len(info_blobs_range_list)
            metrics = [acc, prec, rec, fall, f1, n_blobs]

            metrics_list.append(metrics)

        metrics_list_list.append(metrics_list)

    mets = np.hstack(metrics_list_list)
    df = pd.DataFrame (mets)
    filepath = os.path.join(path_out, 'mets.xlsx')
    df.to_excel(filepath, index=False)

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
