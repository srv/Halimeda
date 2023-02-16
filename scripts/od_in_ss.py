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
parser.add_argument('--path_ss_gt', help='path to the ss gt folder.', type=str)
parser.add_argument('--path_out', help='path to the ss gt folder.', type=str)
parsed_args = parser.parse_args(sys.argv[1:])

path_od = parsed_args.path_od
path_ss_gt = parsed_args.path_ss_gt
path_out = parsed_args.path_out

"""
CALL: 
python get_n.py --path_od path/to/od/preds --path_ss_gt path/to/ss_gts --path_out path/to/out
"""


def main():

    list_od = natsorted(os.listdir(path_od))
    list_ss_gt = natsorted(os.listdir(path_ss_gt))

    if  len(list_ss_gt) != len(list_od):
        print("NOT SAME LENGTH!")
        exit()

    tp_np =  np.zeros(100)
    fp_np = np.zeros(100)

    for idx in range(len(list_od)):

        print("working on:" + list_ss_gt[idx])
        
        # LOAD PREDS
        file_path_od = os.path.join(path_od,list_od[idx])
        file_path_ss_gt = os.path.join(path_ss_gt,list_ss_gt[idx])
        
        # LOAD PREDS
        image_ss_gt = cv2.imread(file_path_ss_gt, cv2.IMREAD_GRAYSCALE)  # read gt image
        instances = getInstances(file_path_od, image_ss_gt.shape)  # read od predictions
        instances = sorted(instances, key=lambda conf: conf[1], reverse=True)

        # BINARIZE SEMANTIC RPEDS
        image_ss_gt = cv2.threshold(image_ss_gt, 127, 255, cv2.THRESH_BINARY)[1]
        image_ss_gt = np.asarray(image_ss_gt)

        for cthr in range(100):

            fp = 0
            tp = 0

            instances_copy  =copy.deepcopy(instances)  

            for i, instance in enumerate(instances_copy):
                 if instance[1] < cthr/100:
                     break
            instances_copy = instances_copy[:i]


        # GET INSTANCES THAT OVERLAP WITH BLOB
            for inst in instances_copy:
                box = getBoxFromInst(inst)
                (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                found = False
                for j in range(top, bottom):
                    if found == False:
                        for k in range(left,right):
                            if image_ss_gt[j, k] == 255:
                                tp = tp+1
                                found = True
                                break
                    else:
                        break
                if found == False:
                    fp = fp+1

            tp_np[cthr] = tp_np[cthr] + tp
            fp_np[cthr] = fp_np[cthr] + fp

    out = np.vstack((tp_np, fp_np))
    df = pd.DataFrame (out)
    filepath = os.path.join(path_out, 'hist.xlsx')
    df.to_excel(filepath, index=False)


def zero_division(n, d):
    return n / d if d else 0


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


if __name__ == "__main__":
    main()
