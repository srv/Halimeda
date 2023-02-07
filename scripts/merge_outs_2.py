import os
import re
import cv2
import sys
import imageio
import argparse
import numpy as np
from natsort import natsorted
from skimage.transform import resize


parser = argparse.ArgumentParser()
parser.add_argument('--path_od', help='path to the od output folder.', type=str)    
parser.add_argument('--path_ss', help='path to the ss output folder.', type=str)
parser.add_argument('--ss_thr', help='semantic segmentation gray scale thr.', type=float, default=100)
parser.add_argument("--od_thr_list", nargs="+", default=[0.5], type=float, help ='confidece threshold list')
parser.add_argument('--path_merge', help='path to merge folder', type=str)
parsed_args = parser.parse_args(sys.argv[1:])

ss_thr = parsed_args.ss_thr
od_thr_list = parsed_args.od_thr_list
path_od = parsed_args.path_od
path_ss = parsed_args.path_ss
path_merge = parsed_args.path_merge  # get class txt path

"""

CALL: 
python merge_outs_2.py --path_od path/to/od/preds --path_ss path/to/ss/preds --ss_thr 100 --od_thr_list 0.1 0.3 0.5 --path_merge path/to/output/folder

"""
def main():

    list_od = natsorted(os.listdir(path_od))
    list_ss = natsorted(os.listdir(path_ss))

    if  len(list_od) == len(list_ss):

        for idx in range(len(list_od)):

            file_path_od = os.path.join(path_od,list_od[idx])
            file_path_ss = os.path.join(path_ss,list_ss[idx])

            print("loading preds")
            instances_od = getInstances(file_path_od)  # read od predictions
            grey = np.ones((1024, 1024), dtype=np.uint8)
            image_ss = imageio.imread(file_path_ss, as_gray = True)  # read ss image
            image_ss = resize(image_ss, (1024, 1024), mode='constant', preserve_range=True)
            #image_ss  = np.where(image_ss>ss_thr,1,0)

            print("detecting blobs")
            params = cv2.SimpleBlobDetector_Params()
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(grey) # <- now works
            print("b")       
            im_with_keypoints = cv2.drawKeypoints(grey, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # Show keypoints
            cv2.imshow("Keypoints", im_with_keypoints)
            cv2.waitKey(0)

            #imageio.imsave(file_path_merge, image_merged) 

    else:

        print("¡NOT SAME LENGTH!")

















def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def getInstances(file):
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
            bbox=yolo_to_xml_bbox([x, y, w, h], 1024, 1024)
            inst = (idClass, bbox[0], bbox[1], bbox[2], bbox[3])
        elif len(splitLine) == 6:
            confidence = float(splitLine[1])
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            inst = (idClass, confidence, x, y, w, h)   
            bbox=yolo_to_xml_bbox([x, y, w, h], 1024, 1024)
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
