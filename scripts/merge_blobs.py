import os
import cv2
import sys
import copy
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from numpy import trapz
from natsort import natsorted
import matplotlib.pyplot as plt

"""
CALL: 

python merge_blobs.py --path_od ../merge/test_im/od/labels/ --path_ss ../merge/test_im/ss/pred/ --path_merge ../merge/test_im/merged_1/  --iter 1  --ns 12 3 1 2 1 1 1 1 1 1


python merge_blobs.py --path_od ../inference_test/OD/all_labels/ --path_ss ../inference_test/SS/all/ \
    --path_merge ../AUC_merge/inference_all_test/  --iter 1  --ns 20	3	1	1	1	1	1	1	1	1

    
    
python merge_blobs.py --path_od ../inference_test/OD/all_labels/ --path_ss ../inference_test/SS/all/ \
    --path_merge ../AUC_merge/inference_all_test_n_only_SS/  --iter 1  --ns 20	 91	1	1	1	3	1	1	1	1	1

    
   

PRINT:
cv2.imshow("erosion", erosion)
cv2.waitKey()
plt.imshow(blob_map, interpolation='none')
plt.show()
cv2.waitKey()
"""

parser = argparse.ArgumentParser()
parser.add_argument('--path_od', help='path to the od output folder.', type=str)  
parser.add_argument('--od_thr', help='semantic segmentation gray scale thr.', type=float, default=0.317)
parser.add_argument('--path_ss', help='path to the ss output folder.', type=str)
parser.add_argument('--ss_thr', help='semantic segmentation gray scale thr.', type=int, default=123)
parser.add_argument('--path_merge', help='path to merge folder', type=str)
parser.add_argument('--iter', help='n iterations for erode dilation', type=int, default=1)
parser.add_argument('--ns', nargs="+", type=int)
parsed_args = parser.parse_args(sys.argv[1:])

ss_thr = parsed_args.ss_thr
od_thr = parsed_args.od_thr
path_od = parsed_args.path_od
path_ss = parsed_args.path_ss
path_merge = parsed_args.path_merge  # get class txt path
iter = parsed_args.iter
ns = parsed_args.ns


def main():

    list_od = natsorted(os.listdir(path_od))
    list_ss = natsorted(os.listdir(path_ss))

    if  len(list_od) != len(list_ss):
        print("¡PREDS NOT SAME LENGTH!")
        exit()     

    for idx in range(len(list_od)):

        print("working on:" + list_od[idx])
        print("working on:" + list_ss[idx])
        file_path_od = os.path.join(path_od,list_od[idx])
        file_path_ss = os.path.join(path_ss,list_ss[idx])
        
        # LOAD PREDS
        image_ss_gray = cv2.imread(file_path_ss, cv2.IMREAD_GRAYSCALE)  # read ss image
        instances_od = getInstances(file_path_od, image_ss_gray.shape)  # read od predictions
        instances_od = sorted(instances_od, key=lambda conf: conf[1], reverse=True)
        

        # INIT FINAL COVERAGE LABEL MAP
        cov_merged = np.zeros(image_ss_gray.shape, dtype="uint8")

        # DELETE INSTANCES WITH CONF < 1
        for i, instance in enumerate(instances_od):
            if instance[1] < 0.01: 
                break
        instances_od = instances_od[:i]

        # PRINT BBOS OF OD PREDICTIONS WITH ENOUGH CONFIDENCE
        for instance in instances_od:
            if instance[1] > od_thr:
                box = getBoxFromInst(instance)
                (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                for j in range(top, bottom):
                    for k in range(left, right):
                        cov_merged[j, k] = 1
    
        # BINARIZE SEMANTIC RPEDS
        image_ss_bw = cv2.threshold(image_ss_gray, ss_thr, 255, cv2.THRESH_BINARY)[1]

        # MORPHOLOGICAL OPERATIONS
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(image_ss_bw, kernel, iterations=iter) 

        # BLOB DETECTION
        print("detecting blobs")
        n_labels, label_map, values, centroid = cv2.connectedComponentsWithStats(erosion,4,cv2.CV_32S)

        print("filtering blobs")
        for i in range(1, n_labels):
            area = values[i, cv2.CC_STAT_AREA]  
            if area<200:
                label_map[np.where(label_map==i)]=0

        print("validating blobs")
        blob_set = set(label_map.flat)
        blob_set.pop()

        for i, b in enumerate(blob_set):
            print("working on blob " + str(i+1) + "/" + str(len(blob_set)))

            blob_map = np.zeros(image_ss_gray.shape, dtype="uint8")
            blob_map[np.where(label_map==b)]=1
            blob_map = cv2.dilate(blob_map, kernel, iterations=iter)

            cov_list, n_list, sum_list = get_validation(blob_map, instances_od)

            auc = trapz(cov_list, dx=1)

            if  0<=auc<=10 and n_list[0]>ns[0]:
                cov_merged = np.clip(cov_merged + blob_map, 0, 1)
            elif 10<auc<=20 and n_list[0]>ns[1]:
                cov_merged = np.clip(cov_merged + blob_map, 0, 1)
            elif 20<auc<=30 and n_list[0]>ns[2]:
                cov_merged = np.clip(cov_merged + blob_map, 0, 1)
            elif 30<auc<=40 and n_list[0]>ns[3]:
                cov_merged = np.clip(cov_merged + blob_map, 0, 1)
            elif 40<auc<=50 and n_list[0]>ns[4]:
                cov_merged = np.clip(cov_merged + blob_map, 0, 1)
            elif 50<auc<=60 and n_list[0]>ns[5]:
                cov_merged = np.clip(cov_merged + blob_map, 0, 1)
            elif 60<auc<=70 and n_list[0]>ns[6]:
                cov_merged = np.clip(cov_merged + blob_map, 0, 1)
            elif 70<auc<=80 and n_list[0]>ns[7]:
                cov_merged = np.clip(cov_merged + blob_map, 0, 1)
            elif 80<auc<=90 and n_list[0]>ns[8]:
                cov_merged = np.clip(cov_merged + blob_map, 0, 1)
            elif 90<auc<=100 and n_list[0]>ns[9]:
                cov_merged = np.clip(cov_merged + blob_map, 0, 1)

        cov_merged = cov_merged*255 
        
        if not os.path.exists(path_merge):
            os.makedirs(path_merge)
        
        file_path_merge=os.path.join(path_merge,list_ss[idx])  # takes od name
        imageio.imsave(file_path_merge, cov_merged) 


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
                    if found == True:
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
