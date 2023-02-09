import os
import cv2
import sys
import argparse
import numpy as np
from tqdm import tqdm
from numpy import trapz
from natsort import natsorted
import matplotlib.pyplot as plt
from scipy.integrate import simpson

parser = argparse.ArgumentParser()
parser.add_argument('--path_od', help='path to the od output folder.', type=str)  
parser.add_argument('--od_thr', help='semantic segmentation gray scale thr.', type=float, default=0.6)
parser.add_argument('--path_ss', help='path to the ss output folder.', type=str)
parser.add_argument('--ss_thr', help='semantic segmentation gray scale thr.', type=float, default=100)
parser.add_argument('--metric_thr', help='semantic segmentation gray scale thr.', type=float, default=0.6)
parser.add_argument("--od_thr_list", nargs="+", default=[0.5], type=float, help ='confidece threshold list')
parser.add_argument('--path_merge', help='path to merge folder', type=str)
parsed_args = parser.parse_args(sys.argv[1:])

ss_thr = parsed_args.ss_thr
od_thr = parsed_args.od_thr
metric_thr = parsed_args.metric_thr
od_thr_list = parsed_args.od_thr_list
path_od = parsed_args.path_od
path_ss = parsed_args.path_ss
path_merge = parsed_args.path_merge  # get class txt path

"""
CALL: 
python merge_outs_2.py --path_od path/to/od/preds --path_ss path/to/ss/preds --ss_thr 100 --od_thr_list 0.1 0.3 0.5 --path_merge path/to/output/folder

"""
od_thr = 0.8
ss_thr = 127
thr_cov = 0.5
thr_box = 5

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

        
        # NMS OF HIGHLY SIMILAR BBOX
        # instances_od_nms = nms(instances_od, 0.9)



        # INIT FINAL COVERAGE LABEL MAP
        cov_merged = np.zeros(image_ss_gray.shape, dtype="uint8")

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
        erosion = cv2.erode(image_ss_bw, kernel, iterations=2) 
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        # BLOB DETECTION
        print("detecting blobs")
        n_labels, label_map, values, centroid = cv2.connectedComponentsWithStats(dilation,4,cv2.CV_32S)

        # FILTERING
        print("filtering blobs")
        for i in range(1, n_labels):
            area = values[i, cv2.CC_STAT_AREA]  
            if area<250:
                label_map[np.where(label_map==i)]=0

        # VALIDATE BLOBS
        print("validating blobs")
        for i in range(1, np.max(label_map)+1):
            print("working on blob " + str(i) + "/" + str(np.max(label_map)))
            blob_map = np.zeros(image_ss_gray.shape, dtype="uint8")
            blob_map[np.where(label_map==i)]=1
            auc, n_list, sum_list = get_validation(blob_map, instances_od)
            print(auc)
            print(n_list)
            print(sum_list)

            metric = auc + len(n_list) + len(sum_list) # calcular metrica definitiva a partir de auc y la lista de confidences en box ESTA METRIC LA PUEDE SACAR DIRECTAMENTE GET_VALIDATION..
            if metric>metric_thr:
                cov_merged[np.where(blob_map==i)]=1
        
        #PRINTS
        if True == True:
            cv2.imshow("image", image_ss_gray)
            cv2.imshow("binarization",image_ss_bw)
            cv2.imshow("Erosion", erosion)
            cv2.imshow("Dilation", dilation)
            plt.imshow(label_map, interpolation='none')
            plt.show()
            cv2.waitKey()


def get_validation(blob, instances):

    # GET INSTANCES THAT OVERLAP WITH BLOB
    inst_blob = list()
    for inst in instances:
        found = False
        box = getBoxFromInst(inst)
        (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        for j in range(top, bottom):
            if (j == top or j == bottom-1) and found == False:
                for k in range(left,right):
                    if blob[j, k] == 1:
                        inst_blob.append(inst)
                        found = True
                    if found == True:
                        break
            elif found == False:
                for k in [left, right-1]:
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

    for thr in tqdm(range(1,100,1)):

        inst_thr= list()
        for inst in inst_blob:
            if inst[1]<(thr/100):
                inst_thr.append(inst)

        n, sum = get_aa(inst_thr)
        n_list.append(n)
        sum_list.append(sum)

        cov = get_coverage(blob,inst_thr)
        cov_list.append(cov)
        print(cov)

    auc = trapz(cov_list, dx=1)   # https://i.ytimg.com/vi/9wz7djdke-U/maxresdefault.jpg
    return auc, n_list, sum_list


def get_coverage(map, instances):
    size = 0
    count = 0
    aux_map = np.zeros(map.shape, dtype="uint8")
    for inst in instances:
        box = getBoxFromInst(inst)
        (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        for j in range(top, bottom):
            for k in range(left, right):
                aux_map[j, k] = 1

    for j in range(map.shape[0]):
        for k in range(map.shape[1]):
            if map[j,k] ==1:
                size = size +1
                if aux_map[j,k] == 1:
                    count = count+1
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
