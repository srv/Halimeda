import os
import re
import glob
import sys
import argparse
import numpy as np
from scipy import ndimage
import scipy.misc
import pandas as pd

import imageio.v2 as imageio

'''
call:
python coverage.py --shape 1024 --path_im ../halimeda/cthr/ --path_out ../halimeda/coverage --grid 500 --thr 84
'''


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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', help='images shape.', type=int)   
    parser.add_argument('--path_im', help='txt input directory.')
    parser.add_argument('--path_out', help='im output directory.')
    parser.add_argument('--grid', default=0, help='grid AxA.')
    parser.add_argument('--thr', help='ss conf thr.', type=int)   

    parsed_args = parser.parse_args(sys.argv[1:])

    shape = parsed_args.shape
    path_im = parsed_args.path_im
    path_out = parsed_args.path_out
    grid = int(parsed_args.grid)
    thr = parsed_args.thr


    file_list = os.listdir(path_im)

    try:
        os.mkdir(path_out)
    except:
        print("")
    
    test_cases = list()
    cov_pix_list = list()
    cov_grid_list = list()

    for file in sorted(file_list):
        name, ext = os.path.splitext(file)
        test_cases.append(name)
        file_path = os.path.join(path_im, file)
        pred_im = imageio.imread(file_path)
        pred_im = np.where(pred_im<=thr,0,1)
        cov_pix = (np.sum(pred_im != 0)/np.size(pred_im))*100
        cov_pix_list.append(cov_pix)

        if grid > 0:
            count = 0
            total = grid*grid
            step_h = shape/grid  
            step_w = shape/grid   
            index_h = list()
            index_w = list()

            for i in range(1, grid):
                index_h.append(int(step_h*i))
                index_w.append(int(step_w*i))

            split1 = np.array_split(pred_im, index_h)

            for sp1 in enumerate(split1):
                split2 = np.array_split(sp1[1], index_w, axis=1)

                for sp2 in enumerate(split2):
                    if np.sum(sp2[1] != 0) > 0:
                        count = count+1

            cov_grid = (count/total)*100
            cov_grid_list.append(cov_grid)

        # save spine results on csv
        header = ['cov_pix', 'cov_grid']
        cov_csv = ({header[0]: cov_pix_list, header[1]: cov_grid_list})
        df = pd.DataFrame.from_records(cov_csv, index=test_cases)
        df.to_csv(path_out + "/coverage_"+str(grid)+".csv")


main()