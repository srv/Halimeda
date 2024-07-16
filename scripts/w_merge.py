import os
import re
import numpy as np
from os import listdir
import argparse
import sys
from natsort import natsorted
import imageio.v2 as imageio


"""
Call Example:

python w_merge.py   --path_od /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_DATASET/INFERENCE_best_models/OD_over_all_test \
                    --path_ss /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_DATASET/INFERENCE_best_models/SS_over_all_test/ \
                    --path_merge /mnt/c/Users/haddo/DL_stack/Halimeda/dataset/NEW_DATASET/INFERENCE_best_models/w_merge_test

"""

def main():
    print('Start!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_od', help='path to the od output folder.')
    parser.add_argument('--path_ss', help='path to the ss output folder.')
    parser.add_argument('--path_merge', help='path to merge folder')
    parser.add_argument('--od_weight', default=0.85, help='weight to be applied to the OD inferences', type=float)
    parser.add_argument('--ss_weight', default=0.15, help='weight to be applied to the SS inferences', type=float)
    parser.add_argument('--binarize', default=False, help='flag to enable binarization', type=bool)
    parser.add_argument('--b_thres', default=127, help='binarization threshold', type=int)
    parsed_args = parser.parse_args(sys.argv[1:])
    print("IMported")
    path_od = parsed_args.path_od
    path_ss = parsed_args.path_ss
    path_merge = parsed_args.path_merge  # get class txt path
    w_od = parsed_args.od_weight # if range not 0-255 change to 0.2*255
    w_ss = parsed_args.ss_weight
    is_binarization_enabled = parsed_args.binarize
    b_thres = parsed_args.b_thres
    print('saved')
    raw_merged_folder = os.path.join(path_merge, 'raw/')
    binary_merged_folder = os.path.join(path_merge, 'binarized/')

    list_od = natsorted([file for file in os.listdir(path_od) if os.path.isfile(os.path.join(path_od,file))])
    list_ss = natsorted(os.listdir(path_ss))

    # list_od =  os.listdir(path_od)
    # list_ss = os.listdir(path_ss)
    print('ready')
    if  len(list_od) == len(list_ss):

        if not os.path.exists(path_merge):
            os.mkdir(path_merge)
            print("creating folder: ", path_merge)

        for idx in range(len(list_od)):

            file_path_od = os.path.join(path_od,list_od[idx])
            file_path_ss = os.path.join(path_ss,list_ss[idx])

            image_od = imageio.imread(file_path_od)  # read od image
            image_ss = imageio.imread(file_path_ss)  # read ss image

            # print("set OD is:", set(image_od.ravel()))
            # print("---------------------------------------------------------------------")
            # print("---------------------------------------------------------------------")
            # print("set SS is:", set(image_ss.ravel()))

            image_merged = (image_od * w_od) + (image_ss * w_ss)
            image_merged = np.asarray(image_merged)
            image_merged = image_merged.astype(np.uint8)

            file_path_merge = os.path.join(raw_merged_folder, list_od[idx])  # takes od name
            if not os.path.exists(raw_merged_folder):
                os.mkdir(raw_merged_folder)
                print("creating folder: ", raw_merged_folder)

            imageio.imsave(file_path_merge, image_merged)

            if is_binarization_enabled:
                file_path_merge = os.path.join(binary_merged_folder, list_od[idx])

                if not os.path.exists(binary_merged_folder):
                    os.mkdir(binary_merged_folder)
                    print("creating folder: ", binary_merged_folder)
                image_binarized = np.where(image_merged >= b_thres, 255, 0)
                image_binarized = image_binarized.astype(np.uint8)
                imageio.imsave(file_path_merge, image_binarized)

    else:
        print("NOT SAME LENGTH!!!!")
        print("od num files: ", len(list_od), " and ss:", len(list_ss))


    # image_od = imageio.imread("/home/uib/Halimeda/temp_od/halimeda_70_cov.jpg")  # read od image
    # image_ss = imageio.imread("/home/uib/Halimeda/temp_ss/halimeda_70_grey.jpg")  # read ss image

    # print("set OD is:",set(image_od.ravel()))
    # print("---------------------------------------------------------------------")
    # print("---------------------------------------------------------------------")
    # print("set SS is:",set(image_ss.ravel()))

    # image_merged = (image_od*w_od)+(image_ss*w_ss)
    # image_merged=np.asarray(image_merged)
    # image_merged=image_merged.astype(np.uint8)
    # imageio.imsave("merged_im.jpg",image_merged)



if __name__ == "__main__":
    main()
