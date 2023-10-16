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


python w_merge.py   --path_od /mnt/c/Users/haddo/DL_stack/Halimeda/inference_test/OD/yolo_nano_all_test \
                    --path_ss /mnt/c/Users/haddo/DL_stack/Halimeda/inference_test/SS/all/ \
                    --path_merge  /mnt/c/Users/haddo/DL_stack/Halimeda/weighted_merge/yolo_nano

"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_od', help='path to the od output folder.')
    parser.add_argument('--path_ss', help='path to the ss output folder.')
    parser.add_argument('--path_merge', help='path to merge folder')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_od = parsed_args.path_od
    path_ss = parsed_args.path_ss
    path_merge = parsed_args.path_merge  # get class txt path

    list_od = natsorted([file for file in os.listdir(path_od) if os.path.isfile(os.path.join(path_od,file))])
    list_ss = natsorted(os.listdir(path_ss))

    list_od =  os.listdir(path_od)
    list_ss = os.listdir(path_ss)

    w_od=0.7 # if range not 0-255 change to 0.2*255
    w_ss=0.3

    w_folder_name="weighted_merge"

    if  len(list_od) == len(list_ss):

        for idx in range(len(list_od)):

            file_path_od = os.path.join(path_od,list_od[idx])
            file_path_ss = os.path.join(path_ss,list_ss[idx])

            image_od = imageio.imread(file_path_od)  # read od image
            image_ss = imageio.imread(file_path_ss)  # read ss image

            print("set OD is:",set(image_od.ravel()))
            print("---------------------------------------------------------------------")
            print("---------------------------------------------------------------------")
            print("set SS is:",set(image_ss.ravel()))

            image_merged = (image_od*w_od)+(image_ss*w_ss)
            image_merged=np.asarray(image_merged)
            image_merged=image_merged.astype(np.uint8)

            file_folder_merge=os.path.join(path_merge,w_folder_name)  # takes od name
            file_path_merge=os.path.join(file_folder_merge,list_od[idx])  # takes od name

            if not os.path.exists(file_folder_merge):
                os.mkdir(file_folder_merge)
                print("creating folder: ",file_folder_merge)

            imageio.imsave(file_path_merge, image_merged)

    else:
        print("NOT SAME LENGTH!!!!")
        print("od num files: ",len(list_od)," and ss:", len(list_ss))


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
