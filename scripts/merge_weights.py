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

python merge_outs.py --path_od /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/inference_test/inference_OD/all_test/coverage \
                    --path_ss /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/inference_test/inference_SS/all_test/ \
                    --path_merge /mnt/c/Users/haddo/DL_stack/Halimeda/combined_model/inference_test/combined_weights_swp/

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

    list_od = natsorted(os.listdir(path_od))
    list_ss = natsorted(os.listdir(path_ss))

    # weights_swep=np.linspace(0.05,0.95,19)
    weights_swep=[0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55,0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1]
    
    print("weights_sweep: ",weights_swep)

    if  len(list_od) == len(list_ss):
        for weight in weights_swep:
            w_folder_name=str(weight)
            od_w=weight
            ss_w=1-weight
            print("-----------------------------------------------------------------------------")
            print("WORKING ON FOLDER: ",w_folder_name)
            print("OD W:",od_w)
            print("SS W:",ss_w)
        
            for idx in range(len(list_od)):

                file_path_od = os.path.join(path_od,list_od[idx])
                file_path_ss = os.path.join(path_ss,list_ss[idx])

                image_od = imageio.imread(file_path_od)  # read od image
                image_ss = imageio.imread(file_path_ss)  # read ss image

                image_merged = (image_od*od_w)+(image_ss*ss_w)
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


if __name__ == "__main__":
    main()
