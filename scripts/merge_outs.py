import os
import re
import numpy as np
from os import listdir
import argparse
import sys
from natsort import natsorted
import imageio




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

    if  len(list_od) == len(list_ss):

        for idx in range(len(list_od)):
            file_path_od = os.path.join(path_od,list_od[idx])
            file_path_ss = os.path.join(path_ss,list_ss[idx])

            image_od = imageio.imread(file_path_od)  # read od image
            image_ss = imageio.imread(file_path_ss)  # read ss image
            image_merged = (image_od*0.5)+(image_ss*0.5)
            #image_merged = (image_od*0.75)+(image_ss*0.25)
            #image_merged = (image_od*0.25)+(image_ss*0.75)

            file_path_merge=os.path.join(path_merge,list_od[idx])  # takes od name
            imageio.imsave(file_path_merge, image_merged) 

    else:

        print("NOT SAME LENGTH!!!!")





if __name__ == "__main__":
    main()
