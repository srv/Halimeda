
import os
import re
import numpy as np
import argparse
import sys
from natsort import natsorted
import pandas as pd


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', help='path to the run folder.')
    parser.add_argument('--path_out', help='path to the run folder.')
    parser.add_argument('--lookfor', help='csv name to look for.')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_in = parsed_args.path_in
    path_out = parsed_args.path_out
    lookfor = parsed_args.lookfor

    rows_data_list = list()
    rows_name_list = list()

    for root, dirs, files in os.walk(path_in):

        for file in enumerate(sorted(files)):     

            if re.search("\.(xlsx)$", file[1]):
                
                if lookfor in file[1]:

                    columns_name_list = list()

                    path_file = os.path.join(root, file[1])
                    excel = pd.ExcelFile(path_file, engine='openpyxl')
                    sheets = excel.sheet_names
                    data = excel.parse(sheets[0])

                    for i in range(data.shape[1]):
                        columns_name_list.append(str(data.columns[i]))

                    data = pd.DataFrame(data, columns=columns_name_list)

                    data_np = data.to_numpy()

                    name = os.path.join(root, file[1])

                    rows_name_list.append(name)
                    for i in range(data_np.shape[0]-1):
                        rows_name_list.append(" ")
                    rows_data_list.append(data_np)

    rows_data = np.vstack(rows_data_list)
    df = pd.DataFrame(data=rows_data, index=rows_name_list, columns=columns_name_list)
    filepath = os.path.join(path_out, lookfor + "_unified.xlsx")
    df.to_excel(filepath, index=True)


if __name__ == "__main__":
    main()