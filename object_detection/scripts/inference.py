from os import system, path, listdir
import argparse

"""
Example: 
command = "python3 inference.py --run_path /mnt/c/Users/haddo/SS_Halimeda/model/ --data_path /mnt/c/Users/haddo/Halimeda/merged_model_0/val --shape 1024 "

"""

parser = argparse.ArgumentParser()
parser.add_argument('--weights', help='Path to the run folder', type=str)
parser.add_argument('--project', help='Path to the run folder', type=str)
parser.add_argument('--name', help='Path to the run folder', type=str)
parser.add_argument('--data', help='Path to the run folder', type=str)
parser.add_argument('--source', help='Path to the run folder', type=str)
parser.add_argument('--conf_thres', help='Path to the run folder', type=str)
parser.add_argument('--iou_thres', help='Path to the run folder', type=str)
parser.add_argument('--imgsz', help='Path to the run folder', type=str)
parsed_args = parser.parse_args()

weights_path = parsed_args.weights
project_path = parsed_args.project
name = parsed_args.name
data_params_path = parsed_args.data
source_data_path = parsed_args.source
conf_thres = parsed_args.conf_thres
iou_thres = parsed_args.iou_thres
image_size = parsed_args.imgsz

current_path = path.dirname(path.dirname(path.realpath(__file__)))
print(current_path)

# Command to execute the OD inference over the images
command_to_execute = "python " + path.join(current_path, "yolov5/detect.py") + " --weights " + weights_path + " --project " + project_path + " --name " + name + " --data " + data_params_path + \
                    " --source " + source_data_path + " --conf-thres " + conf_thres + " --iou-thres " + iou_thres + " --save-txt --save-conf --imgsz " + image_size
print(command_to_execute)
system(command_to_execute)

predictions_path = path.join(project_path, name + "labels/")
labels_files = listdir(predictions_path)
labels_names = []
[labels_names.append(path.splitext(labels_file)[0]) for labels_file in labels_files]

# Infered OD images with no detected instances do not generate a label.txt file. This piece of code is in charge
# of solving that.
images_files = listdir(source_data_path)
for image_file in images_files:
    image_name, ext = path.splitext(image_file)
    if image_name not in labels_names:
        open(path.join(predictions_path, image_name + ".txt"), 'a').close()

