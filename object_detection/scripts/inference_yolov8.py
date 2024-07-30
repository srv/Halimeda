from os import system, path, listdir
import argparse
from ultralytics import YOLO

"""
Example: 
command = "python3 inference.py --run_path /mnt/c/Users/haddo/SS_Halimeda/model/ --data_path /mnt/c/Users/haddo/Halimeda/merged_model_0/val --shape 1024 "

"""

parser = argparse.ArgumentParser()
parser.add_argument('--weights', help='Path to the run folder', type=str)
parser.add_argument('--project', help='Path to the run folder', type=str)
parser.add_argument('--source', help='Path to the run folder', type=str)
parser.add_argument('--conf_thres', help='Path to the run folder', type=str)
parser.add_argument('--iou_thres', help='Path to the run folder', type=str)
parser.add_argument('--imgsz', help='Path to the run folder', type=str)
parsed_args = parser.parse_args()

weights_path = parsed_args.weights
project_path = parsed_args.project
source_data_path = parsed_args.source
conf_thres = float(parsed_args.conf_thres)
iou_thres = float(parsed_args.iou_thres)
image_size = int(parsed_args.imgsz)

# Load model
model = YOLO(weights_path)

images_list = sorted(listdir(source_data_path))
images = [path.join(source_data_path, image) for image in images_list]  
for image in images_list:
    image_path = path.join(source_data_path, image)

    # Do inference to images
    model.predict(image_path, project=project_path, save=True, imgsz=image_size, conf=conf_thres, iou=iou_thres, save_txt=True, save_conf=True, exist_ok=True)



predictions_path = path.join(project_path, "predict/labels/")
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
