

# INFERENCE GUIDE

# 1. Download the pretrained models
[OD model](https://zenodo.org/records/7611869#.Y_xsFSbMJD8)

[SS model](https://github.com/srv/SS_Halimeda/blob/a188edebc8e528a79dcfa74f5decdb0d8f7aacfd/model/000033/model.h5)

# 2. Run inference on your images

### 2.1 OD Inference

#### 2.1.1 Run yolov5 model inference

```bash
cd Halimeda/object_detection/yolov5/

python detect.py --weights $path_to_model.pt \
    --project $path_to_your_project_folder --name $results_od_folder --data data/halimeda.yaml \
    --source $path_to_your_images --conf-thres 0.1 --iou-thres 0.5 \
    --save-txt --save-conf --imgsz 1024
```

ARGS EXPLAINED:

    $path_to_model.pt: path to OD model downloaded from the link to zenodo
    $path_to_your_project_folder: path to your project folder
    $results_od_folder: inference results will be stored in $path_to_your_project_folder/$results_od_folder path
    $path_to_your_images: the path containing the images for performing inference.

#### 2.1.2 Convert the bounding boxes predictions to a coverage


```bash

cd Halimeda/scripts/

python coverage_od.py --shape 1024 --path_txt $path_to_your_project_folder/$results_od_folder/labels \
    --path_out $path_to_your_output_coverage_folder --grid 500

```

ARGS EXPLAINED:

    $path_txt: path to the predicted bounding boxes in txt form resulting from executing detect.py
    $path_to_your_output_coverage_folder: path to the folder where the bounding box predictions converted to grayscale images are stored

#### 2.1.3 Optional Apply a threshold to the coverage grayscale images

```bash

cd Halimeda/scripts
python apply_theshold.py --input_dir $path_to_your_output_coverage_folder --output_dir $path_to_your_output_coverage_folder_thresholded --threshold $threshold_value

```
ARGS EXPLAINED:

    $path_to_your_output_coverage_folder: path to the coverage grayscale images
    $path_to_your_output_coverage_folder_thresholded: path to the output binarized images. (For sparse scenarios) -> pixels in white are halimeda and pixels in black will be considered background.
    $threshold_value: threshold to binarize the images. For an OD scenario, the recommended experimental values are 43 or 75.

### 2.2 SS inference
#### 2.2.1 Run the trained SS_model "model.h5" inference

```bash
cd semantic_segmentation/SS_halimeda/scripts
python3 inference.py --run_path SS_Halimeda/model/ --data_path $path_to_your_images --shape 1024 --path_out results_ss_folder
```
ARGS EXPLAINED:

    $results_ss_folder: inference results will be stored in $results_ss_folder path
    $path_to_your_images: the path containing the images for performing inference.

#### 2.2.2 Optional Apply a threshold to the predictions

```bash

cd Halimeda/scripts
python apply_theshold.py --input_dir $results_ss_folder --output_dir $results_ss_folder_thresholded --threshold $threshold_value
```
ARGS EXPLAINED:

    $results_ss_folder: path to the predictions of the SS net
    $results_ss_folder_thresholded: path to the output binarized images. (For dense scenarios) -> pixels in white are halimeda and pixels in black will be considered background.
    $threshold_value: threshold to binarize the images. For an SS scenario, the recommended experimental value is 123

### 3 Merge
#### 3.1 Weighted merge

```bash
cd Halimeda/scripts
python w_merge.py --path_od $path_to_your_output_coverage_folder \
    --path_ss $results_ss_folder \
    --path_merge $results_w_merge_folder
```
ARGS EXPLAINED:

    $path_to_your_output_coverage_folder: path to the coverage grayscale images corresponding to the OD predictions
    $results_ss_folder: path to the predictions of the SS net
    $results_w_merge_folder: path where the results of the weighted merge will be stored.

#### 3.2 AUC merge

```bash

cd Halimeda/scripts

python merge_blobs.py --path_od $path_to_your_project_folder/$results_od_folder/labels \
    --path_ss $results_ss_folder \
    --path_merge $results_AUC_merge_folder --iter 1  --ns 12 3 1 2 1 1 1 1 1 1
```

# TUNING GUIDE:
In this section, examples of calls for every tuning step are provided.

For more information on how to reparameterize every step of the algorithm, please refer to our [paper](https://www.mdpi.com/2077-1312/12/1/70).

## In Case that you want to retrain the OD model:

```bash
cd Halimeda/object_detection/yolov5

python train.py --img 1024 --batch 8 --epochs 200 --data halimeda.yaml --weights yolov5x.pt --project projects/halimeda/final_trainings/yolo_XL/ --name base_1  --single-cls --hyp data/hyps/final/hyp.base.yaml
```

**Keep in mind:**

- You must modify halimeda.yaml to match the paths to your dataset folder.
- If you want to train from pretrained ImageNet weights use --weights yolov5x.pt, if you want to retrain our model use --weights $path_to_model.pt with the corresponding path where you have stored our downloaded model.
- Change the hyperparameters in data/hyps/final/hyp.base.yaml according to your needs.

**Eval command:**

```bash
python3 Halimeda/scripts/eval.py --run_name 000033 --path_pred ../runs/1/000033/inference --path_out ../runs/1/000033/ --path_gt ../data/test/mask --name xxx.py --pred_path /mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/base_3/inference_test/coverage --gt_im_path  /mnt/c/Users/haddo/yolov5/datasets/halimeda/coverage/test/ --gt_label_path /mnt/c/Users/haddo/yolov5/datasets/halimeda/labels/test/ --run_name base_3 --save_path /mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/base_3/inference_test --shape 1024
```
## In Case that you want to retrain the SS model:

Following, you have an example of calls, substitute in each instruction substitute for your paths and config

```bash

python3 train.py --run_path ../runs/1/0009/ --data_path ../data/1/ --shape 1024 --batch 3 --learning 0.009

python3 inference.py --run_path ../runs/5/0003_da/ --data_path ../data/5_da/val/img --shape 1024

```
Find best threshold with val partition

```bash

python3 eval.py --run_name 0009 --path_pred ../runs/1/0009/inference_val --path_out ../runs/1/0009/ --path_gt ../data/1/val/mask --name 'val'
```

Final evaluation over the independent test set with the obtained thr

```bash

python3 eval_thr.py --run_name 000033 --path_pred ../runs/5/000033/inference_test --path_out ../runs/5/000033/ --path_gt ../data/test_ss/mask --name 'test' --thr 123
```
## Tuning of the w_merge and AUC_merge:

**Find the best w_merge weights:**

```bash

python merge_weights.py --path_od /mnt/c/Users/haddo/DL_stack/Halimeda/inference_val/OD/ \
                    --path_ss /mnt/c/Users/haddo/DL_stack/Halimeda/inference_val/SS/ \
                    --path_merge /mnt/c/Users/haddo/DL_stack/Halimeda/weighted_merge/w_sweep/
```

**Find the n thresholds:**

```bash

python get_n.py --path_od ../inference_val/OD_labels/ \
    --path_ss ../inference_val/SS/ --path_ss_gt ../dataset/all_val/gt/ --path_out ../AUC_merge/cthr0/ --iter 1

```
