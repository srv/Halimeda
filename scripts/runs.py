import os

command = "python3 train.py --run_path ../runs/3/da_lr/1024_3_0003_d/ --data_path ../data/splits/cross2/a/ --shape 1024 --batch 3 --learning 0.003"
os.system(command)

command = "python3 inference.py --run_path ../runs/3/da_lr/1024_3_0003_d/ --data_path ../data/splits/cross2/test/img --shape 1024"
os.system(command)

command = "python3 evaluation.py --run_path ../runs/3/da_lr/1024_3_0003_d/ --mask_path ../data/splits/cross2/test/mask --shape 1024"
os.system(command)

command = "python get_n.py --path_od ../merge/val_n/od/labels/ --path_ss ../merge/val_n/ss/pred/ --path_ss_gt ../merge/val_n/gt/ --path_out ../merge/val_n/merged/cthr1/ --iter 1"
os.system(command)

command = "python merge_outs_2.py --path_od ../merge/test_imgs/od/labels/ --path_ss ../merge/test_imgs/ss/pred/ --path_merge ../merge/test_imgs/merged_1/  --iter 1  --ns 12 8 1 2 1 1 1 1 1 1"
os.system(command)

command = "python3 evaluation.py --run_path ../runs/3/da_lr/1024_3_0003_d/ --mask_path ../data/splits/cross2/test/mask --shape 1024"
os.system(command)


command = "python coverage_ss.py --shape 1024 --path_im ../merge/test/merged_blob/merged_1/ --path_out ../merge/test/merged_blob/merged_1/coverage --grid 50 --thr 0"
os.system(command)

'''
python get_n.py --path_od ../merge/val/n/od/labels/ --path_ss ../merge/val/n/ss/pred/ --path_ss_gt ../merge/val/gt/ --path_out ../merge/val/n/merged/cthr0/ --iter 1 --ss_thr 96

python merge_blobs.py --path_od ../merge/test/inference/od/labels/ --path_ss ../merge/test/inference/ss/ --ss_thr 96 --path_merge ../merge/test/merged_blob/merged_1_panic  --iter 1  --ns 329 3 1 1 1 1 1 1 1 1
python evaluation.py --run_name n_panic  --pred_path ../merge/test/merged_blob/merged_1_panic  --out_path ../merge/test/merged_blob/merged_1_panic  --gt_path ../merge/test/gt --shape 1024

2) GOOD RUN THR OD CORRECTED:

python get_n.py --path_od ../merge/val/n/od/labels/ --path_ss ../merge/val/n/ss/pred/ --path_ss_gt ../merge/val/gt/ --path_out ../merge/val/n/merged/cthr1/ --iter 1 --ss_thr 84
python merge_blobs.py --path_od ../merge/test/inference/od/labels/ --path_ss ../merge/test/inference/ss/ --ss_thr 84 --path_merge ../merge/test/merged_blob/merged_1_thr_corrected  --iter 1  --ns 12 3 1 2 1 1 1 1 1 1
python evaluation.py --run_name n_thr_corrected  --pred_path ../merge/test/merged_blob/merged_1_thr_corrected  --out_path ../merge/test/merged_blob/merged_1_thr_corrected  --gt_path ../merge/test/gt --shape 1024

python coverage_ss.py --shape 1024 --path_im ../merge/test/merged_blob/merged_1_thr_corrected/ --path_out ../merge/test/merged_blob/merged_1_thr_corrected/coverage --grid 50 --thr 0

'''
