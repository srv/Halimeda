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

##PANIC RUNS:

# 1) PANIC RUN -> Ejecutar lo siguiente y esperar que salga peor 



# python get_n.py --path_od ../n_panic/od_labels/ --path_ss ../n_panic/ss_pred/ --path_ss_gt ../n_panic/ss_gt/ --path_out ../n_panic/ --iter 1 --ss_thr 96

#python merge_blobs.py --path_od ../merge/test/inference/od/labels/ --path_ss ../merge/test/inference/ss/ --ss_thr 96 \
#                       --path_merge ../merge/test/merged_blob/merged_1_panic  --iter 1  --ns 1	3	1	11	1	1	1	1	1	1

#python evaluation.py --run_name n_panic  --pred_path ../merge/test/merged_blob/merged_1_panic  \
#                     --out_path ../merge/test/merged_blob/merged_1_panic  --gt_path ../merge/test/gt --shape 1024

# 2) GOOD RUN THR OD CORRECTED:

# python get_n.py --path_od ../n_panic/od_labels/ --path_ss ../n_panic/ss_pred/ --path_ss_gt ../n_panic/ss_gt/ --path_out ../n_thr_corrected/ --iter 1 --ss_thr 84

# python merge_blobs.py --path_od ../merge/test/inference/od/labels/ --path_ss ../merge/test/inference/ss/ \
#                       --ss_thr 84 --path_merge ../merge/test/merged_blob/merged_1_thr_corrected  --iter 1  --ns xxxxx


#python evaluation.py --run_name n_thr_corrected  --pred_path ../merge/test/merged_blob/merged_1_thr_corrected  \
#                     --out_path ../merge/test/merged_blob/merged_1_thr_corrected  --gt_path ../merge/test/gt --shape 1024
