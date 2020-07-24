cat 448_cremi_train.sh 
python -W ignore main.py --env 448_s1.5_m1.0_final_2 --gpu-id  0 1 2 3 4 5 6 7 --workers 8 --lbl-agents 0 --valid-gpu 0 \
--num-steps 6 --max-episode-length 6 --reward seg --model AttUNet2 --out-radius 4 8 \
--use-masks --size 224 224 --log-period 50 --features 32 64 128 256 512 --entropy-alpha 0.05 \
--downsample -1 --data 448_cremi --in-radius 0.8 --lr 1e-4 \
--fgbg-ratio 0.7 --st-fgbg-ratio 0.7 \
--mer_w 1.0 --spl_w 1.5 \
--save-period 50 --minsize 20 \
--max-temp-steps 99 \
--multi 1 \
--log-dir logs/Fer2019/FINAL/ \
--dilate-fac 2 \
--deploy \
--load trained_models/448_cremi/448_s1.5_m0.5_cnt1_AttUNet2_masks_seg_448_cremi/448_s1.5_m0.5_cnt1_AttUNet2_masks_seg_448_cremi_7850.dat 
 \

