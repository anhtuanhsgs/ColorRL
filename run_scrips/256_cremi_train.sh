cat $0
python -W ignore main.py --env test \
--gpu-id  0 1 2 3 --workers 8 --lbl-agents 0 --valid-gpu 0 \
--num-steps 6 --max-episode-length 6 --max-temp-steps 99 \
--reward seg --model AttUNet2 --out-radius 12 18 --use-masks \
--size 224 224 --features 32 64 128 256 512 \
--entropy-alpha 0.05 \
--downsample -1 --data 256_cremi --in-radius 1.1 --lr 1e-4 \
--fgbg-ratio 0.5 --st-fgbg-ratio 0.5 \
--mer_w 1.0 --spl_w 1.5 \
--save-period 5 --log-period 50 \
--minsize 20 \
--max-temp-steps 99 \
--split prox \
--log-dir logs/Aug2020/ \
--save-model-dir trained_models \
--dilate-fac 2 \
--rew-drop 8 --rew-drop-2 8 \

#--load trained_models/cremi/256_s1.5_m0.5_AttUNet2_masks_seg_cremi/256_s1.5_m0.5_AttUNet2_masks_seg_cremi_5350.dat 

#--load trained_models/256_cremi/256_s0.8_m1.2_cnt3_AttUNet2_masks_seg_256_cremi/256_s0.8_m1.2_cnt3_AttUNet2_masks_seg_256_cremi_5200.dat  

#trained_models/256_cremi/256_s1.0_m1.0_cnt2_AttUNet2_masks_seg_256_cremi/256_s1.0_m1.0_cnt2_AttUNet2_masks_seg_256_cremi_2000.dat 

#trained_models/256_cremi/256_s1.2_m0.8_cnt1_AttUNet2_masks_seg_256_cremi/256_s1.2_m0.8_cnt1_AttUNet2_masks_seg_256_cremi_1100.dat \
