cat eval_cvppp.sh
python main.py --env re__256_s1.5_m0.5_r12_28 --gpu-id 2 --workers 0 \
--lbl-agents 0 --num-steps 6 --max-episode-length 6 --reward seg \
--model AttUNet2 --out-radius 12 28 --use-masks --size 176 176 \
--features 32 64 128 256 512 --entropy-alpha 0.05 \
--downsample -1 --data cvppp_eval --in-radius 0.8 --log-period 10 \
--lr 1e-4 --fgbg-ratio 0.3 --st-fgbg-ratio 0.3 --mer_w 0.5 \
--spl_w 1.5 --save-period 50 --minsize 12 \
--multi 1 \
--dilate-fac 2 \
--log-dir logs/Fer2019/Ablation_study/ \
--minsize 50 \
--deploy \
--eval-data train \
--load trained_models/cvppp/re__176_s1.5_m1.0_AttUNet2_masks_seg_cvppp/re__176_s1.5_m1.0_AttUNet2_masks_seg_cvppp_24300.dat 
