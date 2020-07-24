cat cvppp_deploy.sh
python main.py --env deploy --gpu-id 0  --workers 0 \
--lbl-agents 0 --num-steps 6 --max-episode-length 6 --reward seg \
--model AttUNet2 --out-radius 12 28 --use-masks --size 176 176 \
--features 32 64 128 256 512 --entropy-alpha 0.05 \
--downsample -1 --data cvppp --in-radius 0.8 --log-period 50 \
--lr 1e-4 --fgbg-ratio 0.3 --st-fgbg-ratio 0.3 \
--spl_w 1.6 --mer_w 1.0 --save-period 50 --minsize 12 \
--multi 1 \
--dilate-fac 2 \
--log-dir logs/Fer2019/Ablation_study/ \
--deploy \
--load trained_models/cvppp/re__176_s1.75_m0.25_AttUNet2_masks_seg_cvppp/re__176_s1.75_m0.25_AttUNet2_masks_seg_cvppp_2250.dat 
#--load trained_models/cvppp/re__176_s2.0_m0.0_r12_28_AttUNet2_masks_seg_cvppp/re__176_s2.0_m0.0_r12_28_AttUNet2_masks_seg_cvppp_14200.dat 
#--load trained_models/cvppp/re__176_s1.75_m0.25_AttUNet2_masks_seg_cvppp/re__176_s1.75_m0.25_AttUNet2_masks_seg_cvppp_2250.dat 
#--load trained_models/cvppp/re__176_s1.75_m0.25_AttUNet2_masks_seg_cvppp/re__176_s1.5_m0.5_AttUNet2_masks_seg_cvppp_12400.dat 
#--load trained_models/cvppp/re__176_s1.0_m1.0_AttUNet2_masks_seg_cvppp/re__176_s1.0_m1.0_AttUNet2_masks_seg_cvppp_23500.dat 
#--load trained_models/cvppp/re__176_s0.25_m1.75_r12_28_AttUNet2_masks_seg_cvppp/re__176_s0.25_m1.75_r12_28_AttUNet2_masks_seg_cvppp_14100.dat 
#--load trained_models/cvppp/re__176_s0.5_m1.0_AttUNet2_masks_seg_cvppp/re__176_s0.5_m1.0_AttUNet2_masks_seg_cvppp_6200.dat \
