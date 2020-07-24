cat $0
python main.py --env re__256_s2.0_m0.0_r12_28 --gpu-id 0 1 2 3 4 5 6  --workers 7 \
--lbl-agents 0 --num-steps 6 --max-episode-length 6 --reward seg \
--model AttUNet2 --out-radius 12 28 --use-masks --size 256 256 \
--features 32 64 128 256 512 --entropy-alpha 0.05 \
--downsample -1 --data cvppp --in-radius 0.8 --log-period 50 \
--lr 1e-4 --fgbg-ratio 0.3 --st-fgbg-ratio 0.3 
--mer_w 0.0 --spl_w 2.0 \
--save-period 50 --minsize 12 \
--multi 1 \
--dilate-fac 2 \
--log-dir logs/Fer2019/Ablation_study/ \
