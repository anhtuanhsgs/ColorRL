cat cvppp_train.sh
python main.py --env re__176_s0.25_m1.75_r12_28 --gpu-id 0 1 3 4 5 6 7  --workers 7 \
--lbl-agents 0 --num-steps 6 --max-episode-length 6 --reward seg \
--model AttUNet2 --out-radius 12 28 --use-masks --size 176 176 \
--features 32 64 128 256 512 --entropy-alpha 0.05 \
--downsample -1 --data cvppp --in-radius 0.8 --log-period 10 \
--lr 1e-4 --fgbg-ratio 0.3 --st-fgbg-ratio 0.3 \
--spl_w 0.25 --mer_w 1.75 --save-period 50 --minsize 12 \
--multi 1 \
--dilate-fac 2 \
--log-dir logs/Fer2019/Ablation_study/ \
