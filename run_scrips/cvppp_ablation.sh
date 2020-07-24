cat $0
python main.py --env radius_28_28 --gpu-id 1 2 3 4 5 6 --workers 3 --valid-gpu 0 \
--lbl-agents 0 --num-steps 4 --max-episode-length 4 --reward seg \
--model AttUNet2 --out-radius 28 28 --use-masks --size 176 176 \
--features 32 64 128 256 512 --entropy-alpha 0.06 \
--downsample -1 --data cvppp --in-radius 0.8 --log-period 1 \
--lr 1e-4 --fgbg-ratio 0.3 --st-fgbg-ratio 0.3 \
--spl_w 1.0 --mer_w 1.0 --save-period 99999999 --minsize 12 \
--multi 1 \
--dilate-fac 2 \
--DEBUG \
--no-aug \
--log-dir logs/Mar2020/Ablation_study/ \
