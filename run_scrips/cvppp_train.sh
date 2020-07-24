cat $0
python main.py --env auto_wswm_1 \
--gpu-id  0 1 2 3 4 5 6 7  --workers 8 --valid-gpu 0 \
--lbl-agents 0 --num-steps 6 --max-episode-length 6 --reward seg \
--model AttUNet2 --out-radius 12 28 --use-masks --size 256 256 \
--features 16 32 64 128 256 --entropy-alpha 0.05 \
--downsample -1 --data cvppp --in-radius 0.8 --log-period 50 \
--lr 1e-4 --fgbg-ratio 0.3 --st-fgbg-ratio 0.3 \
--spl_w 2.40 --mer_w 0.60 --wctrl s2m \
--split prox \
--save-period 50 --minsize 12 \
--log-dir logs/July2020/ --save-model-dir logs/July2020/trained_models/ \
--dilate-fac 2 \
--rew-drop 20 --rew-drop-2 1 \
--wctrl-schedule 2000 4000 6000 8000 10000 12000 14000 \
