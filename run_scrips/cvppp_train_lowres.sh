cat cvppp_train_lowres.sh
python main.py --env lowres --gpu-id 1 2 3 4 5 6 7 --workers 7 \
--lbl-agents 0 --num-steps 6 --max-episode-length 6 --reward seg \
--model AttUNet2 --out-radius 8 28 --use-masks --size 320 320 \
--features 32 64 64 128 128 256  --entropy-alpha 0.05 \
--downsample -1 --data cvppp --in-radius 0.8 --log-period 10 \
--lr 1e-4 --fgbg-ratio 0.2 --st-fgbg-ratio 0.5 --mer_w 1.0 \
--spl_w 1.3 --save-period 50 --minsize 12 \
--log-dir logs/Jan2019/ \
--lowres \
