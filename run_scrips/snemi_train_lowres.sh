cat snemi_train_lowres.sh 
python -W ignore main.py --env lowres2 --gpu-id 0 1 2 3 4 5 6 7 --workers 8 --lbl-agents 0 --valid-gpu 0 \
--num-steps 6 --max-episode-length 6 --reward seg --model AttUNet2 --out-radius 8 28 \
--use-masks --size 320 320 --log-period 10 --features 32 64 64 128 128 256 --entropy-alpha 0.05 \
--downsample 1 --data snemi --in-radius 0.9 --lr 1e-4 \
--fgbg-ratio 0.6 --st-fgbg-ratio 0.5 --mer_w 1.0 --spl_w 1.5 --save-period 50 --minsize 20 \
--lowres \

