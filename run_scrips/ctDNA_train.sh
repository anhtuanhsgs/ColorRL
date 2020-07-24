cat ctDNA_train.sh
python main.py --env origin --gpu-id 0 1 2 3  --workers 8 --lbl-agents 0 \
--num-steps 4 --max-episode-length 4 --reward seg --model AttUNet3 --out-radius 32 \
--use-masks --size 512 512 --log-period 10 --features 64 128 128 256 --entropy-alpha 0.05 \
--downsample 2 --data ctDNA --in-radius 0.8 --log-period 10 --lr 1e-4 --fgbg-ratio 0.1 \
--st-fgbg-ratio 0.2 --mer_w 1.0 --spl_w 1.5 --save-period 50 --minsize 12 \
--log-dir logs/Jan2019 \
