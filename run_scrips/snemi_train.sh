cat $0
python main.py --env rew_seg_drop \
--gpu-id 1 2 3 4 5 6 7  --workers 8 --valid-gpu 7 \
--num-steps 5 --max-episode-length 5 --max-temp-steps 5 \
--reward seg --model AttUNet2 --out-radius 4 12 --use-masks \
--size 256 256  --features 16 32 64 128 256  \
--entropy-alpha 0.05 --downsample 2 \
--data snemi --in-radius 1.1 --lr 1e-4 \
--fgbg-ratio 0.5 --st-fgbg-ratio 0.5 \
--mer_w 1.0 --spl_w 1.0 \
--save-period 50 --log-period 50 \
--log-dir logs/July2020/ --save-model-dir logs/July2020/trained_models/ \
--minsize 12 \
--dilate-fac 2 \
--rew-drop 6 \
