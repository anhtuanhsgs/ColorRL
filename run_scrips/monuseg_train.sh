cat $0
python main.py --env rew_seg_drop \
--gpu-id 2 3 4 5 6 7 4 6  --workers 8 --valid-gpu 1 \
--num-steps 5 --max-episode-length 5 --max-temp-steps 5 \
--reward seg --model UNet2D --out-radius 12 --use-masks \
--size 256 256  --features  8 16 32 64 128  \
--entropy-alpha 0.05 --downsample 2 \
--data mnseg2018 --in-radius 1.1 --lr 1e-4 \
--fgbg-ratio 0.5 --st-fgbg-ratio 0.5 \
--mer_w 1.0 --spl_w 2.0 \
--save-period 50 --log-period 50 \
--log-dir logs/July2020/ --save-model-dir logs/July2020/trained_models/ \
--minsize 12 \
--dilate-fac 3 \
--rew-drop 6 \

