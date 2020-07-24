cat $0
python main.py --env July_002 \
--gpu-id  1 2 3 4 5 6 7  --workers 7 --valid-gpu 0 \
--num-steps 5 --max-episode-length 5 --max-temp-steps 4 \
--reward seg --model UNet3D --out-radius 12 --use-masks \
--size 128 128 64 --features 8 16 32 64 128  \
--entropy-alpha 0.05 --downsample 3 \
--data zebrafish3D --in-radius 1.1 --lr 1e-4 \
--fgbg-ratio 0.3 --st-fgbg-ratio 0.3 \
--mer_w 1.0 --spl_w 3.0 \
--split ins \
--save-period 50 --log-period 50 \
--log-dir logs/July2020/ --save-model-dir logs/July2020/trained_models/ \
--minsize 12 \
--dilate-fac 3 \
--rew-drop 3 --rew-drop-2 7 \


