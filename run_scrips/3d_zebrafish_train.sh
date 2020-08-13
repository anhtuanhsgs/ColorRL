cat $0
python main.py --env 0803_6 \
--gpu-id  3 4 5 6 7  --workers 5 --valid-gpu 0 \
--num-steps 5 --max-episode-length 5 --max-temp-steps 5 \
--reward seg --model UNet3D --out-radius 12 --use-masks \
--size 96 96 96 --features 16 32 64 128 256 \
--entropy-alpha 0.05 --downsample 3 \
--data zebrafish3D --in-radius 1.1 --lr 1e-5 \
--fgbg-ratio 0.5 --st-fgbg-ratio 0.5 \
--mer_w 1.0 --spl_w 1.5 \
--split prox \
--save-period 50 --log-period 50 \
--log-dir logs/Aug2020/ --save-model-dir logs/Aug2020/trained_models/ \
--minsize 12 \
--dilate-fac 4 \
--rew-drop 7 --rew-drop-2 9 \
--load logs/Aug2020/trained_models/zebrafish3D/0803_5_UNet3D/7000.dat \
