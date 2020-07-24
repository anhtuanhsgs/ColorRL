cat $0
python main.py --env test --gpu-id 0  --workers 0 \
--lbl-agents 0 --num-steps 6 --max-episode-length 6 --reward seg \
--model AttUNet2 --out-radius 12 28 --use-masks --size 176 176 \
--features 32 64 128 256 512 --entropy-alpha 0.05 \
--downsample -1 --data cvppp --in-radius 0.8 --log-period 50 \
--lr 1e-4 --fgbg-ratio 0.3 --st-fgbg-ratio 0.3 \
--spl_w 1.5 --mer_w 1.0 --save-period 50 --minsize 12 \
--multi 1 \
--dilate-fac 2 \
--log-dir logs/ \
--deploy \
--load trained_models/cvppp/test_AttUNet2_masks_seg_cvppp/test_AttUNet2_masks_seg_cvppp_300.dat 
