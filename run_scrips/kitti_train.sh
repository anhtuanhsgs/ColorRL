cat kitti_train.sh

python main.py --data kitti --train-log-period 24 \
--env kitti_train_3_CNN6_03112019 \
--gpu-id 0 1 2 3 --workers 8 --lbl-agents 0 \
--downsample -1 --size 160 480 --minsize 12 \
--num-steps 4 --max-episode-length 4 \
--reward seg --entropy-alpha 0.05 --use-masks  \
--features 16 32 64 128 256 512 --model AttUNet2 \
--out-radius 25 --in-radius 1 \
--fgbg-ratio 0.3 --st-fgbg-ratio 0.3 \
--mer_w 1.5 --spl_w 2 \
--lr 1e-4 --log-period 10 --save-period 50 \


#--load trained_models/kitti/kitti_train_2_CNN6_03112019_AttUNet2_masks_seg_kitti/kitti_train_2_CNN6_03112019_AttUNet2_masks_seg_kitti_100.dat
