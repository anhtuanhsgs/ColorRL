pythonn main.py --gpu-id 0 1 2 3 --workers 8
python main.py --env EM_env_attempt5 --gpu-id 0 1 2 3 4 5 6 7 --workers 16 --reward density --model UNetLstm --radius 96 --use-lbl --size 128 128 --hidden-feat 8 --one-step 12
 python main.py --env EM_env_attempt4 --gpu-id 0 1 2 3 4 5 6 7 --workers 16 --num-steps 3 --max-episode-length 3 --reward density --model UNetLstm --radius 96 --use-lbl --size 128 128 --hidden-feat 128


python main.py --env EM_env_attempt8 --gpu-id 0 1 2 3 4 5 6 7 --workers 12 \
	--num-steps 4 --max-episode-length 4 \
	--reward density --model FCN_GRU --radius 96 --use-lbl --size 160 160 \
	--hidden-feat 128 --features 64 64 128 128 \

python main.py --env EM_env_attempt8 --gpu-id 0 1 2 3 4 5 6 7 --workers 16 \
	--num-steps 4 --max-episode-length 4 --reward density --model UNetGRU \
	--radius 48 --use-lbl --size 128 128 --hidden-feat 48 \
	--log-period 10 --features 16 32 64 128 --speed 1 --downsample

