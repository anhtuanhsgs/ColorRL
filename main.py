from __future__ import print_function, division
import os, sys, glob, time
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import *
from utils import get_cell_prob, get_data
from train import train_func
from test import test_func
from natsort import natsorted

from Utils.img_aug_func import *
from Utils.utils import *
from skimage.measure import label
from shared_optim import SharedRMSprop, SharedAdam
from models.models import *

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--env',
    default='EM_env',
    metavar='ENV',
    )

parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')

parser.add_argument(
    '--gamma',
    type=float,
    default=1,
    metavar='G',
    help='discount factor for rewards (default: 1)')

parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')

parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')

parser.add_argument(
    '--workers',
    type=int,
    default=8,
    metavar='W',
    help='how many training workers to use (default: 8)')

parser.add_argument(
    '--num-steps',
    type=int,
    default=5,
    metavar='NS',
    help='n in n-step learning A3C')

parser.add_argument(
    '--max-episode-length',
    type=int,
    default=5,
    metavar='M',
    help='Maximum number of coloring steps')

parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')

parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')

parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',)

parser.add_argument(
    '--load-model-dir',
    default='../trained_models/',
    metavar='LMD',
    help='folder to load trained models from')

parser.add_argument(
    '--save-model-dir',
    default='logs/trained_models/',
    metavar='SMD',
    help='folder to save trained models')

parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')

parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')

parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')

parser.add_argument(
    '--save-period',
    type=int,
    default=100,
    metavar='SP',
    help='Save period')

parser.add_argument(
    '--log-period',
    type=int,
    default=10,
    metavar='LP',
    help='Log period')

parser.add_argument (
    '--train-log-period',
    type=int,
    default=16,
    metavar='TLP',
)

parser.add_argument(
    '--shared-optimizer',
    action='store_true'
)


parser.add_argument (
    '--recur-feat',
    type=int,
    default=64,
    metavar='HF'
)

parser.add_argument (
    '--in-radius',
    type=float,
    default=[1.1],
    nargs='+',
    help='Splitting radii r'
)

parser.add_argument (
    '--out-radius',
    type=int,
    default=[16, 48, 96],
    nargs='+',
    help='Shrinking factors alpha'
)

parser.add_argument (
    '--eval-data',
    default='test',
    choices=['test', 'valid', 'train', 'all']
)

parser.add_argument (
    '--merge_radius',
    type=int,
    default=[16, 48, 96],
    nargs='+'
)

parser.add_argument (
    '--split_radius',
    type=int,
    default=[16, 48, 96],
    nargs='+', 
)

parser.add_argument (
    '--merge_speed',
    type=int,
    default=[1, 2, 4],
    nargs='+'
)

parser.add_argument (
    '--split_speed',
    type=int,
    default=[1, 2, 4],
    nargs='+'
)

parser.add_argument (
    '--features',
    type=int,
    default= [32, 64, 128, 256],
    nargs='+', 
    help='Feature size Attention UNET'
)

parser.add_argument (
    '--size',
    type=int,
    default= [96, 96],
    nargs='+',
    help='Input processing size of agent, if processing size is smaller than size of images in the data, input will be cropped from the image',
)

parser.add_argument (
    '--entropy-alpha',
    type=float,
    default=0.5,
)

parser.add_argument (
    '--model',
    default='UNet',
    choices=["AttUNet", "ASPPAttUNet", "DeepLab", "ASPPAttUNet2", 
                        "AttUNet2", "AttUNet3", "GCNAttUNet", "UNet3D", "UNet2D"]
)

parser.add_argument (
    "--reward",
    default="normal",
    choices=["normal", "gaussian", "density", "seg"]
)

parser.add_argument (
    "--use-lbl",
    action="store_true"
)

parser.add_argument (
    "--use-masks",
    action="store_true"
)

parser.add_argument (
    '--downsample',
    type=int,
    default=1,
    help='Data down-sampling rate, -1 to have all images down-sampled to input processing size of agent',
)

parser.add_argument (
    '--DEBUG',
    action="store_true"
)

parser.add_argument (
    '--data',
    default='snemi',
    choices=['syn', 'snemi', 'voronoi', 'zebrafish', 'cvppp', 'cvppp_eval', 'zebrafish3D', 'dic-hela', 
            'sb2018', 'kitti', 'mnseg2018', "Cityscape", "256_cremi", "448_cremi", "cremi", "ctDNA"]
)
parser.add_argument (
    '--SEMI_DEBUG',
    action="store_true"
)

parser.add_argument (
    '--deploy',
    action='store_true',
    help='Enable for test set deployment',
)

parser.add_argument (
    '--fgbg-ratio',
    default=0.2,
    type=float,
)

parser.add_argument (
    '--st-fgbg-ratio',
    default=0.5,
    type=float,
)

parser.add_argument (
    '--seg-scale',
    action='store_true'
)

# parser.add_argument (
#     '--lbl-action-ratio',
#     type=float,
#     default=0.125
# )

parser.add_argument (
    '--lbl-agents',
    type=int,
    default=0
)

parser.add_argument (
    '--minsize',
    type=int,
    default=20,
)


parser.add_argument (
    '--spl_w',
    type=float,
    default=2,
    help='Splitting weight w_s',
)

parser.add_argument (
    '--mer_w',
    type=float,
    default=1,
    help='Merging weight w_s',
)

parser.add_argument (
    '--noisy',
    action='store_true',
)

parser.add_argument (
    '--lstm-feats',
    type=int,
    default=0,
)

parser.add_argument (
    '--valid-gpu',
    type=int,
    default=-1,
    help='Choose gpu for validation',
)

parser.add_argument (
    '--atr-rate',
    type=int,
    default= [1, 6, 12, 18],
    nargs='+',
    help='Attrous spatial pooling rates',
)

parser.add_argument (
    '--dilate-fac',
    type=int,
    default=2
)

parser.add_argument (
    '--lowres',
    action="store_true"
)

parser.add_argument (
    '--no-aug',
    action="store_true"
)

parser.add_argument (
    '--multi',
    type=int,
    default=1,
)

parser.add_argument (
    '--max-temp-steps',
    type=int,
    default=99,
)

parser.add_argument (
    '--T0',
    type=int,
    default=0
)

parser.add_argument (
    "--no-test",
    action="store_true"
)

parser.add_argument (
    '--rew-drop',
    type=int,
    default=0
)

parser.add_argument (
    '--rew-drop-2',
    type=int,
    default=0
)


parser.add_argument (
    '--split',
    default='ins',
    choices=['ins', 'prox',]
)

parser.add_argument (
    '--wctrl',
    default='non',
    choices=['non', 's2m']
)

parser.add_argument (
    '--wctrl-schedule',
    default=[2000, 4000, 6000, 8000, 10000, 12000, 14000],
    type=int,
    nargs='+',
)

parser.add_argument (
    '--exp-pool',
    default=0,
    type=int,
)

def setup_env_conf (args):
    env_conf = {
        "data": args.data,
        "T": args.max_episode_length,
        "size": args.size,
        "fgbg_ratio": args.fgbg_ratio,
        "st_fgbg_ratio": args.st_fgbg_ratio,
        "minsize": args.minsize,
        "no_aug": args.no_aug,

        "3D": "3D" in args.data,
        
        "in_radius": args.in_radius,
        "out_radius": args.out_radius,
        "spl_w": args.spl_w,
        "mer_w": args.mer_w,
        "split": args.split,

        "reward": args.reward,
        "use_lbl": args.use_lbl,
        "use_masks": args.use_masks,
        "DEBUG": args.DEBUG,
        "dilate_fac": args.dilate_fac,

        "tempT": args.max_temp_steps,

        "lowres": args.lowres,
        "T0": args.T0,
        "rew_drop": args.rew_drop,
        "rew_drop_2": args.rew_drop_2,

        "exp_pool": args.exp_pool,
    }

    if env_conf ["3D"]:
        env_conf ["size"] = [args.size[2], args.size[0], args.size[1]]

    env_conf ["observation_shape"] = [args.data_channel + 1] + env_conf ["size"]



    args.env += "_" + args.model
    env_conf ["data_chan"] = args.data_channel 
    if args.use_lbl:
        args.env += "_lbl"
        env_conf ["observation_shape"][0] += 1 #Raw, lbl, stop
    if args.use_masks:
        args.env += "_masks"
        env_conf ["observation_shape"][0] += env_conf ["T"]

    args.env += "_" + args.reward
    args.env += "_" + args.data

    args.log_dir += args.data + "/" + args.env + "/"
    args.save_model_dir += args.data + "/" + args.env + "/"
    create_dir (args.save_model_dir)
    create_dir (args.log_dir)
    return env_conf

 
def setup_data (args):
    path_test = None
    if args.data == 'syn':
        path_train = 'Data/syn/'
        path_valid = 'Data/syn/'
        args.data_channel = 1
        args.testlbl = True
    if args.data == 'snemi':
        path_train = 'Data/snemi/train/'
        path_valid = 'Data/snemi/test/'
        path_test = 'Data/snemi/test/'
        args.data_channel = 1
        args.testlbl = True
    if args.data == "zebrafish":
        path_train = "Data/Zebrafish/train/"
        path_valid = "Data/Zebrafish/valid/"
        path_test = "Data/Zebrafish/valid/"
        args.data_channel = 1
        args.testlbl = True
    if args.data == "zebrafish3D":
        path_train = "Data/Zebrafish3D/train/"
        path_valid = "Data/Zebrafish3D/train/"
        path_test = "Data/Zebrafish3D/train/"
        args.data_channel = 1
        args.testlbl = True
    if args.data == "cvppp":
        path_train = "Data/CVPPP_Challenge/train/"
        path_valid = "Data/CVPPP_Challenge/valid/"
        path_test = "Data/CVPPP_Challenge/test/"
        args.data_channel = 3
        args.testlbl = False
    if args.data == "cvppp_eval":
        path_train = "Data/CVPPP_Challenge/train/"
        path_valid = "Data/CVPPP_Challenge/train/"
        path_test = "Data/CVPPP_Challenge/test/"
        args.data_channel = 3
        args.testlbl = True
    if args.data == 'sb2018':
        path_train = "Data/ScienceBowl2018/train/"
        path_valid = "Data/ScienceBowl2018/train/"
        path_test = "Data/ScienceBowl2018/test/"
        args.data_channel = 3
        args.testlbl = False

    if args.data == 'kitti':
        path_train = "Data/kitti/train2/"
        path_valid = "Data/kitti/train2/"
        path_test = "Data/kitti/valid/"
        args.data_channel = 3
        args.testlbl = True

    if args.data == 'mnseg2018':
        path_train = "Data/MoNuSeg2018/train/"
        path_valid = "Data/MoNuSeg2018/test/"
        path_test = "Data/MoNuSeg2018/test/"
        args.data_channel = 3
        args.testlbl = True

    if args.data == "Cityscape":
        path_train = "../Data/cityscape/train/"
        path_test = "../Data/cityscape/valid/"
        path_valid = "../Data/cityscape/valid/"
        args.testlbl = True
        args.data_channel = 3
    if args.data == "256_cremi":
        path_train = "Data/Cremi/256/train/"
        path_test = "Data/Cremi/256/train/"
        path_valid = "Data/Cremi/256/test/"
        args.testlbl = True
        args.data_channel = 1
    if args.data == "448_cremi":
        path_train = "Data/Cremi/448/train/"
        path_test = "Data/Cremi/448/train/"
        path_valid = "Data/Cremi/448/test/"
        args.testlbl = True
        args.data_channel = 1


    if args.data == "dic-hela":
        path_train = "Data/DIC_HeLa/train/"
        path_test = "Data/DIC_HeLa/test/"
        path_valid = "Data/DIC_HeLa/test/"
        args.testlbl = True
        args.data_channel = 1

    if args.data == "ctDNA":
        path_train = "Data/ctDNA/train/"
        path_test = "Data/ctDNA/train/"
        path_valid = "Data/ctDNA/train/"
        args.testlbl = True
        args.data_channel = 3

    relabel = args.data not in ['cvppp', 'sb2018', 'kitti', 'mnseg2018', 'Cityscape', 'zebrafish', "cremi", "ctDNA", "256_cremi", "448_cremi", "zebrafish3D"]
    
    raw, gt_lbl = get_data (path=path_train, relabel=relabel)
    raw_valid, gt_lbl_valid = get_data (path=path_valid, relabel=relabel)
    
    raw_test = None
    gt_lbl_test = None
    if path_test is not None:
        raw_test, gt_lbl_test = get_data (path=path_test, relabel=relabel)


    raw_test_upsize = None
    gt_lbl_test_upsize = None
    if (args.deploy) and (path_test is not None):
        if args.eval_data == "test":
            raw_test_upsize = raw_test
            gt_lbl_test_upsize = gt_lbl_test
        elif args.eval_data == "valid":
            raw_test_upsize = raw_valid
            gt_lbl_test_upsize = gt_lbl_valid
        elif args.eval_data == "train":
            raw_test_upsize = raw
            gt_lbl_test_upsize = gt_lbl
        elif args.eval_data == "all":
            # raw_test_upsize = np.concatenate([raw, raw_valid, raw_test], axis=0)
            # gt_lbl_test_upsize = np.concatenate([gt_lbl, gt_lbl_valid, gt_lbl_test], axis=0)
            raw_test_upsize = np.concatenate([raw_valid, raw_test], axis=0)
            gt_lbl_test_upsize = np.concatenate([gt_lbl_valid, gt_lbl_test], axis=0)


    if (args.DEBUG):
        size = [args.size [i] * args.downsample for i in range (len (args.size))]
        if args.downsample == -1:
            size = raw[0].shape[0]
        if "3D" in args.data:
            if args.DEBUG:
                raw = [raw [i] [0:0+size[2], 0:0+size[0], 0:0+size[1]] for i in range (len (raw))]
                gt_lbl = [gt_lbl [i] [0:0+size[2], 0:0+size[0], 0:0+size[1]] for i in range (len (raw))]
            else:
                raw = [raw [i] for i in range (len (raw))]
                gt_lbl = [gt_lbl [i] for i in range (len (raw))]
        
        else:    
            raw = [raw [i] [0:0+size,0:0+size] for i in range (20, 21)]
            gt_lbl = [gt_lbl [i] [0:0+size,0:0+size] for i in range (20, 21)]
        raw_valid = np.copy (raw)
        gt_lbl_valid = np.copy (gt_lbl)

    if (args.SEMI_DEBUG):
        raw = raw [:1000]
        gt_lbl = gt_lbl [:1000]

    ds = args.downsample
    if args.downsample:
        size = args.size
        raw = resize_volume (raw, size, ds, "3D" in args.data)
        gt_lbl = resize_volume (gt_lbl, size, ds, "3D" in args.data)
        raw_valid = resize_volume (raw_valid, size, ds, "3D" in args.data)
        gt_lbl_valid = resize_volume (gt_lbl_valid, size, ds, "3D" in args.data)

        if raw_test is not None:
            raw_test = resize_volume (raw_test, size, ds, "3D" in args.data)
        if args.testlbl:
            gt_lbl_test = resize_volume (gt_lbl_test, size, ds, "3D" in args.data)

    if (args.deploy):
        return raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test, raw_test_upsize, gt_lbl_test_upsize
    else:
        return raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test

def main (scripts, args):
    scripts = " ".join (sys.argv[0:])
    args = parser.parse_args()
    args.scripts = scripts
    
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')

    if (args.deploy):
        raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test, raw_test_upsize, gt_lbl_test_upsize = setup_data(args)
    else:
        raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test = setup_data (args)

    env_conf = setup_env_conf (args)


    shared_model = get_model (args, args.model, env_conf ["observation_shape"], args.features, 
                        atrous_rates=args.atr_rate, num_actions=2, split=args.data_channel, 
                        multi=args.multi)

    manager = mp.Manager ()
    shared_dict = manager.dict ()
    if args.wctrl == "s2m":
        shared_dict ["spl_w"] = args.spl_w
        shared_dict ["mer_w"] = args.mer_w

    if args.load:
        saved_state = torch.load(
            args.load,
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()
    
    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []
    if not args.no_test:
        if raw_test is not None:
            if (args.deploy):
                p = mp.Process(target=test_func, args=(args, shared_model, env_conf, [raw_valid, gt_lbl_valid], (raw_test, gt_lbl_test, raw_test_upsize, gt_lbl_test_upsize, shared_dict)))
            else:
                p = mp.Process(target=test_func, args=(args, shared_model, env_conf, [raw_valid, gt_lbl_valid], (raw_test, gt_lbl_test), shared_dict))
        else:
            p = mp.Process(target=test_func, args=(args, shared_model, env_conf, [raw_valid, gt_lbl_valid], None, shared_dict))
        p.start()
        processes.append(p)
    
    time.sleep(0.1)

    for rank in range(0, args.workers):
        p = mp.Process(
            target=train_func, args=(rank, args, shared_model, optimizer, env_conf, [raw, gt_lbl], shared_dict))

        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

if __name__ == '__main__':
    scripts = " ".join (sys.argv[0:])
    args = parser.parse_args()
    main (scripts, args)
