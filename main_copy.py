from __future__ import print_function, division
import os, sys, glob, time
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import *
from utils import get_cell_prob, get_data
from models.models import *
from train_copy import train_func
from test_copy import test_func
from natsort import natsorted

from Utils.img_aug_func import *
from Utils.utils import *
from skimage.measure import label
from shared_optim import SharedRMSprop, SharedAdam
from models.models import *

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
        "split_radius": args.split_radius,
        "merge_radius": args.merge_radius,
        "spl_w": args.spl_w,
        "mer_w": args.mer_w,

        "merge_speed": args.merge_speed,
        "split_speed": args.split_speed,
        "reward": args.reward,
        "use_lbl": args.use_lbl,
        "use_masks": args.use_masks,
        "DEBUG": args.DEBUG,
        "dilate_fac": args.dilate_fac,

        "tempT": args.max_temp_steps,

        "lowres": args.lowres,
        "T0": args.T0,
        "rew_drop": args.rew_drop,
    }

    if env_conf ["3D"]:
        env_conf ["size"] = [args.size[2], args.size[0], args.size[1]]

    print ("DEBUG", env_conf ["size"])

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

#     args.log_dir += args.data + "/" + args.env + "/"
#     args.save_model_dir += args.data + "/" + args.env + "/"
#     create_dir (args.save_model_dir)
#     create_dir (args.log_dir)
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
        path_valid = "Data/MoNuSeg2018/train/"
        path_test = "Data/MoNuSeg2018/train/"
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
            print (raw_test_upsize.shape)
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
            raw = [raw [i] [0:0+size[2], 0:0+size[0], 0:0+size[1]] for i in range (len (raw))]
            gt_lbl = [gt_lbl [i] [0:0+size[2], 0:0+size[0], 0:0+size[1]] for i in range (len (raw))]
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
        raw = resize_volume (raw, size, ds)
        gt_lbl = resize_volume (gt_lbl, size, ds)
        raw_valid = resize_volume (raw_valid, size, ds)
        gt_lbl_valid = resize_volume (gt_lbl_valid, size, ds)

        if raw_test is not None:
            raw_test = resize_volume (raw_test, size, ds)
        if args.testlbl:
            gt_lbl_test = resize_volume (gt_lbl_test, size, ds)
            
    io.imsave ("tmp_gt.tif", gt_lbl [0])
            
    if (args.deploy):
        return raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test, raw_test_upsize, gt_lbl_test_upsize
    else:
        return raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test

def main (scripts, args):
    args.scripts = scripts
    
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        

    if (args.deploy):
        raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test, raw_test_upsize, gt_lbl_test_upsize = setup_data(args)
    else:
        raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test = setup_data (args)      

    env_conf = setup_env_conf (args)

    print (env_conf ["observation_shape"])

    shared_model = get_model (args, args.model, env_conf ["observation_shape"], args.features, 
                        atrous_rates=args.atr_rate, num_actions=2, split=args.data_channel, 
                        multi=args.multi)

    print ("WTFFFFFFFFF")
    
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
    
    train_func (0, args, shared_model, optimizer, env_conf, [raw, gt_lbl])
        

