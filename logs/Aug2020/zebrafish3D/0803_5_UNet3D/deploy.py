from models.models import *
import argparse
import torch
import os, sys, glob, time

parser = argparse.ArgumentParser(description='deploy')

parser.add_argument (
    '--model',
    default='UNet',
    choices=['UNet', 'FusionNetLstm', "FusionNet", "UNetLstm", "FCN_GRU", "UNetGRU", 
                "DilatedUNet", "UNetEX", "UNetFuse", "AttUNet", "ASPPAttUNet"]
)

parser.add_argument('--load', default=False, metavar='L', help='load a trained model')

parser.add_argument('--gpu-id', type=int, default=0)

parser.add_argument ('--features', type=int, default=[32, 64, 128, 256], nargs='+')



def inference (args, logger, model, tests, test_env, gpu_id, rng, iter):
    log_img = []
    # idxs = rng.choice (len (tests), 10)
    idxs = []
    # idxs.append (rng.randint (len (tests)))
    idxs.append (0)
    for i in range (min (len (tests), 33)):
        idxs.append ((idxs [-1] + 1) % len (tests))

    if args.data in ['cvppp']:
        resize = True
    else:
        resize = False

    for i in idxs:
        obs = test_env.set_sample (i, resize)
        done = False
        while (not done):
            with torch.no_grad ():
                with torch.cuda.device (gpu_id):
                    t_obs = torch.tensor (obs[None], dtype=torch.float32, device="cuda")
                    value, logit = model (t_obs)
                    prob = F.softmax (logit, dim=1)
                    action = prob.max (1)[1].data.cpu ().numpy ()

            obs, _, done, _ = test_env.step_inference (action [0])
        img = test_env.render ()
        log_img.append (img [:len(img)//2])

    log_img = np.concatenate (log_img, 0)
    log_info = {"test_samples": log_img}
    for tag, img in log_info.items ():
        img = img [None]
        logger.image_summary (tag, img, iter)

def setup_deploy_data (args):
    path_test = None
    if args.data == 'snemi':
        path_test = 'Data/snemi/test/'
        args.data_channel = 1
    if args.data == "zebrafish":
        path_test = "Data/Zebrafish/valid/"
        args.data_channel = 1
    if args.data == "cvppp":
        path_test = "Data/CVPPP_Challenge/test/"
        args.data_channel = 3
    if args.data == 'sb2018':
        path_train = "Data/ScienceBowl2018/train/"
        path_valid = "Data/ScienceBowl2018/train/"
        args.data_channel = 3
    if args.data == 'kitti':
        path_test = "Data/kitti/test/"
        args.data_channel = 3

    raw_test, _ = get_data (path=path_test, relabel=relabel)
    return raw_test

if __name__ == "__main__":
    args = parser.parse_args()
