from models.models import *
import argparse
import torch
import os, sys, glob, time
import skimage.io as io

def deploy (model, args, gpu_id, data):
    save_dir = "deploy/" + args.data + '/' + 'test/'
    os.makedirs (save_dir, exist_ok=True)
    raws, lbls = data

    with torch.no_grad ():
        with torch.cuda.device (gpu_id):
            model = model.cuda ()

    proc_times = []

    for idx, img in enumerate (raws):
        with torch.no_grad ():
            with torch.cuda.device (gpu_id):
                current_time = time.time ()
                if args.data_channel == 1:
                    size = tuple (img.shape)
                else:
                    size = tuple (img.shape [:2])

                done_mask = np.zeros ((1,) + size)
                label_mask = np.zeros ((args.max_episode_length,) + size)

                if args.data_channel == 1:
                    obs = np.concatenate ([img[None], done_mask, label_mask], 0)
                elif args.data_channel == 3:
                    img_trans = np.transpose (img, [2,0,1])
                    obs = np.concatenate ([img_trans.astype (np.float32), done_mask, label_mask], 0)

                obs_t = torch.tensor (obs, dtype=torch.float32).cuda () [None] / 255.0

                label_pred = np.zeros (size, dtype=np.int32)

                for i in range (args.max_episode_length):
                    # Update done mask
                    if i == args.max_episode_length - 1:
                        obs_t [0, args.data_channel] = 1.0

                    # Get new action
                    value, logit = model (obs_t)
                    prob = F.softmax (logit, dim=1)
                    action = prob.max (1)[1][0]
                    # Update state
                    obs_t [0, args.data_channel + 1 + i] = action.type (torch.float32) * 2 - 1
                    action = action.data.cpu ().numpy ()
                    label_pred += action * (2 ** i)

        delta_time = time.time () - current_time 
        print ("Done :", idx, "\tTime: ", delta_time)
        proc_times += [delta_time]
        io.imsave (save_dir + "/raw_" + str (idx) + ".tif", img);
        io.imsave (save_dir + "/" + str (idx) + ".tif", label_pred);
    file = open (save_dir + "/avg_time.txt", "w")
    file.write ("avg_time: " + str(np.mean (proc_times)))
    file.close ()
    print ("avg_time: ", np.mean (proc_times))

