from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import *
from models.models import *
from player_util import Agent
from torch.autograd import Variable
import time, os
import logging
from Utils.Logger import Logger
from Utils.utils import create_dir
from Utils.metrics import GetDices, DiffFGLabels, kitti_metric
from utils import setup_logger, clean_reindex, ScalaTracker
import numpy as np
import cv2
import skimage.io as io
from datetime import datetime


def inference (args, logger, model, tests, test_env, gpu_id, rng, iter, upsize_lbl=None):
    log_img = []
    render_imgs = []

    if args.data in ['cvppp_eval']:
        original_size = (512, 512)
    elif args.data in ['kitti']:
        original_size = (1224, 370)

    # indices testset
    if args.deploy:
        idxs = np.arange(len(tests)).tolist()
    else:
        idxs = []
        if (len (tests) <= 40):
            idxs.append (0)
        else:
            idxs.append (rng.randint (len (tests)))

        for _ in range (min (len (tests), 33)):
            idxs.append ((idxs [-1] + 1) % len (tests))

    # if args.data in ['cvppp']:
    #     resize = True
    # else:
    #     resize = False

    if args.deploy:
        ret = []
        # ---------- Define Evaluating Variables ----------
        # CVPPP
        bestDices = []
        FgBgDices = []
        diffFGs = []

        # Kitti
        MUCovs = []
        MWCovs = []
        AvgFPs = []
        AvgFNs = []

        rand_is = []
        # ----------End Define Evaluating Variables ----------

    eval_times = []

    for idx in idxs:
        # obs = test_env.set_sample (idx, resize)
        obs = test_env.set_sample (idx)
        # render_img = [np.repeat (np.expand_dims (test_env.raw, -1), 3, -1)]
        done = False
        if args.lstm_feats:
            with torch.cuda.device (gpu_id):
                cx, hx = model.lstm.init_hidden (batch_size=1, use_cuda=True)
        currTime = time.time ()
        while (not done):
            with torch.no_grad ():
                with torch.cuda.device (gpu_id):
                    t_obs = torch.tensor (obs[None], dtype=torch.float32, device="cuda")
                    if args.lstm_feats:
                        value, logit, (hx, cx) = model ((t_obs, (hx, cx)))
                    else:
                        value, logit = model (t_obs)
                    prob = F.softmax (logit, dim=1)
                    action = prob.max (1)[1].data.cpu ().numpy ()

            obs, _, done, _ = test_env.step_inference (action [0])
        #     render_img.append (test_env.pred_lbl2rgb(test_env.lbl))

        # render_img = np.concatenate (render_img, 1).astype (np.uint8)
        # render_imgs.append (render_img)
    
        if not "3D" in args.data:
            clean_reindex (test_env.lbl, args.minsize)    
        
        deltaTime = time.time () - currTime
        eval_times.append (deltaTime)
        # print ("eval time: ", deltaTime)
        if not args.deploy:
            img = test_env.render ()
            log_img.append (img [:len(img)//2])
            

        elif args.data in ['cvppp_eval', 'cvppp', "448_cremi", "256_cremi"]:
            # ret.append(test_env.lbl)
            # lbl_upsize = cv2.resize (test_env.lbl, original_size, interpolation=cv2.INTER_NEAREST)
            ret.append(test_env.lbl)

            # # ---------------------------------------- Evaluate ----------------------------------------
            # bestDice, FgBgDice, diffFG, MWCov, MUCov, AvgFP, AvgFN, rand_i = deploy_evaluate(lbl_upsize, upsize_lbl[idx])
            # # CVPPP
            # bestDices.append(bestDice)
            # FgBgDices.append(FgBgDice)
            # diffFGs.append(abs (diffFG))

            # # Kitti
            # MWCovs.append(MWCov)
            # MUCovs.append(MUCov)
            # AvgFPs.append(AvgFP)
            # AvgFNs.append(AvgFN)

            # rand_is.append(rand_i)
            # ---------------------------------------- End Evaluate ----------------------------------------
        elif args.data == 'cvppp':
            ret.append(test_env.lbl)

    print (eval_times)
    print ("Mean eval times: ", np.mean (eval_times))

    if not args.deploy:
        log_img = np.concatenate (log_img, 0)
        # render_imgs = np.concatenate (render_imgs, 0)
        # print (render_imgs.shape)
        # log_info = {"test_samples": log_img, "render_imgs": render_imgs}
        log_info = {"test_samples": log_img, }
        for tag, img in log_info.items ():
            img = img [None]
            logger.image_summary (tag, img, iter)
    else:
        print("\n------------------------------------------------")
        print((str(datetime.now()) + "\n"))
        ret = np.array (ret, dtype=np.int32)
        print (ret.shape)
        io.imsave ("deploy/" + "deploy_" + args.env + ".tif", ret)
        # print ("\nExport Images: ", "deploy/" + "deploy_" + args.data + ".tif")

        # if args.data in ['kitti', 'cremi', 'cvppp_eval']:
        #     print("Evaluate on "+ args.eval_data + " " + args.data)
        #     log_txt = open("deploy/" + "log_" + str(args.env) + ".txt", "w")
        #     log_info = {
        #         "bestDice": np.mean(bestDices),
        #         "FgBgDice": np.mean(FgBgDices),
        #         "diffFG": np.mean(diffFGs),
        #         "MWCov": np.mean(MWCovs),
        #         "MUCov": np.mean(MUCovs),
        #         "AvgFP": np.mean(AvgFPs),
        #         "AvgFN": np.mean(AvgFNs),
        #         "rand_i": np.mean(rand_is),
        #     }
        #     print (log_info)
        #     log_txt.write("##########################################" + "\n")
        #     log_txt.write(str(datetime.now()) + "\n")
        #     for k, v in log_info.items():
        #         log_txt.write(str(k) + ": " + str(v) + "\n")
        #         print(str(k) + ": " + str(v) + "\n")
        #     log_txt.write("##########################################" + "\n")
        #     log_txt.close()

        print("\nDone !")
        print("\n------------------------------------------------")

def evaluate (args, env):

    pred_lbl = clean_reindex (env.lbl, args.minsize)
    gt_lbl = env.gt_lbl

    bestDice, FgBgDice = GetDices (pred_lbl, gt_lbl)

    diffFG = DiffFGLabels (pred_lbl, gt_lbl)

    MWCov, MUCov, AvgFP, AvgFN  = kitti_metric(pred_lbl, gt_lbl)

    rand_i = adjusted_rand_index(gt_lbl,pred_lbl)

    return bestDice, FgBgDice, diffFG , MWCov, MUCov, AvgFP, AvgFN, rand_i

def deploy_evaluate (lbl, gt_lbl):
    pred_lbl = clean_reindex (lbl)

    bestDice, FgBgDice = GetDices (pred_lbl, gt_lbl)

    diffFG = DiffFGLabels (pred_lbl, gt_lbl)

    MWCov, MUCov, AvgFP, AvgFN  = kitti_metric(pred_lbl, gt_lbl)

    rand_i = adjusted_rand_index(gt_lbl,pred_lbl)

    return bestDice, FgBgDice, diffFG , MWCov, MUCov, AvgFP, AvgFN, rand_i


def test_func (args, shared_model, env_conf, datasets=None, tests=None):
    ptitle ('Valid agent')

    if args.valid_gpu < 0:
        gpu_id = args.gpu_ids [-1]
    else:
        gpu_id = args.valid_gpu
        
    env_conf ["env_gpu"] = gpu_id

    if not args.deploy:
        log = {}

        logger = Logger (args.log_dir)

        create_dir (args.log_dir + "models/")
        create_dir (args.log_dir + "tifs/")

        os.system ("cp *.py " + args.log_dir)
        os.system ("cp *.sh " + args.log_dir)
        os.system ("cp models/*.py " + args.log_dir + "models/")

        setup_logger ('{}_log'.format (args.env), r'{0}{1}_log'.format (args.log_dir, args.env))
        log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
            args.env))
        d_args = vars (args)
        env_conf_log = env_conf

    if tests is not None and "3D" not in args.data:
        if args.testlbl:
            test_env = EM_env (tests [0], env_conf, type="test", gt_lbl_list=tests[1])
        else:
            test_env = EM_env (tests [0], env_conf, type="test")

    if not args.deploy:
        for k in d_args.keys ():
            log ['{}_log'.format (args.env)].info ('{0}: {1}'.format (k, d_args[k]))
        for k in env_conf_log.keys ():
            log ['{}_log'.format (args.env)].info ('{0}: {1}'.format (k, env_conf_log[k]))

    torch.manual_seed (args.seed)

    if gpu_id >= 0:
        torch.cuda.manual_seed (args.seed)

    raw_list, gt_lbl_list = datasets
    env = EM_env (raw_list, env_conf, type="train", gt_lbl_list=gt_lbl_list)

    reward_sum = 0
    start_time = time.time ()
    num_tests = 0
    reward_total_sum = 0

    player = Agent (None, env, args, None)
    player.gpu_id = gpu_id
    player.model = get_model (args, args.model, env_conf ["observation_shape"], args.features, 
                            atrous_rates=args.atr_rate, num_actions=2, split=args.data_channel, 
                            gpu_id=gpu_id, multi=args.multi)

    player.state = player.env.reset ()
    player.state = torch.from_numpy (player.state).float ()
    
    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.model = player.model.cuda ()
            player.state = player.state.cuda ()
    player.model.eval ()

    flag = True
    if not args.deploy:
        create_dir (args.save_model_dir)

    recent_episode_scores = ScalaTracker (100)
    recent_FgBgDice = ScalaTracker (100)
    recent_bestDice = ScalaTracker (100)
    recent_diffFG = ScalaTracker (100)

    recent_MUCov = ScalaTracker (100)
    recent_MWCov = ScalaTracker (100)
    recent_AvgFP = ScalaTracker (100)
    recent_AvgFN = ScalaTracker (100)

    recent_rand_i = ScalaTracker (100)

    renderlist = []
    renderlist.append (player.env.render ())
    max_score = 0

    # ----------------------------------------- Deploy / Inference -----------------------------------------
    if args.deploy:
        with torch.cuda.device (gpu_id):
            player.model.load_state_dict (shared_model.state_dict ())

        # inference (args, None, player.model, tests [0], test_env, gpu_id, player.env.rng, len (tests [0]))
        if len(tests)==4:
            inference(args, None, player.model, tests[0], test_env, gpu_id, player.env.rng, len(tests[0]), tests[3])
        else:
            inference(args, None, player.model, tests[0], test_env, gpu_id, player.env.rng, len(tests[0]))

        return
    # ----------------------------------------- End Deploy / Inference -----------------------------------------

    merge_ratios = []
    split_ratios = []

    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.model.load_state_dict (shared_model.state_dict ())
            else:
                player.model.load_state_dict (shared_model.state_dict ())
            player.model.eval ()
            flag = False

        player.action_test ()
        reward_sum += player.reward.mean ()
        renderlist.append (player.env.render ()) 

        if player.done:
            flag = True
            num_tests += 1

            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests

            log ['{}_log'.format (args.env)].info (
                "VALID: Time {0}, episode reward {1}, num tests {4}, episode length {2}, reward mean {3:.4f}".
                format (
                    time.strftime ("%Hh %Mm %Ss", time.gmtime (time.time () - start_time)),
                    reward_sum, player.eps_len, reward_mean, num_tests))

            recent_episode_scores .push (reward_sum)

            if args.save_max and recent_episode_scores.mean () >= max_score:
                max_score = recent_episode_scores.mean ()
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        state_to_save = {}
                        state_to_save = player.model.state_dict ()
                        torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, 'best_model_' + args.env))

            if num_tests % args.save_period == 0:
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        state_to_save = player.model.state_dict ()
                        torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, args.env + '_' + str (num_tests)))

            if num_tests % args.log_period == 0:
                if tests is not None and not args.DEBUG and not "3D" in args.data:
                    inference (args, logger, player.model, tests [0], test_env, gpu_id, player.env.rng, num_tests)

                if (np.max (env.lbl) != 0 and np.max (env.gt_lbl) != 0) and not "3D" in args.data:
                    bestDice, FgBgDice, diffFG, MWCov, MUCov, AvgFP, AvgFN, rand_i = evaluate (args, player.env)

                    recent_FgBgDice.push (FgBgDice)
                    recent_diffFG.push (abs (diffFG))
                    recent_bestDice.push (bestDice)

                    recent_MWCov.push(MWCov)
                    recent_MUCov.push (MUCov)
                    recent_AvgFP.push (AvgFP)
                    recent_AvgFN.push (AvgFN)


                    recent_rand_i.push (rand_i)


                    log_info = {
                        "bestDice": recent_bestDice.mean (),
                        "FgBgDice": recent_FgBgDice.mean (),
                        "diffFG": recent_diffFG.mean (),
                        "MWCov": recent_MWCov.mean(),
                        "MUCov": recent_MUCov.mean (),
                        "AvgFP": recent_AvgFP.mean (),
                        "AvgFN": recent_AvgFN.mean (),
                        "rand_i": recent_rand_i.mean ()
                    }

                    for tag, value in log_info.items ():
                        logger.scalar_summary (tag, value, num_tests)
                else:
                    bestDice, FgBgDice, diffFG = 0, 0, 0
                    MWCov, MUCov, AvgFP, AvgFN = 0, 0, 0, 0
                    rand_i = 0

                print ("----------------------VALID SET--------------------------")
                print (args.env)
                print ("bestDice:", bestDice, "FgBgDice:", FgBgDice, "diffFG:", diffFG,
                       "MWCov:", MWCov, "MUCov:", MUCov, "AvgFP:", AvgFP, "AvgFN:", AvgFN,
                       "rand_i:", rand_i
                       )
                # print ("mean bestDice")
                print ("Log test #:", num_tests)
                print ("rewards: ", player.reward.mean ())
                print ("sum rewards: ", reward_sum)
                print ("#gt_values:", len (np.unique (player.env.gt_lbl)))
                print ("values:")
                values = player.env.unique ()
                print (np.concatenate ([values[0][None], values[1][None]], 0))
                print ("------------------------------------------------")

                log_img = np.concatenate (renderlist [::-1], 0)
                                
                if not "3D" in args.data:            
                    for i in range (3):
                        # if i == 0 and args.seg_scale:
                        #     player.probs.insert (0, player.env.scaler / 1.5)
                        # else:
                        player.probs.insert (0, np.zeros_like (player.probs [0]))
                    while (len (player.probs) - 3 < args.max_episode_length):
                        player.probs.append (np.zeros_like (player.probs [0]))

                        probslist = [np.repeat (np.expand_dims (prob, -1),3, -1) for prob in player.probs]
                        probslist = np.concatenate (probslist, 1)
                        probslist = (probslist * 256).astype (np.uint8, copy=False)
                        # log_img = renderlist [-1]
                        print (probslist.shape, log_img.shape)
                        log_img = np.concatenate ([probslist, log_img], 0)

                log_info = {"valid_sample": log_img}

                print (log_img.shape)
                io.imsave (args.log_dir + "tifs/" + str (num_tests) + "_sample.tif", log_img.astype (np.uint8))
                io.imsave (args.log_dir + "tifs/" + str (num_tests) + "_pred.tif", player.env.lbl.astype (np.uint8))
                io.imsave (args.log_dir + "tifs/" + str (num_tests) + "_gt.tif", player.env.gt_lbl.astype (np.int32))

                if args.seg_scale:
                    log_info ["scaler"] = player.env.scaler


                for tag, img in log_info.items ():
                    img = img [None]
                    logger.image_summary (tag, img, num_tests)

                if not args.deploy and "3D" not in args.data:
                    log_info = {
                        'mean_valid_reward': reward_mean,
                        '100_mean_reward': recent_episode_scores.mean (),
                        'split_ratio': player.env.split_ratio_sum.sum () / np.count_nonzero (player.env.gt_lbl),
                        'merge_ratio': player.env.merge_ratio_sum.sum () / np.count_nonzero (player.env.gt_lbl)
                    }

                    merge_ratios.append (player.env.merge_ratio_sum.sum () / np.count_nonzero (player.env.gt_lbl))
                    split_ratios.append (player.env.split_ratio_sum.sum () / np.count_nonzero (player.env.gt_lbl))

                    print ("split ratio: ", np.max (player.env.split_ratio_sum), np.min (player.env.split_ratio_sum))
                    print ("merge ratio: ", np.max (player.env.merge_ratio_sum), np.min (player.env.merge_ratio_sum))
                    
                    print ("merge ratio: ", merge_ratios)
                    print ("split ratio: ", split_ratios)

                    for tag, value in log_info.items ():
                        logger.scalar_summary (tag, value, num_tests)

            renderlist = []
            reward_sum = 0
            player.eps_len = 0
            
            player.clear_actions ()
            state = player.env.reset (player.model, gpu_id)
            renderlist.append (player.env.render ())

            time.sleep (15)
            player.state = torch.from_numpy (state).float ()
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()