from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import *
from utils import ensure_shared_grads, EspTracker
from models.models import *
from player_util import Agent
from torch.autograd import Variable
from Utils.Logger import Logger

import numpy as np
import time

def train_func (rank, args, shared_model, optimizer, env_conf, datasets=None):
    if args.deploy:
        return
    ptitle('Train {0}'.format(rank))
    print ('Start training agent: ', rank)
    
    if rank == 0:
#         logger = Logger (args.log_dir [:-1] + '_losses/')
        train_step = 0

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    env_conf ["env_gpu"] = gpu_id
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    raw_list, gt_lbl_list = datasets
    env = EM_env (raw_list, env_conf, type="train", gt_lbl_list=gt_lbl_list, seed=args.seed + rank)

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop (shared_model.parameters (), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam (shared_model.parameters (), lr=args.lr, amsgrad=args.amsgrad)

    player = Agent (None, env, args, None)
    player.gpu_id = gpu_id
    player.model = get_model (args, args.model, env.observation_space.shape, args.features, 
                            atrous_rates=args.atr_rate, num_actions=2, split=args.data_channel, gpu_id=gpu_id, multi=args.multi)
    player.state = player.env.reset ()
    player.state = torch.from_numpy (player.state).float ()

    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.state = player.state.cuda ()
            player.model = player.model.cuda ()
    player.model.train ()

    if rank == 0:
        eps_reward = 0
        pinned_eps_reward = 0

    while True:
        if gpu_id >= 0:
            with torch.cuda.device (gpu_id):
                player.model.load_state_dict (shared_model.state_dict ())
        else:
            player.model.load_state_dict (shared_model.state_dict ())
        
        if player.done:
            player.eps_len = 0

            if rank == 0:
                if train_step % args.train_log_period == 0 and train_step > 0:
                    print ("train: step", train_step, "\teps_reward", eps_reward)
                if train_step > 0:
                    pinned_eps_reward = player.env.sum_reward.mean ()
                    eps_reward = 0

            if args.lstm_feats:
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        player.cx, player.hx = player.model.lstm.init_hidden (batch_size=1, use_cuda=True)
                else:
                    player.cx, player.hx = player.model.lstm.init_hidden (batch_size=1, use_cuda=False)
        elif args.lstm_feats:
            player.cx = Variable (player.cx.data)
            player.hx = Variable (player.hx.data)

        for step in range(args.num_steps):
        
            if rank < args.lbl_agents:
                player.action_train (use_lbl=True)  
            else:
                player.action_train () 

            if rank == 0:
                eps_reward = player.env.sum_reward.mean ()
            if player.done:
                break

        if player.done:
            state = player.env.reset (player.model, gpu_id)
            player.state = torch.from_numpy (state).float ()
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()

        if "3D" in args.data:
            R = torch.zeros (1, 1, env_conf ["size"][0], env_conf ["size"][1], env_conf ["size"][2])
        else:
            R = torch.zeros (1, 1, env_conf ["size"][0], env_conf ["size"][1])

        if args.lowres:
            R = torch.zeros (1, 1, env_conf ["size"][0] // 2, env_conf ["size"][1] // 2)

        if not player.done:
            if args.lstm_feats:
                value, _, _ = player.model((Variable(player.state.unsqueeze(0)), (player.hx, player.cx)))
            else:
                value, _ = player.model(Variable(player.state.unsqueeze(0)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        
        if "3D" in args.data:
            gae = torch.zeros(1, 1, env_conf ["size"][0], env_conf ["size"][1], env_conf ["size"][2])
        else:
            gae = torch.zeros(1, 1, env_conf ["size"][0], env_conf ["size"][1])

        if args.rew_drop:
            keep_map = torch.tensor (player.env.keep_map)
        if args.lowres:
            gae = torch.zeros (1, 1, env_conf ["size"][0] // 2, env_conf ["size"][1] // 2)

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda ()
                if args.rew_drop:
                    keep_map = keep_map.cuda ()
        R = Variable(R)

        for i in reversed(range(len(player.rewards))):
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    reward_i = torch.tensor (player.rewards [i]).cuda ()
            else:
                reward_i = torch.tensor (player.rewards [i])

            R = args.gamma * R + reward_i
            if args.rew_drop:
                advantage = R - player.values[i]
                value_loss = value_loss + (0.5 * advantage * advantage * keep_map).mean ()
                delta_t = player.values[i + 1].data * args.gamma + reward_i - player.values[i].data
                gae = gae * args.gamma * args.tau + delta_t
            else:
                advantage = R - player.values[i]
                value_loss = value_loss + (0.5 * advantage * advantage).mean ()
                delta_t = player.values[i + 1].data * args.gamma + reward_i - player.values[i].data
                gae = gae * args.gamma * args.tau + delta_t
            if args.noisy:
                policy_loss = policy_loss - \
                    (player.log_probs[i] * Variable(gae)).mean ()
            else:    
                if args.rew_drop:
                    policy_loss = policy_loss - \
                        (player.log_probs[i] * Variable(gae) * keep_map).mean () - \
                        (args.entropy_alpha * player.entropies[i] * keep_map).mean ()
                else:    
                    policy_loss = policy_loss - \
                        (player.log_probs[i] * Variable(gae)).mean () - \
                        (args.entropy_alpha * player.entropies[i]).mean ()


        player.model.zero_grad ()
        sum_loss = (policy_loss + value_loss)

        curtime = time.time ()
        # print ("backward curtime:", curtime)
        sum_loss.backward ()
        # print ("backward done", time.time () - curtime)
        ensure_shared_grads (player.model, shared_model, gpu=gpu_id >= 0)
        
        curtime = time.time ()
        # print ("optim curtime:", curtime)
        optimizer.step ()
        # print ("optim done", time.time () - curtime)
        

        player.clear_actions ()

        if rank == 0:
            train_step += 1
            if train_step % args.log_period == 0 and train_step > 0:
                log_info = {
                    'train: value_loss': value_loss, 
                    'train: policy_loss': policy_loss, 
                    'train: eps reward': pinned_eps_reward,
                }

                if "EX" in args.model:
                    log_info ["cell_prob_loss"] = cell_prob_loss

#                 for tag, value in log_info.items ():
#                     logger.scalar_summary (tag, value, train_step)
