from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import normal  # , pi
import copy

class Agent (object):
    def __init__ (self, model, env, args, state, rank=0):
        self.args = args

        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        
        self.eps_len = 0
        self.done = True
        self.info = None
        self.reward = 0

        self.t_lbl = None
        self.t_gt_lbl = None

        self.gpu_id = -1
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []
        self.cell_probs = []
        self.probs = []
        
        

    def action_lbl_rand (self, lbl, action, eps):
        val_list = np.unique (lbl)
        ret = np.zeros_like (action)
        if self.env.config ["lowres"]:
            lbl = lbl [::2, ::2]

        for val in val_list:
            if (val == 0):
                continue
            single_cell_map = (lbl == val)

            #DEBUG
            # if val == 0:
            #     continue
            ###############

            # pixels_list = np.where (single_cell_map)
            # rand_index = self.env.rng.randint (len (pixels_list [0]))
            # color_val = action [pixels_list [0][rand_index], pixels_list [1][rand_index]]
            # ret += single_cell_map * color_val

            single_cell_area = np.count_nonzero (single_cell_map)
            action_tmp = action * single_cell_map
            action_1_count = np.count_nonzero (action_tmp)
            ratio = action_1_count / (single_cell_area + 1)
            ratio = np.clip (ratio, eps, 1.0-eps)
            sample = self.env.rng.rand ()
            # if (sample < ratio):
            if (self.env.rng.rand () < 0.2):
                ret += single_cell_map
        
        self.action = ret

        ret = torch.from_numpy (ret [::]).long ().unsqueeze(0).unsqueeze(0)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                ret = ret.cuda()

        return ret

    #Operation on GPU
    def action_lbl_rand_gpu (self, lbl, action):
        with torch.no_grad ():
            with torch.cuda.device (self.gpu_id):
                val_list = self.lbl_list
                # val_list = torch.unique (lbl)
                ret = torch.zeros_like (action, requires_grad=False)

                for val in val_list:
                    if (val == 0):
                        continue
                    single_cell_map = (lbl == val).int ()
                    single_cell_area = single_cell_map.nonzero ().size (0)
                    action_tmp = action * single_cell_map
                    action_1_count = action_tmp.nonzero ().size (0)
                    ratio = action_1_count / single_cell_area
                    ratio = np.clip (ratio, 0.1, 0.9)
                    sample = self.env.rng.rand ()
                    if (sample < ratio):
                        ret += single_cell_map
        self.action = ret
        return ret

    def fetch_lbl_gpu (self):
        with torch.cuda.device (self.gpu_id):
            self.lbl_list = np.unique (self.env.gt_lbl).astype (np.int32).tolist ()
            self.t_lbl = torch.tensor (self.env.lbl, dtype=torch.int32, requires_grad=False).cuda ()
            self.t_gt_lbl = torch.tensor (self.env.gt_lbl, dtype=torch.int32, requires_grad=False).cuda ()

    def action_train_gpu (self, use_max=False, use_lbl=False):
        if self.args.lstm_feats:
            value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        elif "GRU" in self.args.model:
            value, logit, self.hx = self.model((Variable(
                self.state.unsqueeze(0)), self.hx))
        elif "EX" in self.args.model:
            value, logit, cell_prob = self.model (Variable(self.state.unsqueeze(0)))
            self.cell_probs.append (cell_prob)
        else:
            value, logit = self.model (Variable(self.state.unsqueeze(0)))

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        
        prob_tp = prob.permute (0, 2, 3, 1)
        log_prob_tp = log_prob.permute (0, 2, 3, 1)
        distribution = torch.distributions.Categorical (prob_tp)
        shape = prob_tp.shape
        if not use_max:
            action_tp = distribution.sample ().reshape (1, shape[1], shape[2], 1).int ()
            action = action_tp.permute (0, 3, 1, 2)
            self.t_action = action [0][0]

            if use_lbl:
                self.t_action = self.action_lbl_rand_gpu (self.t_gt_lbl, self.t_action)
            self.t_new_lbl = self.t_lbl + self.t_action * (2 ** self.env.step_cnt)    

            log_prob = log_prob.gather(1, Variable(action.long ()))
            state, self.reward, self.done, self.info = self.env.step_g(
                self.t_action, self.t_lbl, self.t_new_lbl, self.t_gt_lbl, self.t_density, self.gpu_id)

        if not use_max:
            self.state = torch.from_numpy(state).float()

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        # self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward [None][None])
        return self


    def action_train (self, use_max=False, use_lbl=False, eps=0.99):
        if self.args.lstm_feats:
            value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        elif "GRU" in self.args.model:
            value, logit, self.hx = self.model((Variable(
                self.state.unsqueeze(0)), self.hx))
        elif "EX" in self.args.model:
            value, logit, cell_prob = self.model (Variable(self.state.unsqueeze(0)))
            self.cell_probs.append (cell_prob)
        else:
            value, logit = self.model (Variable(self.state.unsqueeze(0)))

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        
        if self.env.is3D:
            prob_tp = prob.permute (0, 2, 3, 4, 1)
            log_prob_tp = log_prob.permute (0, 2, 3, 4, 1)
        else:
            prob_tp = prob.permute (0, 2, 3, 1)
            log_prob_tp = log_prob.permute (0, 2, 3, 1)
        distribution = torch.distributions.Categorical (prob_tp)
        # distribution = torch.distributions.Categorical (torch.clamp (prob_tp, 0.05, 0.95))
        shape = prob_tp.shape
        if not use_max:
            if self.env.is3D:
                action_tp = distribution.sample ().reshape (1, shape[1], shape[2], shape[3], 1)
                action = action_tp.permute (0, 4, 1, 2, 3)
            else:
                action_tp = distribution.sample ().reshape (1, shape[1], shape[2], 1)
                action = action_tp.permute (0, 3, 1, 2)
            self.action = action.cpu().numpy() [0][0]

            if use_lbl:
                action = self.action_lbl_rand (self.env.gt_lbl, self.action, eps)

            log_prob = log_prob.gather(1, Variable(action))
            state, self.reward, self.done, self.info = self.env.step(
                self.action)

        if not use_max:
            self.state = torch.from_numpy(state).float()

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        # self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward [None][None])
        return self

    def action_test (self):
        with torch.no_grad():
            if self.args.lstm_feats:
                if self.done:
                    if self.gpu_id >= 0:
                        with torch.cuda.device (self.gpu_id):
                            self.cx, self.hx = self.model.lstm.init_hidden (batch_size=1, use_cuda=True)
                    else:
                        self.cx, self.hx = self.model.lstm.init_hidden (batch_size=1, use_cuda=False)
                else:
                    self.cx = Variable (self.cx)
                    self.hx = Variable (self.hx)
                value, logit, (self.hx, self.cx) = self.model((Variable (self.state.unsqueeze(0)), (self.hx, self.cx)))
            elif "GRU" in self.args.model:
                if self.done:
                    if self.gpu_id >= 0:
                        with torch.cuda.device (self.gpu_id):
                            self.hx = self.model.gru.init_hidden (batch_size=1, use_cuda=True)
                    else:
                        self.hx = self.model.gru.init_hidden (batch_size=1, use_cuda=False)
                else:
                    self.hx = Variable (self.hx)
                value, logit, self.hx = self.model((Variable (self.state.unsqueeze(0)), self.hx))
            elif "EX" in self.args.model:
                value, logit, cell_prob = self.model (Variable (self.state.unsqueeze(0)))
                self.cell_probs.append (cell_prob)
            else:
                value, logit = self.model(Variable (self.state.unsqueeze(0)))
            
        prob = F.softmax (logit, dim=1)
        self.probs.append (prob.data.cpu ().numpy () [0][1])

        action = prob.max (1)[1].data.cpu ().numpy ()
        state, self.reward, self.done, self.info = self.env.step (action [0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device (self.gpu_id):
                self.state = self.state.cuda ()
        self.rewards.append (self.reward)
        # print ("action test", self.rewards)
        self.actions.append (action [0])
        self.eps_len += 1
        return self

    def clear_actions (self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []
        self.cell_probs = []
        self.probs = []
        return self

