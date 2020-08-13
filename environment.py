import os, sys, glob, time, copy
from os import sys, path
import gym
import numpy as np
from collections import deque
from gym.spaces.box import Box

from skimage.measure import label, block_reduce
from skimage.morphology import disk
from skimage.morphology import binary_dilation
import skimage.io as io

from sklearn.metrics import adjusted_rand_score
from skimage.transform import resize as resize3D
from Utils.utils import *
from Utils.img_aug_func import *
import albumentations as A
import cv2
import random
from gym.spaces import Box, Discrete, Tuple
import matplotlib.pyplot as plt
from malis import rand_index 
from random import shuffle
from PIL import Image, ImageFilter
from utils import guassian_weight_map, density_map, malis_rand_index, malis_f1_score, adjusted_rand_index
from skimage.draw import line_aa
from misc.Voronoi import *
import time
from rewards import *

# python main.py --env EM_env_DEBUG_1 --gpu-id 0 1 2 3 4 5 6 7 --workers 12 --lbl-agents 2 \--num-steps 5 --max-episode-length 5 --reward normal --model DilatedUNet --merge_radius 16 --merge_speed 2 --split_radius 64 --split_speed 4  --use-lbl --size 128 128 --hidden-feat 2  --log-period 10 --features 32 64 128 256 --downsample 2 --data zebrafish

class General_env (gym.Env):
    def init (self, config):
        self.T = config ['T']
        self.tempT = config ["tempT"]
        self.size = config ["size"]

        if config ["use_lbl"]:
            self.observation_space = Box (0, 1.0, shape=[config["observation_shape"][0]] + self.size, dtype=np.float32)
        else:
            self.observation_space = Box (-1.0, 1.0, shape=[config["observation_shape"][0]] + self.size, dtype=np.float32)
        self.rng = np.random.RandomState(time_seed ())
        self.max_lbl = 2 ** (self .T) - 1
        self.pred_lbl2rgb = color_generator (self.max_lbl + 1)
        self.gt_lbl2rgb = color_generator (111)
        self.is3D = self.config ["3D"]
        if self.config ["exp_pool"] > 0:
            self.pool = []
            self.pool_capacity = self.config ["exp_pool"]
            self.pool_iter = 0
            self.pool_period = 10

    def seed (self, seed):
        self.rng = np.random.RandomState(seed)

    def aug (self, image, mask):

        if self.is3D:
            if not (self.size[1] == self.size[2] == self.size[0]):
                [image, mask] = FlipRev3D ([image, mask], self.rng)
                rotn = self.rng.randint (4)
                [image, mask] = [rotate3D (img, rotn) for img in [image, mask]]
            else:
                [image, mask] = RotFlipRev3D ([image, mask], self.rng)
            ret = {"image": image, "mask": mask}
            return ret ['image'], ret ['mask']
        
        if self.config ["data"] == "zebrafish":
            randomBrightness =  A.RandomBrightness (p=0.3, limit=0.1)
            RandomContrast = A.RandomContrast (p=0.1, limit=0.1)
        else:
            randomBrightness = A.RandomBrightness (p=0.7, limit=0.1)
            RandomContrast = A.RandomContrast (p=0.5, limit=0.1)

        if image.shape [-1] == 3:
            if self.config ["data"] in ["Cityscape", "kitti"]:
                aug = A.Compose([
                        A.HorizontalFlip (p=0.5),
                        A.OneOf([
                            A.ElasticTransform(p=0.9, alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
                     
                            A.OpticalDistortion(p=0.9, distort_limit=(0.2, 0.2), shift_limit=(0, 0), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),                 
                            ], p=0.7),
                        A.ShiftScaleRotate (p=0.7, shift_limit=0.2, rotate_limit=10, interpolation=cv2.INTER_NEAREST, scale_limit=(-0.4, 0.4), border_mode=cv2.BORDER_CONSTANT),
                        A.RandomBrightness (p=0.7, limit=0.5),
                        A.RandomContrast (p=0.5),
                        A.GaussNoise (p=0.5),
                        A.Blur (p=0.5, blur_limit=4),
                        ]
                    )
            else:
                aug = A.Compose([
                            A.HorizontalFlip (p=0.5),
                            A.VerticalFlip(p=0.5),              
                            A.RandomRotate90(p=0.5),
                            A.Transpose (p=0.5),
                            A.OneOf([
                                A.ElasticTransform(p=0.9, alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
                                A.GridDistortion(p=0.9, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
                                A.OpticalDistortion(p=0.9, distort_limit=(0.2, 0.2), shift_limit=(0, 0), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),                 
                                ], p=0.7),
                            A.ShiftScaleRotate (p=0.7, shift_limit=0.3, rotate_limit=180, interpolation=cv2.INTER_NEAREST, scale_limit=(-0.3, 0.5), border_mode=cv2.BORDER_CONSTANT),
                            A.CLAHE(p=0.3),
                            A.RandomBrightness (p=0.7, limit=0.5),
                            A.RandomContrast (p=0.5),
                            A.GaussNoise (p=0.5),
                            A.Blur (p=0.5, blur_limit=4),
                            ]
                        )
        else:
            aug = A.Compose([
                        A.HorizontalFlip (p=0.5),
                        A.VerticalFlip(p=0.5),              
                        A.RandomRotate90(p=0.5),
                        A.Transpose (p=0.5),
                        A.OneOf([
                            A.ElasticTransform(p=0.5, alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
                            A.GridDistortion(p=0.5, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
                            A.OpticalDistortion(p=0.5, distort_limit=(0.2, 0.2), shift_limit=(0, 0), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),                 
                            ], p=0.6),
                        A.ShiftScaleRotate (p=0.5, shift_limit=0.3, rotate_limit=180, interpolation=cv2.INTER_NEAREST, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT),
                        # A.CLAHE(p=0.3),
                        randomBrightness,
                        RandomContrast,
                        A.GaussNoise (p=0.5),
                        A.Blur (p=0.3, blur_limit=4),
                        ]
                    )
        if self.config ["DEBUG"] or self.config ["no_aug"]:
            aug = A.Compose ([])

        ret = aug (image=image, mask=mask)        

        return ret ['image'], ret ['mask']

    def highres_action (self, action):
        return cv2.resize (action, (self.size [1], self.size[0]), interpolation=cv2.INTER_NEAREST)

    def lowres_reward (self, reward):
        return block_reduce (reward, (2, 2), np.mean)

    def step_inference (self, action):
        if self.config ["lowres"]:
            action = self.highres_action (action)
        self.action = action
        
        self.new_lbl = self.lbl + action * (2 ** self.step_cnt)
        self.lbl = self.new_lbl
        done = False
        info = {}
        reward = np.zeros (self.size, dtype=np.float32)

        
        self.mask [self.step_cnt:self.step_cnt+1] += (2 * action - 1) * 255
        self.step_cnt += 1

        if self.step_cnt >= min (self.T, self.tempT):
            done = True

        if self.config ["lowres"]:
            reward = self.lowres_reward (reward)
        ret = (self.observation (), reward, done, info)
        return ret

    def step (self, action):

        if self.config ["lowres"]:
            action = self.highres_action (action)
        self.action = action
        
        self.new_lbl = self.lbl + action * (2 ** self.step_cnt)
        done = False

        self.mask [self.step_cnt:self.step_cnt+1] += (2 * action - 1) * 255
        info = {}

        if (self.step_cnt == 0):
            reward = self.first_step_reward ()
            self.lbl = self.new_lbl
            self.step_cnt += 1
            self.rewards.append (reward)    
            self.sum_reward += reward
            if self.config ["lowres"]:
                reward = self.lowres_reward (reward)
            ret = (self.observation (), reward, done, info)
            return ret

        reward = np.zeros (self.size, dtype=np.float32)

        # reward += self.foreground_reward (self.step_cnt>=self.T)
        reward += self.background_reward (False)
        
        split_reward = np.zeros (self.size, dtype=np.float32)
        merge_reward = np.zeros (self.size, dtype=np.float32)
        split_reward_inr = np.zeros (self.size, dtype=np.float32)

        merge_ratio = np.zeros (self.size, dtype=np.float32)
        split_ratio = np.zeros (self.size, dtype=np.float32)

        range_split = 2.0 * 2 * len (self.bdrs) * self.config ["spl_w"] 
        range_merge = 2.0 * 2 * len (self.inrs) * self.config ["mer_w"] 

        if self.config ["reward"] == "seg":
            scaler = None
            # print (len (self.bdrs [1]), len (self.bdrs [0]), len (np.unique (self.gt_lbl)), len (self.segs), len (self.inrs))
            # while (True):
            #     pass
            for i in range (len (self.bdrs)):
                if self.config ["split"] == 'prox':
                    split_reward += split_reward_s (self.lbl, self.new_lbl, self.gt_lbl, self.step_cnt==0, 
                        self.inrs [0], self.inrs [0], self.bdrs [i], self.T, scaler, self.idx_list, self.keep)
                if self.config ["split"] == 'ins':
                    split_reward += split_reward_ins (self.lbl, self.new_lbl, self.gt_lbl, self.step_cnt==0, 
                        self.inrs [0], self.inrs [0], self.bdrs [i], self.T, scaler, self.idx_list, self.keep)
            for i in range (len (self.inrs)):
                merge_reward += merge_reward_s (self.lbl, self.new_lbl, self.gt_lbl, self.step_cnt==0, 
                    self.segs, self.inrs [i], self.bdrs [0], self.T, scaler, self.idx_list, self.keep)
                # merge_reward += merge_reward_step (action, self.gt_lbl, self.step_cnt==0, self.segs, self.inrs [0], self.bdrs [0], self.T, scaler, self.idx_list)
            # merge_reward += merge_pen_action (action, self.gt_lbl, self.step_cnt==0, self.segs, self.inrs [0], self.bdrs [0], self.T, scaler)
            # split_reward += split_rew_action (action, self.gt_lbl, self.step_cnt==0, self.segs, self.inrs [0], self.bdrs [0], self.T, scaler)
            
            # split_reward_inr += split_reward_s_onlyInr (self.lbl, self.new_lbl, self.gt_lbl, self.step_cnt==0, self.inrs, self.inrs, self.bdrs, self.T, scaler)
            reward += self.config ["spl_w"] * split_reward + self.config ["mer_w"] * merge_reward #+ split_reward * merge_reward`
            merge_ratio += ((merge_reward ) / range_merge) * (self.gt_lbl > 0)
            split_ratio += ((split_reward ) / range_split) * (self.gt_lbl > 0)

        self.split_ratio_sum = self.split_ratio_sum + split_ratio
        self.merge_ratio_sum = self.merge_ratio_sum + merge_ratio

        self.lbl = self.new_lbl
        self.step_cnt += 1
        
        #Reward
        self.rewards.append (reward)    
        self.sum_reward += reward
        if self.step_cnt >= min (self.tempT, self.T):
            done = True
        if self.config ["lowres"]:
            reward = self.lowres_reward (reward)
        ret = (self.observation (), reward, done, info)
        return ret

    def unique (self):
        return np.unique (self.lbl, return_counts=True)

    def random_init_lbl (self):
        if (self.T0 == 0):
            return
        action = self.gt_lbl > 0
        self.step (action)
        for t in range (1, self.T0):
            action = np.zeros_like (self.lbl)
            for i in np.unique (self.gt_lbl):
                if i == 0:
                    continue
                action += (self.gt_lbl == i) * self.rng.randint (0, 2)
            if self.type == "train":
                self.step (action)
            else:
                self.step_inference (action)

    def reset_end (self):
        """
            Must call after reset
        """
        self.w_map = None

        # Updating information for new data point
        if self.config ["exp_pool"] <= 0 or len (self.pool) < self.pool_capacity or self.pool_iter % self.pool_period == 0:
            if self.config ["reward"] == "seg" and (self.type == "train" or self.is3D):
                if not self.is3D:
                    self.gt_lbl = relabel (reorder_label (self.gt_lbl))

                # Get all unique id from ground truth
                unique_list = np.unique (self.gt_lbl, return_counts=True)
                # Remove small segment
                self.idx_list = [unique_list [0][i] for i in range (len (unique_list [0])) if unique_list [1][i] > 0]

                # Remove background
                if 0 in self.idx_list:
                    self.idx_list.remove (0)

                if self.config ["rew_drop"]:
                    # Choose number of initial cells for reward calculation
                    self.keep = self.rng.choice (self.idx_list, min (self.config ["rew_drop"], len (self.idx_list)), replace=False).tolist ()
                    
                    # From the current keep list, add more neighbor cells to keeps, get boundary and cell body mask
                    for idx in np.copy (self.keep):
                        # Dilate for boundary
                        dilated_seg = budget_binary_dilation (self.gt_lbl==idx, self.config ["out_radius"][0], fac=self.config["dilate_fac"])
                        # Multiply with boundary mask and get all the unique neighbors id
                        neighbor_ids = np.unique (dilated_seg * self.gt_lbl).tolist ();
                        # Remove background
                        if 0 in neighbor_ids:
                            neighbor_ids.remove (0)
                        # Remove its self
                        if idx in neighbor_ids:
                            neighbor_ids.remove (idx)
                        # Add up more neighbor cells to the list of reward calculation
                        neighbor_ids = self.rng.choice(neighbor_ids, min (self.config ["rew_drop_2"], len (neighbor_ids)), replace=False).tolist ()
                        # Ignore the added ones
                        for _idx  in neighbor_ids:
                            if not (_idx in self.keep):
                                self.keep.append (_idx)

                    # Get a map of keep list
                    self.keep_map = np.isin (self.gt_lbl, self.keep)

                    # Calculate foreground ratio
                    fg_ratio = np.count_nonzero (self.keep_map) / np.prod (self.keep_map.shape)
                    # fg_ratio = min (fg_ratio, 0.1)
                    # Sampling the ratio so that the number of sampled background pixel will be calculated for reward
                    bg_sampling_map = self.rng.choice ([False,True], self.keep_map.shape, replace=True, p=[1.0-fg_ratio, fg_ratio])
                    self.keep_map = self.keep_map | (bg_sampling_map & (self.gt_lbl == 0))
                    self.keep_map = self.keep_map.astype (np.float32)

                # Update cells body of reward calculation list [keep]
                self.segs = [self.gt_lbl == idx for idx in self.keep]

                self.bdrs = []
                self.inrs = []

                # A neighbor area map from all the cells in the keep list
                adj_map = np.zeros (self.gt_lbl.shape, dtype=np.bool)
                for radius in self.config ["out_radius"]:
                    bdrs = []
                    for seg in self.segs:
                        # For each cell in the keep list, get its dilated boundary
                        bdr = seg ^ budget_binary_dilation (seg, radius, fac=self.config["dilate_fac"])
                        # Update boundary list
                        bdrs.append (bdr)
                        # Update the adj map
                        adj_map = adj_map | bdr | seg
                    self.bdrs += [bdrs]

                # List of neighbor to the cells in the keep list (excluding the cells in the list itself)
                adj_list = np.unique (adj_map * self.gt_lbl).tolist ()
                self.idx_list = copy.deepcopy (self.keep)

                # Update the cells body and boundary of the just listed neighbor cells
                for idx in adj_list:
                    # For each neighbor that is not background, and not in the copy of keep list (will be updated)
                    if idx != 0 and idx not in self.idx_list:
                        seg = self.gt_lbl == idx
                        # Get the bdrs and boundaries
                        self.segs.append (seg)
                        for i, radius in enumerate (self.config ["out_radius"]):
                            self.bdrs [i].append (seg ^ budget_binary_dilation (seg, radius, fac=self.config["dilate_fac"]))
                        self.idx_list.append (idx)

                if not self.is3D:
                    for radius in self.config ["in_radius"]:
                        self.inrs += [[budget_binary_erosion (seg, radius, minsize=self.config["minsize"]) for seg in self.segs]]
                else:
                    self.inrs = [[seg for seg in self.segs]]


                if self.config ["exp_pool"] > 0:
                    training_sample = {}
                    training_sample ["raw"] = self.raw 
                    training_sample ["gt_lbl"] = self.gt_lbl  
                    training_sample ["inrs"] = self.inrs 
                    training_sample ["segs"] = self.segs  
                    training_sample ["bdrs"] = self.bdrs  
                    training_sample ["keep"] = self.keep  
                    training_sample ["keep_map"] = self.keep_map  
                    training_sample ["idx_list"] = self.idx_list  

                    self.pool.append (training_sample)
                    if len (self.pool) > self.pool_capacity:
                        self.pool.pop (0)

        if self.config ["exp_pool"] > 0:
            self.pool_iter += 1

        self.random_init_lbl ()

    def first_step_reward (self, density=None):
        reward = np.zeros (self.size, dtype=np.float32)
        st_foregr_ratio = self.config ["st_fgbg_ratio"]
        reward += ((self.new_lbl != 0) & (self.gt_lbl != 0)) * (1.0 - st_foregr_ratio)
        reward += ((self.new_lbl == 0) & (self.gt_lbl == 0)) * (st_foregr_ratio)
        reward -= ((self.new_lbl == 0) & (self.gt_lbl != 0)) * (1.0 - st_foregr_ratio)
        reward -= ((self.new_lbl != 0) & (self.gt_lbl == 0)) * (st_foregr_ratio)

        return reward

    def fgbg_reward (self, scaler=None):
        reward = np.zeros (self.size, dtype=np.float32)
        foregr_ratio = self.config ["fgbg_ratio"]
        # backgr reward, penalty
        reward += ((self.new_lbl == 0) & (self.gt_lbl == 0)) * foregr_ratio
        reward -= ((self.new_lbl != 0) & (self.gt_lbl == 0)) * foregr_ratio
        # foregr reward, penalty
        reward += ((self.new_lbl != 0) & (self.gt_lbl != 0)) * (1 - foregr_ratio)
        reward -= ((self.new_lbl == 0) & (self.gt_lbl != 0)) * (1 - foregr_ratio)

        return reward

    def background_reward (self, last_step):
        reward = np.zeros (self.size, dtype=np.float32)
        foregr_ratio = self.config ["fgbg_ratio"]
        if last_step:
            reward += ((self.new_lbl == 0) & (self.gt_lbl == 0)) * foregr_ratio
        reward -= ((self.new_lbl != 0) & (self.lbl == 0) & (self.gt_lbl == 0)) * foregr_ratio

        return reward   
    
    def foreground_reward (self, last_step):
        reward = np.zeros (self.size, dtype=np.float32)
        foregr_ratio = self.config ["fgbg_ratio"]
        reward += ((self.new_lbl != 0) & (self.lbl == 0) & (self.gt_lbl != 0)) * (1 - foregr_ratio)
        if last_step:
            reward -= ((self.new_lbl == 0) & (self.gt_lbl != 0)) * (1 - foregr_ratio)

        return reward

    def observation (self):
        lbl = self.lbl / self.max_lbl * 255.0
        done_mask = np.zeros (self.size, dtype=np.float32)
        if self.step_cnt >= self.T:
            done_mask += 255.0
        if self.config ["data_chan"] == 1:
            obs = [self.raw [None].astype (np.float32), done_mask [None]]
        elif self.config ["data_chan"] == 3:
            obs = [np.transpose (self.raw.astype (np.float32), [2, 0, 1]), done_mask [None]]
        if self.config ["use_lbl"]:
            obs.append (lbl [None])
        if self.config ["use_masks"]:
            obs.append (self.mask)

        obs = np.concatenate (obs, 0)

        return obs / 255.0

    def render (self):
        index = len (self.raw) // 2

        if self.is3D:
            tmp_raw = self.raw [index]
            tmp_lbl = self.lbl [index]
            tmp_gt_lbl = self.gt_lbl [index]
        else:
            tmp_raw = self.raw
            tmp_lbl = self.lbl
            tmp_gt_lbl = self.gt_lbl

        if self.config ["data_chan"] == 1:
            raw = np.repeat (np.expand_dims (tmp_raw, -1), 3, -1).astype (np.uint8)
        elif self.config ["data_chan"] == 3:
            raw = tmp_raw

        lbl = tmp_lbl.astype (np.int32)
        lbl = self.pred_lbl2rgb (lbl)

        gt_lbl = tmp_gt_lbl % 111
        gt_lbl += ((gt_lbl == 0) & (tmp_gt_lbl != 0))
        gt_lbl = self.gt_lbl2rgb (gt_lbl)
        
        masks = []
        for i in range (self.T):
            if self.is3D:
                mask_i = self.mask [i][index]
            else:
                mask_i = self.mask [i]
            mask_i = np.repeat (np.expand_dims (mask_i, -1), 3, -1).astype (np.uint8)
            masks.append (mask_i)

        max_reward = 7

        rewards = []
        for reward_i in [self.sum_reward] + self.rewards:
            if self.is3D:
                reward_i = reward_i [index]
            reward_i = ((reward_i + max_reward) / (2 * max_reward) * 255).astype (np.uint8) 

            reward_i = np.repeat (np.expand_dims (reward_i, -1), 3, -1)
            rewards.append (reward_i)

        while (len (rewards) < self.T + 1):
            rewards.append (np.zeros_like (rewards [0]))

        if self.is3D:
            split_ratio_sum = np.repeat (np.expand_dims ((self.split_ratio_sum [index] * 255).astype (np.uint8), -1), 3, -1)
            merge_ratio_sum = np.repeat (np.expand_dims ((self.merge_ratio_sum [index] * 255).astype (np.uint8), -1), 3, -1)
        else:
            split_ratio_sum = np.repeat (np.expand_dims ((self.split_ratio_sum * 255).astype (np.uint8), -1), 3, -1)
            merge_ratio_sum = np.repeat (np.expand_dims ((self.merge_ratio_sum * 255).astype (np.uint8), -1), 3, -1)

        line1 = [raw, lbl, gt_lbl,] + masks

        while (len (rewards) < len (line1)):
            rewards = [np.zeros_like (rewards [-1])] + rewards

        rewards[0] = split_ratio_sum
        rewards[1] = merge_ratio_sum

        line1 = np.concatenate (line1, 1)
        line2 = np.concatenate (rewards, 1)

        ret = np.concatenate ([line1, line2], 0)
        return ret

class EM_env (General_env):
    def __init__ (self, raw_list, config, type, gt_lbl_list=None, obs_format="CHW", seed=0):
        self.type = type
        self.raw_list = raw_list
        self.gt_lbl_list = gt_lbl_list
        self.rng = np.random.RandomState(seed)
        self.config = config
        self.obs_format = obs_format
        self.init (config)

    def random_crop (self, size, imgs):
        y0 = self.rng.randint (imgs[0].shape[0] - size[0] + 1)
        x0 = self.rng.randint (imgs[0].shape[1] - size[1] + 1)
        ret = []
        if self.is3D:
            z0 = self.rng.randint (imgs[0].shape[0] - size[0] + 1)
            y0 = self.rng.randint (imgs[0].shape[1] - size[1] + 1)
            x0 = self.rng.randint (imgs[0].shape[2] - size[2] + 1)

            for img in imgs:
                ret += [img[z0:z0+size[0], y0:y0+size[1], x0:x0+size[2]]]
        else:
            for img in imgs:
                ret += [img[y0:y0+size[0], x0:x0+size[1]]]
        return ret

    def reset (self, model=None, gpu_id=0):
        self.T0 = self.config ["T0"]
        self.step_cnt = 0
        idx = self.rng.randint (0, len (self.raw_list))
        self.raw = np.copy (np.array (self.raw_list [idx], copy=True))

        if (self.gt_lbl_list is not None):
            self.gt_lbl = np.copy(self.gt_lbl_list [idx])
        else:
            self.gt_lbl = np.zeros_like (self.raw)
        columns = 2
        rows = 2

        # Sampling new data point when not using pool, pool is not full or it is pool update period
        if self.config ["exp_pool"] <= 0 or len (self.pool) < self.pool_capacity or self.pool_iter % self.pool_period == 0:
            self.raw, self.gt_lbl = self.aug (self.raw, self.gt_lbl)
            self.raw, self.gt_lbl = self.random_crop (self.size, [self.raw, self.gt_lbl])
        else:
        # Get data point from pool, pool_iter will be updated in reset_end
            training_sample = self.pool [self.rng.randint (0, len (self.pool))]
            self.raw = training_sample ["raw"]
            self.gt_lbl = training_sample ["gt_lbl"]
            self.inrs = training_sample ["inrs"]
            self.segs = training_sample ["segs"]
            self.bdrs = training_sample ["bdrs"]
            self.keep = training_sample ["keep"]
            self.keep_map = training_sample ["keep_map"]
            self.idx_list = training_sample ["idx_list"]

        self.split_ratio_sum = (np.zeros (self.size, dtype=np.float32) + 0.5) * (self.gt_lbl > 0)
        self.merge_ratio_sum = (np.zeros (self.size, dtype=np.float32) + 0.5) * (self.gt_lbl > 0)

        self.mask = np.zeros ([self.T] + self.size, dtype=np.float32)
        self.lbl = np.zeros (self.size, dtype=np.int32)
        self.sum_reward = np.zeros (self.size, dtype=np.float32)
        self.rewards = []

        self.reset_end ()
        return self.observation ()

    def set_sample (self, idx, resize=False):
        self.step_cnt = 0
        self.T0 = self.config ["T0"]
        idx = idx
        if not self.is3D:
            while (self.raw_list [idx].shape [0] < self.size [0] \
                or self.raw_list [idx].shape [1] < self.size [1]):
                idx = self.rng.randint (len (self.raw_list))
        else:
            while (self.raw_list [idx].shape [0] < self.size [0] \
                or self.raw_list [idx].shape [1] < self.size [1] \
                or self.raw_list [idx].shape [2] < self.size [2]):
                idx = self.rng.randint (len (self.raw_list))

        self.raw = np.array (self.raw_list [idx], copy=True)
        if self.gt_lbl_list is not None:
            self.gt_lbl = np.array (self.gt_lbl_list [idx], copy=True)
        else:
            self.gt_lbl = np.zeros (self.size, dtype=np.int32)

        if (not resize):
            if self.gt_lbl_list is not None:
                self.raw, self.gt_lbl = self.random_crop (self.size, [self.raw, self.gt_lbl])
            else:
                self.raw = self.random_crop (self.size, [self.raw]) [0]
        else:
            self.raw = cv2.resize (self.raw, (self.size [1], self.size[0]), interpolation=cv2.INTER_NEAREST)
            self.gt_lbl = cv2.resize (self.gt_lbl.astype (np.int32), (self.size [1], self.size [0]), interpolation=cv2.INTER_NEAREST)

        self.split_ratio_sum = (np.zeros (self.size, dtype=np.float32) + 0.5) * (self.gt_lbl > 0)
        self.merge_ratio_sum = (np.zeros (self.size, dtype=np.float32) + 0.5) * (self.gt_lbl > 0)

        self.mask = np.zeros ([self.T] + self.size, dtype=np.float32)
        self.lbl = np.zeros (self.size, dtype=np.int32)
        self.sum_reward = np.zeros (self.size, dtype=np.float32)
        self.rewards = []

        self.reset_end ()
        return self.observation ()
