from __future__ import division
import numpy as np
import torch
import json
import logging
import math as m
from torch.autograd import Variable
from scipy import ndimage as ndi
from natsort import natsorted
import os, sys, glob, time, warnings
from Utils.img_aug_func import *
from skimage.measure import label
from skimage.filters import sobel
from malis import rand_index 
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage import img_as_bool


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

def malis_rand_index (gt_lbl, pred_lbl):
    ret = rand_index (gt_lbl, pred_lbl) [0]
    ret = float (ret)
    return ret

def malis_f1_score (gt_lbl, pred_lbl):
    if (np.max (gt_lbl) == 0):
        gt_lbl += 1
    ret = rand_index (gt_lbl, pred_lbl) [1]
    ret = float (ret)
    return ret

def adjusted_rand_index (gt_lbl, pred_lbl):
    gt_lbl = gt_lbl.flatten ()
    pred_lbl = pred_lbl.flatten ()
    return adjusted_rand_score (gt_lbl, pred_lbl)

def build_blend_weight (shape):
    # print ("patch shape = ", shape)
    yy, xx = np.meshgrid (
            np.linspace(-1,1,shape[0], dtype=np.float32),
            np.linspace(-1,1,shape[1], dtype=np.float32)
        )
    d = np.sqrt(xx*xx+yy*yy)
    sigma, mu = 0.5, 0.0
    v_weight = 1e-6+np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    v_weight = v_weight/v_weight.max()
    return v_weight

def guassian_weight_map (shape):
    # print ("patch shape = ", shape)
    yy, xx = np.meshgrid (
            np.linspace(-1,1,shape[0], dtype=np.float32),
            np.linspace(-1,1,shape[1], dtype=np.float32)
        )
    d = np.sqrt(xx*xx+yy*yy)
    sigma, mu = 0.5, 0.0
    v_weight = 1e-6+np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    v_weight = v_weight/v_weight.max()
    return v_weight

def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        if param.grad is None:
            shared_param._grad = None
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()

def density_map (lbl):
    lbl = np.pad (lbl, 1, 'constant', constant_values=0)
    if (np.min (lbl) > 0) or (np.max (lbl) == 0):
        return np.ones (lbl.shape, dtype=np.float32)
    distance = ndi.distance_transform_edt(lbl)
    idx_list = np.unique (lbl)
    
    max_dist = np.max (distance)
    local_peak_dist_list = []
    ret = np.zeros (lbl.shape, dtype=np.float32)
    for i in idx_list:
        if i == 0:
            continue
        local_dist_map = distance * (lbl == i)
        local_peak_dist = np.max (local_dist_map)
        local_peak_dist_list.append (local_peak_dist)
        ret += local_dist_map * (max_dist / local_peak_dist)

    ret = ret / np.max (ret)
    ret = np.clip (ret, 0.33, 1.0) * (ret > 0)
    ret = np.clip (ret, 0.1, 1.0)
    return ret [1:, 1:][:-1,:-1]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def reward_scaler (r, alpha, beta):
    r = m.pow (alpha, (r * beta)) / m.pow (alpha, 1 * beta)
    return r

def normal(x, mu, sigma, gpu_id, gpu=False):
    pi = np.array([m.pi])
    pi = torch.from_numpy(pi).float()
    if gpu:
        with torch.cuda.device(gpu_id):
            pi = Variable(pi).cuda()
    else:
        pi = Variable(pi)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b

def get_cell_prob (lbl, dilation, erosion):
    ESP = 1e-10
    elevation_map = []
    # print (len (lbl), lbl [0].shape)
    for img in lbl:
        elevation_map += [sobel (img)]
    elevation_map = np.array (elevation_map)
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     elevation_map = img_as_bool (elevation_map)
    elevation_map = elevation_map > ESP
    cell_prob = [((lbl [i] > 0) ^ elevation_map [i]) & (lbl [i] > 0) for i in range (len (lbl))]
    for i in range (len (cell_prob)):
        for j in range (erosion):
            cell_prob [i] = binary_erosion (cell_prob [i])
    for i in range (len (cell_prob)):
        for j in range (dilation):
            cell_prob [i] = binary_dilation (cell_prob [i])
    return np.array (cell_prob, dtype=np.uint8) * 255

def clean (lbl, minsize=40):
    sizes = np.bincount (lbl.ravel ())
    mask_sizes = sizes >= minsize
    mask_sizes [0] = 0
    lbl = lbl * mask_sizes [lbl]
    return lbl

def clean_reindex (lbl, minsize=40):
    lbl = clean (lbl)
    ret = np.zeros (lbl.shape, dtype=np.int32)
    cur_max_val = 0
    val_list = np.unique (lbl)
    for val in val_list:
        if (val == 0):
            continue
        mask = (lbl == val)
        sub_lbl = label (mask, connectivity=1).astype (np.int32)
        sub_lbl = clean (sub_lbl, minsize)
        sub_lbl = label (sub_lbl, connectivity=1).astype (np.int32)

        sub_lbl += cur_max_val * (sub_lbl > 0)
        ret += sub_lbl
        cur_max_val = np.max (ret)
    return ret

def vols2list (vols):
    ret = []
    for vol in vols:
        for img in vol:
            ret += [img]
    return ret

def get_data (path, relabel):
    train_path = natsorted (glob.glob(path + 'A/*.tif'))
    train_label_path = natsorted (glob.glob(path + 'B/*.tif'))
    train_path += natsorted (glob.glob (path + "A/*.npy"))
    train_label_path += natsorted (glob.glob (path + "B/*.npy"))

    
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    if "3D" in path:
        return X_train, y_train

    
    if (len (X_train) > 0):
        if len (X_train) == 1:
            X_train = X_train [0]
        elif X_train [0].ndim > 2:
            X_train =  vols2list (X_train)
    if (len (y_train) > 0):
        if len (y_train) == 1:
                y_train = y_train [0]
        elif y_train [0].ndim > 2:
            y_train = vols2list (y_train)
        if (relabel):
            

            gt_prob = get_cell_prob (y_train, 0, 1)
            # plt.imshow (gt_prob [0])
            # plt.show ()
            y_train = []
            for img in gt_prob:
                if relabel:
                    y_train += [label (img).astype (np.int32)]
                else:
                    y_train += [img]
    else:
        y_train = np.zeros_like (X_train)

    return X_train, y_train

class Scheduler ():
    def __init__ (self, var, schedule, delta):
        self.var = var
        self.schedule = schedule
        self.delta = delta
        self.iter = 0
        self.schedule_index = 0

    def next (self):
        self.iter += 1
        idx = self.schedule_index
        if idx < len (self.schedule) and self.iter >= self.schedule [idx] :
            self.var += self.delta
            self.schedule_index += 1
        return self.var

    def value (self):
        return self.var

class EspTracker ():
    def __init__ (self, eps, eps_step):
        self.eps = eps
        self.eps_step = eps_step
        self.index = 0
        self.value = eps [0]
        self.nstep = 0

    def step (self, n):
        self.nstep += n
        if (self.nstep <= self.eps_step [0]):
            return
        if self.value <= self.eps[-1] or self.index >= len (self.eps) - 1:
            return
        index = self.index
        self.value -= (self.eps[index-1] - self.eps[index]) / (self.eps_step[index]-self.eps_step[index-1]) * n
        if index < len (self.eps) and self.value <= self.eps [index]:
            self.index += 1

class ScalaTracker ():
    def __init__ (self, size):
        self.arr = []
        self.size = size

    def push (self, x):
        self.arr.append (x)
        if len (self.arr) > self.size:
            self.arr.pop (0)

    def mean (self):
        if len (self.arr) == 0:
            return 0
        return np.mean (self.arr)

if __name__ == "__main__":
    r = float (input ())
    print (reward_scaler (r))