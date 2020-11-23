import time, os, warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import binary_dilation, binary_erosion, disk, ball
from skimage import img_as_bool
import cv2
import math as m


def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

def create_dir (directory):
    folders = directory.split ("/")
    path = ""

    for folder in folders:
        path += folder + "/"
        if not os.path.exists(path):
            os.makedirs(path)

def reorder_label (lbl):
    ret = np.zeros_like (lbl)
    val_list = np.unique (lbl).tolist ()
    if val_list [0] != 0:
        for i in range (len (val_list)):
            if val_list [i] == 0:
                val_list.pop (i)
                val_list = [0] + val_list
                break
    for i, val in enumerate (val_list):
        if val == 0:
            continue
        ret [lbl == val] = i
    return ret.astype (np.int32, copy=False)

def relabel (lbl):
    ret = np.zeros (lbl.shape, dtype=np.int32)
    cur_max_val = 0
    val_list = np.unique (lbl)
    for val in val_list:
        if (val == 0):
            continue
        mask = (lbl == val)
        # sub_lbl = label (mask, connectivity=1).astype (np.int32)
        sub_lbl = mask.astype (np.int32)

        sub_lbl += cur_max_val * (sub_lbl > 0)
        ret += sub_lbl
        cur_max_val = np.max (ret)
    return ret

def budget_binary_dilation (img, radius, fac=2):

    if radius < 0:
        return np.ones (img.shape, dtype=np.bool)

    ori_shape = img.shape
    # plt.imshow (img)
    # plt.show ()

    if (len (img.shape) == 3):
        img = img [::fac,::fac,::fac]
        img = binary_dilation (img, ball (radius // fac))
    else:
        img = img [::fac,::fac]
        img = binary_dilation (img, disk (radius // fac))
    
    
    # plt.imshow (img)
    # plt.show ()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = img_as_bool (resize (img, ori_shape, order=cv2.INTER_NEAREST, mode='reflect', anti_aliasing=False))
        # plt.imshow (img)
        # plt.show ()
    return img

def budget_binary_erosion (img, fac, minsize=20):
    ori_sqr_area = m.sqrt (np.count_nonzero (img))
    sqr_area = ori_sqr_area
    cnt = 1
    inr = np.pad (img, 1, mode='constant', constant_values=0)
    while (sqr_area > fac * ori_sqr_area and sqr_area > minsize):
        inr = binary_erosion (inr)
        sqr_area = m.sqrt (np.count_nonzero (inr))
        cnt += 1
    if (len (img.shape) == 3):
        return inr [1:-1,1:-1,1:-1]
    return inr [1:-1,1:-1]    

def resize_volume (vol, size, ds, is3D):
    if abs (int (ds) - ds) > 1e-4:
        ret = []
        for img in vol:
            size = [int (img.shape [0] * ds), int (img.shape [1] * ds)]
            if img.dtype == np.uint8:
                ret = [cv2.resize (vol[i].astype (np.int32), (size[1], size[0]), interpolation=cv2.INTER_LINEAR) for i in range (len (vol))]
            else:
                ret = [cv2.resize (vol[i], (size[1], size[0]), interpolation=cv2.INTER_NEAREST) for i in range (len (vol))]
        return ret

    if ds <= 0:
        if (len (vol) > 0) and vol[0].dtype == np.uint32:
            ret = [cv2.resize (vol[i].astype (np.int32), (size[1], size[0]), interpolation=cv2.INTER_NEAREST) for i in range (len (vol))]
        else:
            ret = [cv2.resize (vol[i], (size[1], size[0]), interpolation=cv2.INTER_NEAREST) for i in range (len (vol))]
    else:
        if (is3D):
            ret = [vol [i][::ds, ::ds, ::ds] for i in range (len (vol))]
        else:
            ret = [vol [i][::ds, ::ds] for i in range (len (vol))]
    return ret
