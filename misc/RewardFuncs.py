import numpy as np
import skimage.io as io
from skimage.morphology import disk
from skimage.morphology import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
import time
from skimage import img_as_bool
from skimage.measure import label
from skimage.transform import resize
import cv2
import math as m

# def reorder_label (lbl):
#     ret = np.zeros_like (lbl)
#     val_list = np.unique (lbl).tolist ()
#     if val_list [0] != 0:
#         for i in range (len (val_list)):
#             if val_list [i] == 0:
#                 val_list.pop (i)
#                 val_list = [0] + val_list
#                 break
#     for i, val in enumerate (val_list):
#         if val == 0:
#             continue
#         ret [lbl == val] = i
#     return ret.astype (np.int32, copy=False)


# def relabel (lbl):
#     ret = np.zeros (lbl.shape, dtype=np.int32)
#     cur_max_val = 0
#     val_list = np.unique (lbl)
#     for val in val_list:
#         if (val == 0):
#             continue
#         mask = (lbl == val)
#        # sub_lbl = label (mask, connectivity=1).astype (np.int32)

#         sub_lbl += cur_max_val * (sub_lbl > 0)
#         ret += sub_lbl
#         cur_max_val = np.max (ret)
#     return ret

# label = io.imread ("../Data/CVPPP_Challenge/train/B/train_set_B.tif")
# label_ds = io.imread ("../Data/snemi/train/B/train-labels.tif")

# def budget_binary_dilation (img, radius):
#     fac = 2
#     ori_shape = img.shape
#     img = img [::fac,::fac]
#     img = binary_dilation (img, disk (radius // fac))
#     img = img_as_bool (resize (img.astype (np.int8), ori_shape, order=cv2.INTER_NEAREST))
#     return img

# def budget_binary_erosion (img, fac):
#     sqr_area = m.sqrt (np.count_nonzero (img))
#     inr = binary_erosion (img)
#     while (m.sqrt (np.count_nonzero (inr)) > fac * sqr_area):
#         inr = binary_erosion (inr)
#     return inr

# img_id = 44
# radius = 36
# img = relabel (reorder_label (label_ds [img_id][::2, ::2][:256,:256]))
# lbl = [img == id for id in np.unique (img)]
# segs = [seg for seg in lbl]
# curtime = time.time ()
# bdrs = [seg ^ budget_binary_dilation (seg, radius) for seg in segs]
# inrs = [budget_binary_erosion (seg, 0.5) for seg in segs]
# print (time.time () - curtime)

def bdr_cnt_mask (bdr, seg, bdr_sum, T, debug=False):
    bdr_cnt = np.array ([0] * ((2**T) + 1))
    bdr_uni = np.unique (bdr, return_counts=True)
    for i in range (len(bdr_uni [0])):
        bdr_cnt [bdr_uni [0][i]] = bdr_uni[1][i]
    _bdr_cnt = bdr_sum - bdr_cnt
    _bdr_cnt [-1] = bdr_cnt [-1] = 0
#     print (bdr_cnt [0], _bdr_cnt [0])

#     plt.imshow (bdr_cnt [seg])
#     plt.show ()
    return (bdr_cnt [seg].astype (np.int32, copy=False), _bdr_cnt [seg].astype (np.int32, copy=False))
    
def split_reward (old_lbl, lbl, gt_lbl, first_step, T, scaler=None):
    t_spl_rew = np.zeros (lbl.shape, dtype=np.float32) #True split reward
    f_mer_pen = np.zeros (lbl.shape, dtype=np.float32) #False merge penalty

    for i in np.unique (gt_lbl):
        if i == 0:
            continue
#         print (i)
        out1 = (True ^ segs [i])
        out2 = (True ^ bdrs[i])
        bdr = bdrs [i] * lbl; seg = segs [i] * lbl 
        o_bdr = bdrs[i] * old_lbl; o_seg = segs [i] * old_lbl 
        bdr [(gt_lbl==0)|out2] = (2 ** T); seg [(gt_lbl==0)|out1] = (2 ** T)
        o_bdr [(gt_lbl==0)|out2] = (2 ** T); o_seg [(gt_lbl==0)|out1] = (2 ** T)
        
        bdr_sum = np.count_nonzero (bdrs[i] * gt_lbl) + 1 #Total non background pixels in bdr 
        bdr_cnt, _bdr_cnt = bdr_cnt_mask (bdr, seg, bdr_sum, T, i==3) # #of sames, diffs count in each pixel of inner
        o_bdr_cnt, _o_bdr_cnt = bdr_cnt_mask (o_bdr, o_seg, bdr_sum, T)

        t_spl_rew += (_bdr_cnt - _o_bdr_cnt) / bdr_sum
        f_mer_pen += bdr_cnt / (bdr_sum * T)
        
    ret = t_spl_rew - f_mer_pen
    if scaler is not None:
        ret *= scaler
    return ret.astype (np.float32, copy=False)

def inr_cnt_mask (inr, seg, inr_sum, T, debug=False):
    inr_cnt = np.array ([0] * ((2**T) + 1))
    inr_uni = np.unique (inr, return_counts=True)
    for i in range (len (inr_uni [0])):
        inr_cnt [inr_uni [0][i]] = inr_uni [1][i]

    _inr_cnt = inr_sum - inr_cnt
    _inr_cnt [-1] = inr_cnt [-1] = 0
    
    return (inr_cnt [seg].astype (np.int32, copy=False), _inr_cnt [seg].astype (np.int32, copy=False)) 

def merge_reward (old_lbl, lbl, gt_lbl, first_step, T, scaler=None):
    t_mer_rew = np.zeros (lbl.shape, dtype=np.float32)
    f_spl_pen = np.zeros (lbl.shape, dtype=np.float32)
    
    for i in np.unique (gt_lbl):
        if i == 0:
            continue
        out0 = (True ^ inrs [i] ) # exclude only inner
        out1 = (True ^ segs [i]) # exclude only segment
        inr = inrs [i] * lbl; seg = segs [i] * lbl 
        o_inr = inrs [i] * old_lbl; o_seg = segs [i] * old_lbl
        inr [(gt_lbl==0)|out0] = (2 ** T); seg [(gt_lbl==0)|out1] = (2 ** T)
        o_inr [(gt_lbl==0)|out0] = (2 ** T); o_seg [(gt_lbl==0)|out1] = (2 ** T)
        
        inr_sum = np.count_nonzero (inrs[i] * gt_lbl) + 1 #Total non background pixels in seg 
        inr_cnt, _inr_cnt = inr_cnt_mask (inr, seg, inr_sum, T)
        o_inr_cnt, _o_inr_cnt = inr_cnt_mask (o_inr, o_seg, inr_sum, T)
   
        t_mer_rew += inr_cnt / (inr_sum * T)
        f_spl_pen += (_inr_cnt - _o_inr_cnt) / inr_sum

    ret = t_mer_rew - f_spl_pen
    if scaler is not None:
        ret *= scaler
    return ret.astype (np.float32, copy=False)


# if __name__ == '__main__':
#     state = np.zeros ((256, 256), dtype=np.int32)
#     img = img.astype (np.int32)
#     plt.imshow (img, cmap='tab20c')
#     plt.show ()
#     print ("___________________________________________________")
#     np.random.seed (1)
#     currtime = time.time ()
#     for i in range (4):
#         new_state = np.copy (state)
#         for val in np.unique (img):

#             if (val == 0):
#                 continue
#             # if 0.5 > np.random.rand ():
#             #     new_state += (2**i) * (val == img)
#             if val == 13:
#                 plt.imshow (inrs [val])
#                 plt.show ()
#                 # new_state += ((2**i) * (val == img) and inrs [val]) * (np.random.rand (*state.shape) > 0.5)
#                 new_state += ((2**i) * ((val == img) & inrs [val]))

#         split_rew = split_reward (state, new_state, img, (i==0), 4)
#         merge_rew = merge_reward (state, new_state, img, (i==0), 4)
#         fig=plt.figure(figsize=(8, 8))
#         split_rew [0,0] = 1.0
#         split_rew [0,1] = 0.0
#         fig.add_subplot(1, 3, 1);    plt.imshow (new_state, cmap='tab20');
#         fig.add_subplot(1, 3, 2);    plt.imshow (split_rew, cmap='gray')
#         fig.add_subplot(1, 3, 3);    plt.imshow (merge_rew, cmap='gray')
        
#         print (np.max (split_rew), np.min (split_rew))
#         plt.show ()
#         print ("___________________________________________________")
#         state = new_state
#     print (time.time () - currtime)