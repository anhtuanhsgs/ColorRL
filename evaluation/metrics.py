import numpy as np
from skimage.measure import label
from skimage.segmentation import relabel_sequential
from sklearn.metrics import adjusted_rand_score
from pprint import pprint

from cremi.evaluation.voi import voi
from cremi.evaluation.rand import adapted_rand as ARAND

def GetDices(inLabel,gtLabel):
    # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    # output: BestDice: best possible Dice score
    #         FgBgDice: dice score for joint foreground
    #
    # We assume that the background is labelled with 0
    #
    # For the original Dice score, labels corresponding to each other need to
    # be known in advance. Here we simply take the best matching label from
    # gtLabel in each comparison. We do not make sure that a label from gtLabel
    # is used only once. Better measures may exist. Please enlighten me if I do
    # something stupid here...

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        print('Shapes of label images not identical.')
        return 0, 0

    inLabels = np.unique(inLabel)
    maxInLabel = np.amax(inLabels)
    maxGtLabel = np.amax(gtLabel)

    if(len(inLabels)<=1): # trivial solution
        print('Only one label given, assuming all background.')
        return 0, 0

    # calculate Dice between all labels using 2d histogram
    xedges = np.linspace( - 0.5, maxInLabel+0.5,maxInLabel+2) # interval edges of histogram
    yedges = np.linspace( - 0.5, maxGtLabel+0.5,maxGtLabel+2) # interval edges of histogram

    # histograms
    H2D, xedges, yedges = np.histogram2d(inLabel.flatten(), gtLabel.flatten(), bins=(xedges, yedges))
    inH1D, edges = np.histogram(inLabel.flatten(), bins=xedges)
    gtH1D, edges = np.histogram(gtLabel.flatten(), bins=yedges)

    # reshape 1d histograms for broadcast
    inH1D = np.reshape(inH1D,[len(inH1D),1])
    gtH1D = np.reshape(gtH1D,[1,len(gtH1D)])

    # best Dice is (2*overlap(A,B)/(size(A)+size(B)))
    perCombinationDice = 2*H2D/(inH1D + gtH1D + 1e-16)
    sMax = np.amax(perCombinationDice[1:,1:],1)
    bestDice = np.mean(sMax)

    # FgBgDice
    Overlap = np.sum(H2D[1:,1:])
    inFG = np.sum(inH1D[1:])
    gtFG = np.sum(gtH1D[:,1:])
    if ((inFG + gtFG)>1e-16):
        FgBgDice = 2*Overlap/(inFG + gtFG)
    else:
        FgBgDice = 1 # gt is empty and in has it found correctly

    return bestDice, FgBgDice

def get_fast_aji(true, pred):
    """
    AJI version distributed by MoNuSeg, has no permutation problem but suffered from 
    over-penalisation similar to DICE2
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.
    """
    true = np.copy(true) # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    
    # prefill with value
    pairwise_inter = np.zeros([len(true_id_list) -1, 
                               len(pred_id_list) -1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) -1, 
                               len(pred_id_list) -1], dtype=np.float64)

    # caching pairwise
    for true_id in true_id_list[1:]: # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0: # ignore
                continue # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id-1, pred_id-1] = inter
            pairwise_union[true_id-1, pred_id-1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care 
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
    #
    paired_true = (list(paired_true + 1)) # index to instance ID
    paired_pred = (list(paired_pred + 1))
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score

def DiffFGLabels(inLabel,gtLabel):
    # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    # output: difference of the number of foreground labels

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return -1

    inLabels = np.unique(inLabel)
    gtLabels = np.unique(gtLabel)
    # maxInLabel = np.int(np.max(inLabels)) # maximum label value in inLabel
    # minInLabel = np.int(np.min(inLabels)) # minimum label value in inLabel
    # maxGtLabel = np.int(np.max(gtLabels)) # maximum label value in gtLabel
    # minGtLabel = np.int(np.min(gtLabels)) # minimum label value in gtLabel

    # return  (maxInLabel-minInLabel) - (maxGtLabel-minGtLabel)
    return (abs (len (gtLabels) - len (inLabel)))
    

def getIOU(inLabel,gtLabel):
    intersect_cnt = np.count_nonzero (inLabel & gtLabel)
    union_cnt = np.count_nonzero (inLabel | gtLabel)
    if (union_cnt == 0):
        return 0
    return intersect_cnt / union_cnt

# # overlap 1 pixel => True
# def is_overlap(inLabel,gtLabel):
#   result = np.count_nonzero(inLabel & gtLabel)
#
#   if result > 0:
#       return True
#   else:
#       return False

# overlap >= 50% pixel => True
def is_overlap(inLabel,gtLabel, thres=0.5):
    result = np.count_nonzero(inLabel & gtLabel)
    gt_numpix = np.count_nonzero(gtLabel)

    if result > gt_numpix*thres:
        return True
    else:
        return False

def kitti_metric(inLabel,gtLabel):
    # Calculates weighted and unweighted coverage
    #   Args:
    #       a: [H, W], index mask, prediction
    #       b: [H, W], index mask, ground - truth
    #   Returns:
    #       MWCov, MUCov,

    if (np.max(inLabel) == 0) and (np.max(gtLabel) == 0):
        return 0, 0, 0, 0

    inLabels = np.unique(inLabel).tolist()
    if 0 in inLabels:
        inLabels.remove(0)

    gtLabels = np.unique(gtLabel).tolist()
    if 0 in gtLabels:
        gtLabels.remove(0)

    MUCov = 0
    MWCov = 0
    AvgFP = 0
    AvgFN = 0
    fp_list = []
    fn_list = []

    # -------------------------
    # Calculate FP
    # -------------------------
    for r_lbl in inLabels:
        if r_lbl == 0:
            continue
        fp = 0
        for r_G in gtLabels:
            if (r_G == 0) or (r_G in fp_list):
                continue

            if is_overlap(inLabel==r_lbl, gtLabel==r_G):
                fp +=1
                fp_list.append(r_G)
                break

        if fp == 0:
            AvgFP +=1

    if len(inLabels) != 0:
        AvgFP = AvgFP / len(inLabels)
    else: # len (inLabels) == 0:
        # if len(gtLabels) == 0:
        #   AvgFP = 0
        # elif len(gtLabels) != 0:
        #   AvgFP = 1
        AvgFP = 1

    #-------------------------
    # Calculate FN
    # -------------------------
    for r_G in gtLabels:
        if r_G == 0:
            continue
        fn = 0
        for r_lbl in inLabels:
            if (r_lbl == 0) or (r_lbl in fn_list):
                continue

            if is_overlap(inLabel==r_lbl, gtLabel==r_G):
                fn +=1
                fn_list.append(r_lbl)
                break

        if fn == 0:
            AvgFN +=1

    if len (gtLabels) != 0:
        AvgFN = AvgFN / len (gtLabels)
    else: # len (gtLabels) == 0:
        # if len(inLabels) == 0:
        #   AvgFN = 0
        # elif len(gtLabels) != 0:
        #   AvgFN = 1
        AvgFN = 1

    # -------------------------
    # Calculate MUcov and MWCov
    # -------------------------
    for r_G in gtLabels:
        if r_G == 0:
            continue
        max_iou = 0
        for r_lbl in inLabels:
            if r_lbl == 0:
                continue
            current_iou = getIOU(inLabel==r_lbl, gtLabel==r_G)
            if current_iou > max_iou:
                max_iou = current_iou

        num_r_g = np.count_nonzero(gtLabel == r_G)
        MUCov += max_iou
        MWCov += max_iou * num_r_g


    if len (gtLabels) != 0:
        MUCov = MUCov / len (gtLabels)
    else:
        MUCov = 0

    # MUCov = MUCov / len(gtLabels)
    MWCov = MWCov/(np.count_nonzero(gtLabel > 0) + 1)

    return MWCov, MUCov, AvgFP, AvgFN

def evaluate (lbl, gt_lbl):
    pred_lbl = lbl

    bestDice, FgBgDice = GetDices (pred_lbl, gt_lbl)

    diffFG = DiffFGLabels (pred_lbl, gt_lbl)

#     MWCov, MUCov, AvgFP, AvgFN  = kitti_metric(pred_lbl, gt_lbl)

    rand_i = adjusted_rand_index(gt_lbl,pred_lbl)
    
    (voi_split, voi_merge) = voi(np.expand_dims(lbl,0), np.expand_dims(gt_lbl,0), ignore_groundtruth = [0])
    adapted_rand = ARAND(np.expand_dims(lbl,0), np.expand_dims(gt_lbl,0))

    return bestDice, rand_i, voi_split, voi_merge, adapted_rand, get_fast_aji (gt_lbl, lbl)

def adjusted_rand_index (gt_lbl, pred_lbl):
    gt_lbl = gt_lbl.flatten ()
    pred_lbl = pred_lbl.flatten ()
    return adjusted_rand_score (gt_lbl, pred_lbl)

def get_separate_labels(label_img):
    w = 32
    l32 = label_img.astype('int32')
    l32i = ((l32[:, :, 0] << 2 * w) + (l32[:, :, 1] << w) + l32[:, :, 2])
    return relabel_sequential(l32i)[0].astype('int32')