import numpy as np

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

def DiffFGLabels(inLabel,gtLabel):
    # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    # output: difference of the number of foreground labels

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return -1

    inLabels = np.unique(inLabel)
    gtLabels = np.unique(gtLabel)
    maxInLabel = np.int(np.max(inLabels)) # maximum label value in inLabel
    minInLabel = np.int(np.min(inLabels)) # minimum label value in inLabel
    maxGtLabel = np.int(np.max(gtLabels)) # maximum label value in gtLabel
    minGtLabel = np.int(np.min(gtLabels)) # minimum label value in gtLabel

    return  (maxInLabel-minInLabel) - (maxGtLabel-minGtLabel)

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








