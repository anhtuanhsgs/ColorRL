

import matplotlib.pyplot as plt
import numpy as np
import h5py
import skimage.io as io



train_img = h5py.File ('CVPPP2017_training_images.h5')
train_lbl = h5py.File ('CVPPP2017_training_truth.h5')
test_img = h5py.File ('CVPPP2017_testing_images.h5')



train_img.keys ()


import cv2
data_train = {}
set_name = 'A1'
data_train [set_name] = []
print ("number of images in train set: ", len (train_img [set_name]))
for i, name in enumerate (train_img [set_name].keys ()):
    img = np.array (train_img [set_name][name]["rgb"]) [:, :, :3]
    img = cv2.resize (img, (512, 512), interpolation=cv2.INTER_NEAREST)
    data_train [set_name].append (img)


import cv2
data_test = {}
set_name = 'A1'
data_test [set_name] = []
print ("number of images in test set: ", len (test_img [set_name]))
for i, name in enumerate (test_img [set_name].keys ()):
    img = np.array (test_img [set_name][name]["rgb"]) [:, :, :3]
    img = cv2.resize (img, (512, 512), interpolation=cv2.INTER_NEAREST)
    data_test [set_name].append (img)



import cv2
data_label = {}
set_name = 'A1'
data_label [set_name] = []
for i, name in enumerate ( train_lbl[set_name].keys ()):
    img = np.array (train_lbl [set_name][name]["label"])
    img = cv2.resize (img, (512, 512), interpolation=cv2.INTER_NEAREST)
    data_label [set_name].append (img)

print ("saving train and test set")
train_set_A = (np.array (data_train ["A1"][:100])).astype (np.uint8)
valid_set_A = (np.array (data_train ["A1"][100:])).astype (np.uint8)
io.imsave ("train/A/train_set_A.tif", train_set_A)
io.imsave ("valid/A/valid_set_A.tif", valid_set_A)


train_set_B = (np.array (data_label ["A1"][:100])).astype (np.uint8)
valid_set_B = (np.array (data_label ["A1"][100:])).astype (np.uint8)
io.imsave ("train/train_set_B.tif", train_set_B)
io.imsave ("valid/valid_set_B.tif", valid_set_B)


print ("saving test set")
test_set_A = (np.array (data_test ["A1"])).astype (np.uint8)
io.imsave ("test/A/test_set_A.tif", test_set_A)
