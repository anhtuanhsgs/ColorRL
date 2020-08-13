# import cv2
from scipy.ndimage.interpolation    import map_coordinates
# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation
from skimage import io
from scipy.ndimage.filters import gaussian_filter as gaussian
import time
from scipy.ndimage.morphology import binary_erosion
from skimage.segmentation import find_boundaries
from random import shuffle


import colorsys

def color_generator (N):
    HSV_tuples = [(x*1.0/N, 0.5, (x%8)*0.07 + 0.5) for x in range(N)]
    RGB_tuples = list (map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    COLOR_LIST = [(int (rgb[0] * 255), int (rgb[1] * 255), int (rgb[2] * 255)) for rgb in RGB_tuples]
    # shuffle (COLOR_LIST)
    COLOR_LIST [0] = (0, 0, 0)

    def index2rgb (index):
        return COLOR_LIST [index]
    def lbl2rgb (lbl):
        original_shape = np.squeeze (lbl).shape
        lbl = lbl.flatten ().tolist ()
        lbl = list (map (index2rgb, lbl))
        lbl = np.array (lbl).reshape (original_shape + (3,))
        return lbl
    return lbl2rgb

def read_im (paths):
    ret = []
    for path in paths:
        if ".tif" in path:
            ret.append (io.imread (path))
        elif ".npy" in path:
            print (path)
            vol = np.load (path, allow_pickle=True)
            ret += [vol.tolist ()]

    return ret

def random_gaussian_blur (image, n, seed=None):
    blured = []
    for i in range (n):
        x = np.random.randint (0, len (image))
        blured += [x]
        if not x in blured:
            image[x] = gaussian (image[x], sigma=1)
    return image

def random_blackout (image, n, randt, range_xy = (50, 256)):
    blacked = []
    for i in range (n):
        x = randt.randint (len (image))
        while (x in blacked):
            x = randt.randint (len (image))
        blacked += [x]
    for i in blacked:
        lenx = randt.randint (range_xy[0], range_xy[1])
        leny = randt.randint (range_xy[0], range_xy[1])
        x0 = randt.randint (image.shape[1] - lenx + 1)
        y0 = randt.randint (image.shape[2] - leny + 1)
        value = float (1.0 * randt.randint (255) / 255.0) 
        image[i, y0:y0+leny, x0:x0 + lenx] = value
    return image


def rotate(image, n):
    assert ((image.ndim == 2) | (image.ndim == 3))
    assert (n < 4)        
    rot_k = n
    rotated = image.copy()
    if image.ndim==2:
        rotated = np.rot90(image, rot_k, axes=(0,1))
    elif image.ndim==3:
        rotated = np.rot90(image, rot_k, axes=(1,2))
    image = rotated
    return image

def rotate3D (image, n):
    rot_k1 = n // 16
    rot_k2 = (n % 16) // 4
    rot_k3 = (n % 4)
    rotated = np.rot90(image, rot_k1, axes=(0,1))
    rotated = np.rot90(image, rot_k2, axes=(0,2))
    rotated = np.rot90(image, rot_k3, axes=(1,2))
    return rotated


def reverse(image, n):
    assert ((image.ndim == 2) | (image.ndim == 3))
    assert (n < 2)

    if n==0:
        reverse = image
    elif n==1:
        reverse = image[::-1,...]

    return reverse

def erode_label (imgs, iterations=1):
    ret = []
    for img in imgs:
        bndr_map = 1 - find_boundaries (img)
        bndr_map = binary_erosion (bndr_map, iterations=iterations)
        ret.append (np.multiply (bndr_map, img).astype (img.dtype))
    return ret


def flip(image, n):
    assert ((image.ndim == 2) | (image.ndim == 3))
    assert (n < 4)
    random_flip = n
    if random_flip==0:
        flipped = image[...,::1,::-1]
        image = flipped
    elif random_flip==1:
        flipped = image[...,::-1,::1]
        image = flipped
    elif random_flip==2:
        flipped = image[...,::-1,::-1]
        image = flipped
    elif random_flip==3:
        flipped = image
        image = flipped
    return image

def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

def apply_aug (block, labels, func, seed=None):
    return func (block, seed), func (labels, seed)

def RotFlipRev3D (volumes, rng):
    volumes = [np.copy (vol) for vol in volumes]
    nFlip = rng.randint (4)
    nRev = rng.randint (2)
    nRot = rng.randint (4 * 4 * 4)
    ret = [flip (vol, nFlip) for vol in volumes]
    ret = [rotate3D (vol, nRot) for vol in ret]
    ret = [reverse (vol, nRev) for vol in ret]
    return ret


def FlipRev3D (volumes, rng):
    volumes = [np.copy (vol) for vol in volumes]
    nFlip = rng.randint (4)
    nRev = rng.randint (2)
    ret = [flip (vol, nFlip) for vol in volumes]
    ret = [reverse (vol, nRev) for vol in ret]
    return ret
    




