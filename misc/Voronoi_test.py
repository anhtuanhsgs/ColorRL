
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib as mpl
import matplotlib.cm as cm
from skimage.measure import label
from skimage.transform import resize

import matplotlib
matplotlib.use('agg')
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def create_voronoi_2d ():
    # make up data points
    # np.random.seed(1234)
    points = np.random.rand(21, 2)
    points = np.clip (points, 0.05, 0.95)
    # compute Voronoi tesselation
    vor = Voronoi(points)
    minima = 0
    maxima = 20
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues_r)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)

    fig, ax = plt.subplots()
    fig.set_figheight (7.12)
    fig.set_figwidth (7.12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.subplots_adjust(0,0,1,1)
    ax.axis('off')

    color = 1
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), color=mapper.to_rgba(color))
        color += 1
        
    fig.canvas.draw()
    data = np.fromstring (fig.canvas.tostring_rgb (), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data [...,2]


# In[2]:


data = create_voronoi_2d ()
print (data.shape)


# In[3]:


def show (img, cm='gray'):
    plt.figure (figsize=(7, 7))
    plt.imshow (img, cmap=cm)
    plt.show ()


# In[4]:


from skimage.transform import resize
from skimage.morphology import binary_erosion, binary_dilation
from skimage.filters import sobel

def get_cell_prob (lbl, dilation, erosion):
    ESP = 1e-5
    elevation_map = []
    for img in lbl:
        elevation_map += [sobel (img)]
    elevation_map = np.array (elevation_map)
    elevation_map = elevation_map > ESP
    cell_prob = ((lbl > 0) ^ elevation_map) & (lbl > 0)
    for i in range (len (cell_prob)):
        for j in range (erosion):
            cell_prob [i] = binary_erosion (cell_prob [i])
    for i in range (len (cell_prob)):
        for j in range (dilation):
            cell_prob [i] = binary_dilation (cell_prob [i])
    ret = np.array (cell_prob, dtype=np.uint8) * 255
    ret[0,0] *= 0
    ret[0,-1] = 0
    ret[0,:,-1] = 0
    ret[0,:,0] = 0
    return ret


# In[5]:


# show (data)


# In[6]:


img_prob = get_cell_prob (data [None], 0, 0) [0]
# show (img_prob)


# In[7]:


# show (label (img_prob), cm='tab20c')


# In[ ]:


from PIL import Image, ImageFilter
img_pil = Image.fromarray (img_prob)
cropped_img = img_pil.crop ((200, 200, 400, 400))
# blurred_img = cropped_img.filter (ImageFilter.GaussianBlur (radius=3))
for i in range (4):
    cropped_img = cropped_img.filter (ImageFilter.MaxFilter (size=5))

img_pil.paste (cropped_img, (200, 200, 400, 400))
img_pil.show ()

