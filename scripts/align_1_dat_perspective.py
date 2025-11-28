# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: pathbundle
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## IN_DATA (manually configured)
# - step 1: 
#     1. create `data/cam_all_mice` folder
#     2. given a mouse, take a VLC snapshot from each session, rename it to `cam_mouseX_sessY.png`
#     3. place these snapshots in the folder
#
# - step 2: 
#     1. create `data/anchors_all_mice` folder
#     2. open the above snapshots in Illustrator, draw 4 black anchor points on the maze (to align with perspective transformation)
#     3. keep only anchor points, delete the snapshot.
#     4. save to `data/anchors_all_mice/cam_mouseX_sessY.jpg` 
#
# ## OUT_DATA
# - `dat_perspective`
# - `cam_aligned_all_mice`
#
# ## MAIN_FUNCTIONS
# - `get_xy_anchors_for_one_day`
# - `cv2.getPerspectiveTransform`
# - `cv2.warpPerspective`

# %% [markdown]
# ## PACKAGE

# %%
# %load_ext autoreload
# %autoreload 2

from src.utils import *
from src.metadata import *
from src.align import *

# %% [markdown]
# ## BASH PARAMETERS

# %%
# set directories
in_dir = project_dir / "data"
out_dir = project_dir / "results"
#
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# %% [markdown]
# ## GLOBAL PARAMETERS

# %%
# choose base anchors
base_key = (10,0) # mouse 10, session 6

# assign permutation to rotate anchors 90 degrees
perm_dict = {
    (6, 2): [1,3,0,2],
    (6, 3): [1,3,0,2],
        }

# %% [markdown]
# ## RUN: find perspective transformation by manual anchors

# %%
# load
dir_dict = {tuple([eval(y) for y in re.findall(r"\d+", str(x))]): align_data_dir/x for x in os.listdir(align_data_dir)}
dir_dict = {k:v for k,v in dir_dict.items() if len(k)==2}

# load xy anchors for all days
xy_anchor_dict = {x:get_xy_anchors_for_one_day(y, perm=perm_dict.get(x, None)) for x,y in dir_dict.items()}

# get perspective transform matrices
base_anchors = xy_anchor_dict[base_key]
M_dict = {x: cv2.getPerspectiveTransform(y, base_anchors) for x,y in xy_anchor_dict.items()}

# fill in missing days
key_ext = fl([[(x,z) for z in range(y)] for x,y in n_sessions_dict.items()])
for mouse, day in key_ext:
    try: 
        M_dict[(mouse, day)]
    except:
        try:
            M_dict[(mouse, day)] = M_dict[(mouse, day-1)]
        except:
            M_dict[(mouse, day)] = None

# %% [markdown]
# ## RUN: transform movie snapshots for trajectory background

# %%
# load image directories
img_dir_dict = {tuple([eval(y) for y in re.findall(r"\d+", str(x))]): snapshot_dir/x for x in os.listdir(snapshot_dir)}

# load base image
base_img = plt.imread(img_dir_dict[base_key])
base_img_shape = base_img.shape[:2]

# transform snapshots from raw movies
transformed_img_dict = {}
for mouse, day in key_ext:
    try: 
        img = plt.imread(img_dir_dict[(mouse, day)])
        M = M_dict[(mouse, day)]
        transformed_img = cv2.warpPerspective(img, M, base_img_shape[::-1])
        transformed_img_dict[(mouse, day)] = transformed_img
    except:
        try:
            transformed_img_dict[(mouse, day)] = transformed_img_dict[(mouse, day-1)]
        except:
            transformed_img_dict[(mouse, day)] = None

# %% [markdown]
# ## PICKLE

# %%
dat_perspective = M_dict, transformed_img_dict, xy_anchor_dict, base_key
pickle.dump(dat_perspective, open(out_dir / "dat_perspective", "wb"))

# %%
plt.figure(figsize=(10,4), dpi=200)
plt.subplot(121)
plt.title('mouse 10 session 0')
plt.imshow(transformed_img_dict[(10,0)])
plt.axis('off')

plt.subplot(122)
plt.title('mouse 3 session 0')
plt.imshow(transformed_img_dict[(3,0)])
plt.axis('off')
