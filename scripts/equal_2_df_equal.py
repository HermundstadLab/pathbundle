# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
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
# ## IN_DATA
# - `df_raw`
#
# ## OUT_DATA
# - `df_equal`
# - `df_raw`
#
# ## MAIN_FUNCTIONS
# - `gen_df_equal` *
# - `get_all_diameter`
# - `label_raw_traj_from_equalized_traj`

# %% [markdown]
# ## PACKAGE

# %%
# %load_ext autoreload
# %autoreload 2

from src.utils import *
from src.metadata import *
from src.equal import *

# %% [markdown]
# ## BASH PARAMETERS

# %%
mouse_id = 3#int(sys.argv[1])

# %%
# set directories
tag = f"mouse_{mouse_id}"
in_dir = project_dir / "results" / tag
out_dir = in_dir
#
if not os.path.exists(in_dir):
    os.makedirs(in_dir)

# %% [markdown]
# ## LOAD

# %%
# load metadata
n_sessions = n_sessions_dict[mouse_id]

# load perspective transform
M_dict, img_dict, xy_anchor_dict, base_key = pickle.load(open(project_dir / 'results' / "dat_perspective", "rb"))

# load df
df_raw = pickle.load(open(in_dir / "df_raw", "rb"))

# load tile data
df_tile = pickle.load(open(project_dir / 'results' / "df_tile", "rb"))
df_tile_mean = pickle.load(open(project_dir / 'results' / "df_tile_mean", "rb"))

# %% [markdown]
# ## GLOBAL PARAMETERS

# %%
upsampl_fac = 2
d_downsampl = d_tile / fs_equal  # this determines the point-to-point distance
d_pair_th = d_tile / fs_pair_th # used to assign wall following

# %% [markdown]
# ## RUN: generate df_equal from df_raw

# %%
## RUN 
# mouse 3: 2m for upsampl_fac=2
# mouse 10: 42m for upsampl_fac=2 (a bit too long, consider pre-splitting into two mouse ids)
df_equal = gen_df_equal(df_raw, d_downsampl, upsampl_fac=upsampl_fac)

# %% [markdown]
# ## RUN: attr small diameter to `df_equal` and `df_raw`

# %% [markdown]
# ### step 1: get small diameter of equalized traj from `df_equal` and assign to `df_raw`

# %%
## RUN (1m for mouse 10)
# get diameters
t0_all = df_equal.index.values
diameter_all, bool_small_all = get_all_diameter(t0_all, df_equal)

# attr
df_equal['diameter'] = diameter_all
df_equal['is_small'] = bool_small_all

# attr df_raw
df_raw = label_raw_traj_from_equalized_traj(bool_small_all, 'is_small', df_equal, df_raw)

# %%
df_raw.is_small.mean(), df_equal.is_small.mean()

# %% [markdown]
# ### step 2: label wall following

# %%
xy_wall = get_xy_wall()
xy_all = df_equal[['xe','ye']].values
d_wall_all = (((xy_all[:,None] - xy_wall[None])**2).sum(-1)**.5).min(-1)
bool_wall_all = d_wall_all <= d_pair_th

# attr
df_equal['is_wall'] = bool_wall_all

# attr df_raw
df_raw = label_raw_traj_from_equalized_traj(bool_wall_all, 'is_wall', df_equal, df_raw)

# %%
df_raw.is_wall.mean(), df_equal.is_wall.mean()

# %% [markdown]
# ### step 3: label jump

# %%
# append backward jump to dictionary
bwd_jump_dict = {x[::-1]: y[::-1]  for x,y in fwd_jump_dict.items()}
tiles_jump_dict = {**fwd_jump_dict, **bwd_jump_dict}

# dictionary to map jump labels to unique ids
label_fwd_uniq = sorted(set(fwd_jump_dict.values()))
jump_label_id_dict = {}
jump_id = 0
for label_fwd in label_fwd_uniq:
    jump_label_id_dict[label_fwd] = jump_id
    jump_id += 1
    #
    label_bwd = label_fwd[::-1]
    jump_label_id_dict[label_bwd] = jump_id
    jump_id += 1

# %%
# get jumps as 3-tile transition
tile_123_all = get_all_jumps_as_tile_123(df_equal)
bool_jump_all = [x in tiles_jump_dict for x in tile_123_all]

# get jump segments
tt_jump_list = get_ii_segm_from_bool_list(bool_jump_all)

# filter by jump speed
bool_jump_list = [df_equal[x:y].vg.max() > vg_jump_th for x,y in tt_jump_list]
tt_jump_list = tt_jump_list[bool_jump_list]

# padding each jump segment with 1-tile width (4 points)
tt_jump_padded = tt_jump_list.copy()
tt_jump_padded[:,0] -= fs_equal
tt_jump_padded[:,1] += fs_equal
bool_jump_padded = np.zeros(len(df_equal), dtype=bool)
for x,y in tt_jump_padded:
    bool_jump_padded[x:y] = True

# get jump labels
df_equal['tile_123'] = tile_123_all
jump_label_list = [tiles_jump_dict[x] for x in df_equal.loc[tt_jump_list[:,0]].tile_123]

# assign jump labels to all equal time
jump_label_all = np.zeros(len(df_equal), dtype=object) - 1
for (x,y),z in zip(tt_jump_padded, jump_label_list):
    jump_label_all[x:y] = [z] * (y-x)
    
# map to jump_id
jump_id_all = np.array([jump_label_id_dict.get(x, -1) for x in jump_label_all], dtype=int)

# attr
df_equal['is_jump'] = bool_jump_padded
df_equal['jump_label'] = jump_label_all
df_equal['jump_id'] = jump_id_all

# attr df_raw
df_raw = label_raw_traj_from_equalized_traj(bool_jump_padded, 'is_jump', df_equal, df_raw)

# %%
df_raw.is_jump.mean(), df_equal.is_jump.mean()

# %% [markdown]
# ## PICKLE

# %%
df_equal.to_pickle(out_dir / "df_equal")
df_raw.to_pickle(out_dir / "df_raw")

# %% [markdown]
# ## TEST

# %%
t0, t1 = len(df_equal) // 100 * 22, len(df_equal) // 100 * 23
rt0, rt1 = df_equal.loc[[t0, t1], "t_raw"].values
session = df_equal.loc[t0, "session"]
img = img_dict[(mouse_id, session)]
#
plt.figure(figsize=(8, 4), dpi=200)
plt.suptitle('test df_equal')
plt.subplot(121)
plt.title("raw")
plt.imshow(img, zorder=0, alpha=0.3, origin="lower")
xy = df_raw[rt0:rt1][["xg", "yg"]].values.T
bool_small = df_raw[rt0:rt1]["is_small"].values
plt.scatter(*xy, c=bool_small, cmap="coolwarm", s=2, edgecolors="none")
plt.axis("off")

plt.subplot(122)
plt.title("equalized")
plt.imshow(img, zorder=0, alpha=0.3, origin="lower")
xy = df_equal[t0:t1][["xe", "ye"]].values.T
bool_small = df_equal[t0:t1]["is_small"].values
plt.scatter(*xy, c=bool_small, cmap="coolwarm", s=3, edgecolors="none")
plt.axis("off")

plt.tight_layout()
plt.savefig(out_dir / "df_equal_test.png")

# %% [markdown]
# ### plot wall following

# %%
plt.figure(figsize=(8, 4), dpi=200)
plt.suptitle('test wall following')

dff = df_raw[df_raw.is_wall]
xy = dff[["xg", "yg"]].values
plt.subplot(121)
plt.title("raw")
plt.imshow(img, zorder=0, alpha=0.3, origin="lower")
plt.scatter(*xy.T, c='k', s=.1, lw=0)
plt.scatter(*xy_wall.T, c='lime', s=3, lw=0)
plt.axis("off")

dff = df_equal[df_equal.is_wall]
xy = dff[["xe", "ye"]].values
plt.subplot(122)
plt.title("equalized")
plt.imshow(img, zorder=0, alpha=0.3, origin="lower")
plt.scatter(*xy.T, c='k', s=1, lw=0)
plt.scatter(*xy_wall.T, c='lime', s=3, lw=0)
plt.axis("off")

plt.tight_layout()
plt.savefig(out_dir / "df_equal_wall_following_test.png")

# %% [markdown]
# ### plot jumps

# %%
dff = df_equal[df_equal.is_jump]
xy = dff[["xe", "ye"]].values.T
jump_id_all = dff.jump_id.values

# plot
img = img_dict[(mouse_id, 0)]
plt.figure(figsize=(6,5), dpi=200)
plt.title('color: jump_id')
plt.imshow(img, zorder=0, alpha=0.3, origin="lower")
plt.scatter(*xy, c=jump_id_all, s=1, lw=0, cmap='rainbow')
plt.axis("off")
plt.axis("equal")
plt.tight_layout()
plt.savefig(out_dir / "df_equal_jumps_test.png")

# %% [markdown]
# ### DEV: label jumps

# %%
img = img_dict[(mouse_id, 0)]

plt.figure(figsize=(11,5), dpi=200)
plt.subplot(121)
df_equal.vg.hist(bins=np.linspace(0, 40, 100), alpha=.5)
plt.xlabel("vg")

plt.subplot(122)
dff = df_equal[(df_equal.vg > 10) & (df_equal.vg < 40)]
xy_tile = df_tile_mean[['x', 'y']].values
plt.imshow(img, zorder=0, alpha=0.3, origin="lower")
xy = dff[["xe", "ye"]].values.T
plt.scatter(*xy, c='k', s=1, lw=0)
for tile_id, (x, y) in enumerate(xy_tile):
    plt.text(x, y, str(tile_id), color='r', fontsize=8, ha='center', va='center')
plt.axis("off")
plt.axis("equal")

plt.tight_layout()
