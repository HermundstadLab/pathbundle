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
# - `bigdata_dir/*.h5` for trajectory
# - `bigdata_dir/*.csv` for timestamp
# - `dat_perspective`
#
# ## OUT_DATA
# - `df_raw`
#
# ## MAIN_FUNCTIONS
# - `gen_df_raw_for_n_sessions`

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
mouse_id = 3 #int(sys.argv[1])

# %%
# set directories
tag = f"mouse_{mouse_id}"
in_dir = project_dir / "results" / tag
out_dir = in_dir
#
if not os.path.exists(in_dir):
    os.makedirs(in_dir)

# %% [markdown]
# ## LOCAL PARAMETERS

# %%
# this remove some tracking errors
xy_removed_list = []

# split xy_traj to batches for fast assigning tile ids 
batchsize = 100000

# %% [markdown]
# ## LOAD

# %%
# load metadata
n_sessions = n_sessions_dict[mouse_id]
sigma_smooth = sigma_smooth_dict[mouse_id]

# load perspective transform
M_dict, img_dict, xy_anchor_dict, base_key = pickle.load(open(project_dir / 'results' / "dat_perspective", "rb"))

# load tile data
if use_tile_data:
    df_tile = pickle.load(open(project_dir / 'results' / "df_tile", "rb"))

# %% [markdown]
# ## RUN: generate df_raw for many days


# %%
# run to concate multiple sessions (13s)
df_raw = gen_df_raw_for_n_sessions(mouse_id, n_sessions, xy_removed_list, sigma_smooth, M_dict)

# %% [markdown]
# ## RUN: assign tile ids to raw trajectory

# %%
## RUN (10s)
if use_tile_data:
    # prep
    xy_all = df_raw[['xg', 'yg']].values
    xy_pixel_all = df_tile[['x', 'y']].values
    tile_id_all = df_tile.tile_id.values

    def run_mp(xy_batch):
        tile_id_batch = [get_one_tile_id(xy, xy_pixel_all, tile_id_all) for xy in xy_batch]
        return tile_id_batch

    # mp
    xy_batch_list = [xy_all[i:i+batchsize] for i in range(0, len(xy_all), batchsize)]
    with multiprocess.Pool() as pool:
        tile_id_batch_list = pool.map(run_mp, xy_batch_list)
    tile_id_all = np.hstack(tile_id_batch_list)

    # df
    df_raw['tile_id'] = tile_id_all

# %% [markdown]
# ## PICKLE

# %%
df_raw.to_pickle(out_dir / "df_raw")

# %% [markdown]
# ## TEST

# %%
fig_width = 3
n_col = min(5, n_sessions)
n_row = int(np.ceil(n_sessions / n_col))
#
plt.figure(figsize=(fig_width*n_col,fig_width*n_row), dpi=200)
plt.suptitle(f"test df_raw, {n_sessions} sessions")
for session in range(n_sessions):
    img = img_dict[(mouse_id, session)]
    dff = df_raw[df_raw["session"] == session]
    plt.subplot(n_row, n_col, session+1)
    plt.imshow(img, cmap='gray', origin='lower', alpha=.1)
    plt.plot(*dff[['xg', 'yg']].values.T, lw=.05, c='k')
    plt.axis('equal')
    plt.axis('off')
# plt.tight_layout()
plt.savefig(out_dir / "df_raw_sessions_test.png")
