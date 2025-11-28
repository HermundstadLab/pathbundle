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
# - `df_equal`
# - `df_pair_r2`
#
# ## OUT_DATA
# - `df_segm_root`
# - `dat_segm_root`

# %% [markdown]
# ## VERSION

# %% [markdown]
# ## PACKAGE

# %%
# %load_ext autoreload
# %autoreload 2

from src.utils import *
from src.metadata import *
from src.bundle import *

# %% [markdown]
# ## BASH PARAMETERS

# %%
mouse_id = 3#int(sys.argv[1])

# %%
tag = f"mouse_{mouse_id}"
in_dir = project_dir / "results" / tag
out_dir = in_dir
#
if not os.path.exists(in_dir):
    os.makedirs(in_dir)

# %% [markdown]
# ## LOAD

# %%
# load perspective transform
M_dict, img_dict, xy_anchor_dict, base_key = pickle.load(open(project_dir / 'results' / "dat_perspective", "rb"))

# load df
df_raw = pickle.load(open(in_dir / "df_raw", "rb"))
df_equal = pickle.load(open(in_dir / "df_equal", "rb"))

# load bundle
t0_overlap_all, l_overlap_all = pickle.load(open(out_dir / "dat_bundle", "rb"))

# variables
t_equal_max = len(df_equal)
n_sessions = n_sessions_dict[mouse_id]
t_session_list = [df_equal[df_equal["session"] == x].index[0] for x in range(n_sessions)]
t_session_middle = np.array(t_session_list) + np.diff(t_session_list + [t_equal_max])/2

# %% [markdown]
# ## GLOBAL PARAMETERS

# %%
# d_pair_th = d_tile / fs_pair_th
# l_bundle_max = 200
# l_bundle_min = 4

# %% [markdown]
# ## INPUT
# ### select `t_bins`
# - `get_equal_intervals_for_all_sessions` gets 1-hour bins across all sessions
# - you can also replace this cell to define your own `t_bins` here

# %%
# this function gets 1-hour bins across all sessions
t_bins_equal = get_equal_intervals_for_all_sessions(df_raw, df_equal, duration=3600)

# %%
# select bins
t_bins_select = [t_bins_equal[x] for x in [2,3,5,6]]
# t_bins_select = t_bins_equal

# %% [markdown]
# ### filter `t0_base_all` by selected path labels

# %%
# choose labels to filter
# t0_excluded = df_equal[df_equal.is_small].index.values # checking excluded is faster
t0_base_all = df_equal[~df_equal.is_small].index.values # checking excluded is faster

# %%
# split t0_base_all into groups
t0_base_groups = [t0_base_all[np.isin(t0_base_all, range(x, y))] for x,y in t_bins_select]

# %% [markdown]
# ## RUN
# ### get `p_tensor` for `t_bins`

# %%
## MP (15s)
# prep
t0_base_groups_fl = np.hstack(t0_base_groups)
group_bins = np.cumsum([0] + [len(x) for x in t0_base_groups])

def run_mp(t0_base):
    l_overlap_hist = get_l_overlap_hist_for_one_bundle(t0_base, t0_base_all, t0_overlap_all, l_overlap_all, t_bins_select, l_th=l_bundle_min)
    return l_overlap_hist

with multiprocess.Pool() as pool:
    l_overlap_hist_all = pool.map(run_mp, t0_base_groups_fl)
    
# pack
l_arr = np.array(l_overlap_hist_all)
p_arr = l_arr / (l_arr.sum(1) + 1e-12)[:, None]
p_tensor = [p_arr[x:y].T for x,y in zip(group_bins[:-1], group_bins[1:])]

# %%
p_tensor[0][[0,2]]

# %%
t0_base_groups[0].shape

# %% [markdown]
# ### correlation matrix v1

# %%
# prep
p_arr_binned = np.array([x.sum(1) for x in p_tensor])
p_arr_binned = p_arr_binned / (p_arr_binned.sum(1) + 1e-12)[:, None]

# plot
plt.figure(figsize=(6.5,5), dpi=200)
plt.imshow(p_arr_binned.T, aspect=1, interpolation='nearest', cmap='Greys')
plt.colorbar()
plt.xlabel('interval j (1-hr)')
plt.ylabel('interval i (1-hr)')

# %% [markdown]
# ### correlation matrix v2

# %%
persist_arr, emerge_arr = get_corr_matrices(p_tensor)

# %%
plt.figure(figsize=(10,4), dpi=200)
plt.subplot(121)
plt.title('p(persist)')
plt.imshow(persist_arr + persist_arr.T/2, aspect=1, interpolation='nearest', cmap='Greys')
plt.xlabel('interval j (1-hr)')
plt.ylabel('interval i (1-hr)')

plt.subplot(122)
plt.title('p(emerge) for i<j, p(vanish) for i>j')
plt.imshow(emerge_arr, aspect=1, interpolation='nearest', cmap='Greys')
plt.xlabel('interval j (1-hr)')
plt.ylabel('interval i (1-hr)')
