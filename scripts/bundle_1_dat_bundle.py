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
df_equal = pickle.load(open(in_dir / "df_equal", "rb"))

# variables
t_equal_max = len(df_equal)
n_sessions = n_sessions_dict[mouse_id]
t_session_list = [df_equal[df_equal["session"] == x].index[0] for x in range(n_sessions)]
t_session_middle = np.array(t_session_list) + np.diff(t_session_list + [t_equal_max])/2

# %% [markdown]
# ## GLOBAL PARAMETERS

# %%
d_pair_th = d_tile / fs_pair_th

# %% [markdown]
# ## RUN: get overlapping paths

# %%
## MP (3m; 15m)
# prep
t0_base_all, bool_small_all, l_base_max_all, xy_all, xy_kdtree = get_all_base_t0_l(df_equal, l_bundle_max)

# mp
def run_mp(t0_l_base):
    # load
    t0_base, l_base_max = t0_l_base
    try:
        t0_overlap, l_overlap = get_bundle_for_one_t0_base(t0_base, l_base_max, xy_all, xy_kdtree, bool_small_all, df_equal, d_pair_th)
        return t0_overlap, l_overlap
    except Exception as e:
        print(f"Error processing t0_base {t0_base}: {e}")
        return None, None
    
# run
param = list(zip(t0_base_all, l_base_max_all))
with multiprocess.Pool() as pool:
    t0_overlap_all, l_overlap_all = zip(*pool.map(run_mp, param))

# %% [markdown]
# ## PICKLE

# %%
dat_bundle = t0_overlap_all, l_overlap_all
pickle.dump(dat_bundle, open(out_dir / "dat_bundle", "wb"))

# %% [markdown]
# ## TEST

# %% [markdown]
# ### path length: small-diameter vs not

# %%
# prep
l_max_all = np.array([max(x) if len(x) > 0 else -1 for x in l_overlap_all])

# %%
l_bin_max = l_max_all.max() + 2

# plot
plt.figure(figsize=(6,3), dpi=200)
_ = plt.hist(l_max_all[~bool_small_all], bins=range(l_bin_max), alpha=1, label='not')
_ = plt.hist(l_max_all[bool_small_all], bins=range(l_bin_max), alpha=1, label='small-diameter')
plt.xlabel("path length")
plt.legend()

# %% [markdown]
# ### example path bundle

# %%
np.where((l_max_all>=60) & (~bool_small_all))[0][:50]

# %%
t0_base = 8297 # 268477, 284243
idx_base = t0_base_all.tolist().index(t0_base)

t0_select, l_select = t0_overlap_all[idx_base], l_overlap_all[idx_base]
xy_bundle = fl([df_equal.loc[range(t0, t0+l), ['xg', 'yg']].values.tolist() + [[np.nan, np.nan]] for t0, l in zip(t0_select, l_select) if l>0])
xy_bundle = np.array(xy_bundle)

# plot
plt.figure(figsize=(12,3), dpi=200)
plt.subplot(121)
plt.scatter(t0_select, l_select, s=1, c='k', lw=0, alpha=1)
plt.axvline(t0_base, color='r', lw=1, linestyle='--')
for x in t_session_list: plt.axvline(x, color='k', linestyle=':', lw=.5, zorder=0)
_ = plt.xticks(t_session_middle, ['session0', 'session1', 'session2', 'session3', 'session4'][:n_sessions])
plt.ylabel("path length")
# plt.xlim([t_session_list[-2]+50000, t_session_list[-1]])


plt.subplot(122)
img = img_dict[(mouse_id, 0)]
plt.imshow(img, zorder=0, alpha=0.3, origin="lower")
plt.plot(*xy_bundle.T, lw=.5, c='k')
plt.plot(*df_equal.loc[range(t0_base, t0_base+max(l_select)), ['xg', 'yg']].values.T, lw=.5, c='r')
plt.axis("off")

plt.tight_layout()
