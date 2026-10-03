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
#
# ## MAIN_FUNCTIONS
# - `get_equal_hourly_intervals_for_all_sessions`
# - `gen_one_column_of_p_tensor_from_bundle`
# - `gen_p_tensor`
# - `get_corr_matrices`

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
mouse_id = 18 #int(sys.argv[1])

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
# load df
df_equal = pickle.load(open(in_dir / "df_equal", "rb"))
t0_base_all = df_equal[~df_equal.is_small].index.values # exclude jagged paths

# load bundle
t0_bundle_all, l_bundle_all = pickle.load(open(out_dir / "dat_bundle", "rb"))

# %% [markdown]
# ## LOCAL PARAMETERS

# %%
# set parameter space
n_bins_sessions = [16,16,16] # 8hrs/16 makes 30min bins
l_th_select = np.arange(1,21)
bin_select = np.arange(np.sum(n_bins_sessions))

# %%
n_l_th_select, n_bins_select = len(l_th_select), len(bin_select)
tt_bins_equal = get_tt_bins_equal(n_bins_sessions, df_equal)
tt_bins_select = [tt_bins_equal[x] for x in bin_select]


# %% [markdown]
# ## RUN

# %%
# MP (6m for 48 bins x 20 l_th)
def run_mp(lij):
    l_th, i, j = lij
    d_PB = get_pathbundle_distance(i, j, l_th, t0_bundle_all, l_bundle_all, tt_bins_select, t0_base_all)
    return d_PB

param = list(product(l_th_select, range(n_bins_select), range(n_bins_select)))
with multiprocess.Pool() as pool:
    d_PB_fl = pool.map(run_mp, param)
d_PB_tensor = np.array(d_PB_fl).reshape(n_l_th_select, n_bins_select, n_bins_select)

# %% [markdown]
# ## PICKLE

# %%
pickle.dump(d_PB_tensor, open(out_dir / 'd_PB_tensor', 'wb'))

# %% [markdown]
# ## TEST

# %%
il = 7
l_th = l_th_select[il]

plt.figure(figsize=(6.5,5), dpi=200)
plt.title(f'mouse {mouse_id}, l_th={l_th}')
plt.imshow(d_PB_tensor[il], cmap='binary', vmin=0, vmax=1)
plt.colorbar()
ticks = np.array([0,16,32,48]) - .5
plt.xticks(ticks=ticks, labels=['']*4)
plt.yticks(ticks=ticks, labels=['']*4)
plt.xlabel('bin j (30min)')
plt.ylabel('bin i (30min)')
