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
mouse_id = 20#int(sys.argv[1])

# %%
# tag = f"mouse_{mouse_id}"
# in_dir = project_dir / "results" / tag
# out_dir = in_dir
# #
# if not os.path.exists(in_dir):
#     os.makedirs(in_dir)

# %% [markdown]
# ## LOAD

# %%
# # load df
# df_equal = pickle.load(open(in_dir / "df_equal", "rb"))
# t0_base_all = df_equal[~df_equal.is_small].index.values # exclude jagged paths

# # load bundle
# t0_bundle_all, l_bundle_all = pickle.load(open(out_dir / "dat_bundle", "rb"))

# %%
d_PB_tensor_list = [pickle.load(open(project_dir / "results" / f"mouse_{mouse_id}/d_PB_tensor", "rb")) for mouse_id in mouse_ids]
d_PB_tensor_list = np.array(np.clip(d_PB_tensor_list, 0, 1))
n_mice, n_l_th, n_bins, _ = d_PB_tensor_list.shape

# %% [markdown]
# ## LOCAL PARAMETERS

# %%
l_th_select = np.arange(1,21)

# %%
d_PB_tensor_mean = np.nanmean(d_PB_tensor_list, 0)

# %% [markdown]
# ## TEST

# %%
# d_PB_tensor_mean = d_PB_tensor_list[[0,1,2,3,5,6]].mean(0)

il = 7
l_th = l_th_select[il]

plt.figure(figsize=(6.5,5), dpi=200)
plt.title(f'l_th={l_th}')
plt.imshow(d_PB_tensor_mean[il], cmap='binary', vmin=0, vmax=1)
# plt.imshow(d_PB_tensor_list[6][il], cmap='binary', vmin=0, vmax=1)
plt.colorbar()
ticks = np.array([0,16,32,48]) - .5
plt.xticks(ticks=ticks, labels=['']*4)
plt.yticks(ticks=ticks, labels=['']*4)
plt.xlabel('bin j (30min)')
plt.ylabel('bin i (30min)')


# %%
def get_dij_dict(n_bins):
    ij_list = sorted(list(combinations(range(n_bins), 2)) + [(x,x) for x in range(n_bins)])
    dij_dict = {}
    for i,j in ij_list:
        dij = j-i
        try:
            dij_dict[dij].append((i,j))
        except KeyError:
            dij_dict[dij] = [(i,j)]
    return dij_dict

def get_d_PB_mean_ci_for_one_dij(dij, l_th, dij_dict, d_PB_tensor_list):
    ij_select = dij_dict[dij]
    d_PB_select = fl([d_PB_tensor_list[:,l_th-1,*x] for x in ij_select])
    d_PB_ci, d_PB_mean, d_PB_std = get_confidence_interval(d_PB_select, axis=0)
    return d_PB_ci, d_PB_mean, d_PB_std

def get_d_PB_mean_ci_list(l_th, dij_dict, d_PB_tensor_list):
    d_PB_mean_list, d_PB_ci_list, d_PB_std_list = [], [], []
    for dij in dij_dict:
        d_PB_ci, d_PB_mean, d_PB_std = get_d_PB_mean_ci_for_one_dij(dij, l_th, dij_dict, d_PB_tensor_list)
        
        # append
        d_PB_mean_list.append(d_PB_mean)
        d_PB_ci_list.append(d_PB_ci)
        d_PB_std_list.append(d_PB_std)

    d_PB_mean_list = np.array(d_PB_mean_list)
    d_PB_ci_list = np.array(d_PB_ci_list)
    d_PB_std_list = np.array(d_PB_std_list)
    return d_PB_mean_list, d_PB_ci_list, d_PB_std_list


# %%
l_th_plot = [1,4,8,12,16,20]

# prep
dij_dict = get_dij_dict(n_bins)

# plot
plt.figure(figsize=(16,6), dpi=200)
plt.suptitle(f'pathbundle distance (d_PB) matrices at different threshold of path length (l_th); n_mice={len(mouse_ids)}; heatmap:mean(d_PB),  error band:std(d_PB)', fontsize=16)
for i, l_th in enumerate(l_th_plot):
    # iter
    d_PB_mean_list, d_PB_ci_list, d_PB_std_list = get_d_PB_mean_ci_list(l_th, dij_dict, d_PB_tensor_list)
    
    # plot matrix
    plt.subplot(2,6,i+1)
    plt.title(f'l_th = {l_th}')
    plt.imshow(d_PB_tensor_mean[l_th-1], aspect=1, cmap='binary', vmin=0, vmax=1)
    # ticks = np.array([0,16,32,48]) - .5
    plt.xticks(ticks=ticks, labels=['']*4)
    plt.yticks(ticks=ticks, labels=['']*4)
    plt.xlabel('bin j (30min)')
    plt.ylabel('bin i (30min)')
    # plt.axis('equal')
    # plt.colorbar()
    
    # plot
    plt.subplot(2,6,i+6+1)
    plt.plot(list(dij_dict)[1:], d_PB_mean_list[1:], lw=1)
    plt.fill_between(list(dij_dict)[1:], d_PB_mean_list[1:]-d_PB_std_list[1:], d_PB_mean_list[1:]+d_PB_std_list[1:], alpha=0.1, edgecolor='none')
    
    # setting
    plt.ylim(-.05, 1.05)
    plt.xlabel('lapse')
    plt.ylabel('average d_PB')
    ticks = np.array([0,16,32,48])
    labels = ['0 day', '1 day', '2 days', '3 days']
    plt.xticks(ticks=ticks, labels=labels)
    plt.grid(alpha=.2)

# setting
plt.tight_layout()

