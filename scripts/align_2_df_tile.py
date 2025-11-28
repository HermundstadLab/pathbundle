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
# ## IN_DATA (manually configured)
# - step 1: 
#     1. draw hexagons to annotate tiles in Illustrator
#     2. Make sure to name the layer identical to `tile_id`.
#     3. Check the "transparency" box when using `MultiExporter.jsx` to export hexagons.
# - step 2 (**TODO**: incoporating height map and assigning to tile_id)
#
# ## OUT_DATA
# - `df_tile`
# - `df_tile_mean`
#
# ## MAIN_FUNCTIONS
# - `gen_one_dff_hexagon`

# %% [markdown]
# ## VERSION

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
# ## RUN: get `df_tile`, `df_tile_mean`

# %%
# load
hexagon_dir_list = get_hexagon_dir_list(tile_data_dir)

# mp
with multiprocess.Pool() as pool:
    dff_list, xy_mean_list = zip(*pool.map(gen_one_dff_hexagon, hexagon_dir_list))
    
# pack
df, df_mean = gen_df_tile(dff_list, xy_mean_list)

# %% [markdown]
# ## PICKLE

# %%
pickle.dump(df, open(out_dir / "df_tile", "wb"))
pickle.dump(df_mean, open(out_dir / "df_tile_mean", "wb"))

# %% [markdown]
# ## TEST

# %%
plt.figure(figsize=(5,5), dpi=200)

# plot tile pixels
for tile_id in df.tile_id.unique():
    x_, y_ = df[df.tile_id==tile_id][['x','y']].values.T
    plt.scatter(x_, y_, s=3, lw=0)
plt.axis('equal')

# plot tile centroids
x_, y_ = df_mean[['x', 'y']].values.T
plt.scatter(x_, y_, c='k', lw=0, s=5)
plt.axis('equal')

plt.savefig(out_dir / "tile_ids_test.png")
