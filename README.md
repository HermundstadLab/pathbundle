## Create Conda Environment
1. Download and install [miniconda](https://www.anaconda.com/download) (NOTE: No need to sign up anything!)
2. Create a new environment for the project
```shell
$ conda create --name treesegm python=3.13
$ conda activate treesegm
```
3. install packages
```shell
$ conda install -c conda-forge ipython ipykernel nbformat sphinx sphinx-rtd-theme myst-parser matplotlib opencv pandas multiprocess h5py scipy scikit-image scikit-learn networkx
```

## Install `src` Folder (Source Codes)
```shell
$ cd /dir_to_project_folder
$ conda activate treesegm
$ pip install -e .
```

## Configure `data` Folder
1. See **IN_DATA** in `align_1_dat_perspective.ipynb` to set update *snapshots* and *anchors*
2. See **IN_DATA** in `align_2_df_tile.ipynb` to upload annotated tile data


## Configure `src/metadata.py`
1. Modify `bigdata_dir` to load your dataset
2. For each mouse, add index and directories to the following:
    - `mouse_ids`
    - `csv_timestamp_dict`
    - `h5_dict`

## Run Codes
Run `scripts/*.ipynb`  in the following series:
1. `align_1,2`
2. `equal_1,2`
3. `bundle_1,2`


## View Documentation
1. Open the html file in `pathbundle/docs/build/html/index.html`