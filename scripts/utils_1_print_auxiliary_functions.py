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

# %%
# %load_ext autoreload
# %autoreload 2

from src.utils import *

# %%
func_dir = 'src.equal'
func_name = 'gen_df_equal'
func_name_all = get_all_function_names(func_dir)
print_auxiliary_functions(func_name, func_name_all, func_dir)

# %%
func_name_all
