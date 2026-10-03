## PACKAGES
import os, sys, warnings, pickle, multiprocess, ast, inspect
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations, product, islice, groupby
from skimage.measure import label, regionprops
from collections import Counter
from scipy.ndimage.filters import maximum_filter
from scipy.signal import argrelextrema
from scipy.spatial import KDTree
import pandas as pd
from PIL import Image
import h5py
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import poisson
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon

# import sparse
from scipy.spatial import cKDTree
import networkx as nx
import re
import cv2


## SETTING
num_cpus = int(multiprocess.cpu_count() * 1)
np.set_printoptions(legacy="1.25")


## GENERAL MEASURES, SCORES
def fl(l):
    """_summary_

    _extended_summary_

    Args:
        l (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [x for y in l for x in y]


def js_sim(p1, p2, axis=0):
    """_summary_

    NOTE:
        - JSS = 1-JSD which is bounded between 0 and 1 when using base 2
        - JSS: Jensen-Shannon Similarity
        - JSD: Jensen-Shannon Divergence

    Args:
        p1 (_type_): _description_
        p2 (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    return 1 - jensenshannon(p1, p2, base=2, axis=axis)


def mutual_coverage(p1, p2, axis=-1):
    """_summary_

    NOTE:
        mutual coverages is defined as the product of
        1. how much p1 covers p2 and
        2. how much p2 covers p1

    Args:
        p1 (_type_): _description_
        p2 (_type_): _description_
        axis (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    # assert (p1.sum(-1) <= 1 + 1e-6).prod() == 1
    # assert (p2.sum(-1) <= 1 + 1e-6).prod() == 1
    #
    cover_1 = (p1 * (p2 > 0)).sum(axis)
    cover_2 = ((p1 > 0) * p2).sum(axis)
    mc = cover_1 * cover_2
    return mc


def base_coverage(p_base, p, axis=-1):
    """_summary_

    NOTE:
        base coverages is defined as how much p_base covers p

    Args:
        p_base (_type_): _description_
        p (_type_): _description_
        axis (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    assert (p_base.sum(-1) <= 1 + 1e-6).prod() == 1
    assert (p.sum(-1) <= 1 + 1e-6).prod() == 1
    #
    cover = (p * (p_base > 0)).sum(axis)
    return cover


def categorical_entropy(p, axis=-1, is_base_2=False):
    """use base 2 to compute how many bits given a discrete distribution

    _extended_summary_

    Args:
        p (_type_): _description_
        axis (int, optional): _description_. Defaults to -1.
        is_base_2 (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert np.isclose(np.sum(p, axis).mean(), 1)
    p = np.array(p)
    p += 1e-16
    if is_base_2:
        en = np.sum(-p * np.log2(p), axis)
    else:
        en = np.sum(-p * np.log(p) / np.log(len(p)), axis)
    return en


## UTILITY ALGORITHMS
def reduce_df_via_halving(df, attr, cumsum_frac, if_ascending):
    """_summary_

    NOTE:
        the name halving here does not necessarily mean dividing by 2

    Args:
        df (_type_): _description_
        attr (_type_): _description_
        cumsum_frac (_type_): _description_
        if_ascending (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_sort = df.sort_values(attr, ascending=if_ascending)
    attr_sort = df_sort[attr].values
    attr_cumsum = attr_sort.cumsum() / attr_sort.sum()
    n_kept = np.where(attr_cumsum > cumsum_frac)[0][0]
    dff = df_sort[:n_kept]
    return dff


def reduce_rows_via_halving(rows, i_column, cumsum_frac, if_ascending):
    """_summary_

    _extended_summary_

    Args:
        rows (_type_): _description_
        i_column (_type_): _description_
        cumsum_frac (_type_): _description_
        if_ascending (_type_): _description_

    Returns:
        _type_: _description_
    """
    # load
    row_arr = np.array(rows)
    column = row_arr[:, i_column]

    # sort
    if if_ascending:
        idx_sort = np.argsort(column)
    else:
        idx_sort = np.argsort(column)[::-1]
    row_arr_sort = row_arr[idx_sort]
    column_sort = column[idx_sort]

    # filter
    idx_removed = np.cumsum(column_sort) / column_sort.sum() > cumsum_frac
    idx_kept = ~idx_removed
    row_arr_kept = row_arr_sort[idx_kept]
    return row_arr_kept


def find_maxima_of_2d_array(z_array, maxima_mask=1.0, r_footprint=3, z_value_thres=0):
    """_summary_

    NOTE:
        - mask=1 for all elements is allowed to be a maximum
        - use the convention that flips x,y; i.e., axis_0:y, axis_1:x
        - keep only z_maxima > z_value_thres

    Args:
        z_array (_type_): _description_
        maxima_mask (float, optional): _description_. Defaults to 1.0.
        r_footprint (int, optional): _description_. Defaults to 3.
        z_value_thres (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    z_masked = z_array * maxima_mask
    z_conv = maximum_filter(z_masked, footprint=np.ones([r_footprint, r_footprint]))
    detected_peaks = (z_conv == z_masked) & (z_masked > z_value_thres)
    y_maxima, x_maxima = np.where(detected_peaks)
    z_maxima = z_array[y_maxima, x_maxima]
    return x_maxima, y_maxima, z_maxima


def find_repeated_sequences(data, size):
    """
    Finds all repeated sequences of a specified size within a list.

    Args:
        data (_type_): _description_
        size (_type_): _description_

    Returns:
        A list of tuples, where each tuple contains a repeated sequence and a list of its starting indices.
    """
    if size > len(data) or size <= 0:
        return []

    sequences = {}
    for i in range(len(data) - size + 1):
        sequence = tuple(data[i : i + size])
        if sequence in sequences:
            sequences[sequence].append(i)
        else:
            sequences[sequence] = [i]

    repeated_sequences = [
        (seq, indices) for seq, indices in sequences.items() if len(indices) > 1
    ]
    return repeated_sequences


def get_ii_segm_from_bool_list(occ_list):
    """_summary_

    _extended_summary_

    Args:
        occ_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    bool_list = np.array(occ_list) > 0
    ts = [0] + np.cumsum([len(list(y)) for x, y in groupby(bool_list)]).tolist()
    tts = np.array(list(zip(ts[:-1], ts[1:])))
    # bool_tt = [x == 1 for x, y in groupby(bool_list)]
    bool_tt = [x == 1 for x, y in groupby(bool_list)]
    tts_r = tts[bool_tt]
    return tts_r


## GET FUNCTION DEPENDENCY TREE
def is_module_function(module, func):
    """_summary_

    _extended_summary_

    Args:
        module (_type_): _description_
        func (_type_): _description_

    Returns:
        _type_: _description_
    """
    return inspect.isfunction(func) and inspect.getmodule(func) == module


def get_all_function_names(func_dir):
    """_summary_

    _extended_summary_

    Args:
        func_dir (_type_): _description_
    """
    # load source code directory
    exec("import " + func_dir, globals())

    func_name_all = [
        name
        for name, obj in inspect.getmembers(eval(func_dir))
        if is_module_function(eval(func_dir), obj)
    ]
    return func_name_all


def get_one_level_inner_functions(func_name, func_name_all, func_dir):
    """_summary_

    _extended_summary_

    Args:
        func_name (_type_): _description_
        func_name_all (_type_): _description_
        func_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    func = eval(func_dir + "." + func_name)
    node = ast.parse(inspect.getsource(func))

    inner_func_list = []
    for x in ast.walk(node):
        y = getattr(x, "id", None)
        if (y in func_name_all) & (y not in inner_func_list):
            inner_func_list.append(y)
    return inner_func_list


def print_auxiliary_functions(func_name, func_name_all, func_dir):
    """_summary_

    _extended_summary_

    Args:
        func_name (_type_): _description_
        func_name_all (_type_): _description_
        func_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    # iter to get edges and nodes
    nodes_list = [[(0, func_name)]]
    edge_list = []
    node_id = 0
    for iter in range(10):
        for parent_node_id, func_name in nodes_list[-1]:
            inner_nodes = get_one_level_inner_functions(
                func_name, func_name_all, func_dir
            )
            nodes_ = []
            for x in inner_nodes:
                node_id += 1
                nodes_.append((node_id, x))
                edge_list.append((parent_node_id, node_id))
            nodes_list.append(nodes_)

    # get graph
    nodes_dict = {x: y for x, y in fl(nodes_list)}
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    G = nx.relabel_nodes(G, nodes_dict)

    # print
    print(".. admonition:: Auxiliary functions:\n")
    print("    .. line-block::")
    # nx.write_network_text(G, with_labels=True)
    for y in [x for x in nx.generate_network_text(G, with_labels=True)]:
        print("    " * 2 + y)
    print("\n.. autosummary::")
    for x in list(G)[1:]:
        print(f"    {x}")



## STAT
def get_confidence_interval(x, axis):
    """c=1.96 for 95% CI"""
    x = np.array(x)
    mean = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    n_samples = np.sum(~np.isnan(x), axis=axis)
    ci = 1.96 * std / np.sqrt(n_samples)
    return ci, mean, std