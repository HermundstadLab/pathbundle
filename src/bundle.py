from src.utils import *
from src.metadata import *


## GET BUNDLE TRAJECTORY FROM DATA
def get_all_base_t0_l(df_equal, l_bundle_max):
    t0_base_all = df_equal.index.values
    xy_all = df_equal[["xg", "yg"]].values
    xy_kdtree = KDTree(xy_all)
    bool_small_all = df_equal.is_small.values
    l_base_max_all = [
        len([list(y) for _, y in groupby(bool_small_all[x : x + l_bundle_max])][0])
        for x in t0_base_all
    ]
    return t0_base_all, bool_small_all, l_base_max_all, xy_all, xy_kdtree


def get_bundle_for_one_t0_base(
    t0_base, l_base_max, xy_all, xy_kdtree, bool_small_all, df_equal, d_th, l_th=2
):
    # get filter based on whether base is small
    is_base_small = bool_small_all[t0_base]
    if not is_base_small:
        bool_small_all = ~bool_small_all

    # load base path
    xys_base = df_equal.loc[t0_base : t0_base + l_base_max - 1, ["xg", "yg"]].values
    xy_base = xys_base[0]

    # override
    l_base_max = len(xys_base)

    # find neighbors
    t0_overlap = np.sort(xy_kdtree.query_ball_point(xy_base, d_th))
    t0_overlap = t0_overlap[t0_overlap < len(df_equal) - l_base_max]

    # filter self
    t0_overlap = t0_overlap[np.abs(t0_overlap - t0_base) > l_base_max]

    # filer either small or not
    bool_t0_overlap = bool_small_all[t0_overlap]
    t0_overlap = t0_overlap[bool_t0_overlap]
    if len(t0_overlap) == 0:
        return np.array([]), np.array([])

    # find path length
    xy_all_ext = np.vstack([xy_all, np.zeros([l_base_max, 2]) + np.nan])
    xys_overlap = np.stack(
        [xy_all_ext[t0_overlap + T] for T in range(l_base_max)], axis=1
    )  # use ext to align array
    bool_xy_overlap = ((xys_overlap - xys_base[None]) ** 2).sum(-1) ** 0.5 < d_th

    # fragment overlap by small or not
    bool_small_overlap = np.stack(
        [bool_small_all[x : x + l_base_max] for x in t0_overlap]
    )
    bool_xy_small_overlap = bool_xy_overlap & bool_small_overlap

    # get length of continuous segment for both conditions
    l_overlap = np.array(
        [len([list(y) for x, y in groupby(z)][0]) for z in bool_xy_small_overlap]
    )

    # filter by l_th
    bool_kept = l_overlap >= l_th
    t0_overlap = t0_overlap[bool_kept]
    l_overlap = l_overlap[bool_kept]
    return t0_overlap.astype(np.int32), l_overlap.astype(np.int16)


## CONSTRAINED SHUFFLE CONTROL
def get_one_fixed_pair(t0_base, tl_overlap_dict):
    # load
    try:
        t0_overlap, l_overlap = tl_overlap_dict[t0_base]
    except:
        return None
    # if len(l_overlap)==0:
    #     return None
    # choose one max length
    l_max = l_overlap.max()
    t0_fixed = np.random.choice(t0_overlap[l_overlap == l_max])
    return t0_fixed


def shuffle_one_row(
    t0_base, bool_used, pair_fixed_dict, tl_overlap_dict, l_base, t_equal_max
):
    # load
    try:
        t0_overlap, l_overlap = tl_overlap_dict[t0_base]
    except:
        return np.array([]), np.array([])
    t0_l_dict = {x: y for x, y in zip(t0_overlap, l_overlap)}
    t0_max = pair_fixed_dict[t0_base]
    l_max = [t0_l_dict[x] for x in t0_max]

    # fix max length
    bool_fixed = bool_used.copy()
    bool_fixed[t0_max] = True

    # fix self
    t_lower = np.clip(t0_base - l_base, 0, t_equal_max - l_base)
    t_upper = np.clip(t0_base + l_base, 0, t_equal_max - l_base)
    bool_fixed[t_lower : t_upper + 1] = True

    t0_fixed = np.where(bool_fixed)[0]
    t0_unfixed_nz = list(set(t0_overlap) - set(t0_fixed))
    l_unfixed = [t0_l_dict[x] for x in t0_unfixed_nz]

    # shuffle unfixed
    t0_unfixed = np.where(~bool_fixed)[0]
    np.random.shuffle(t0_unfixed)

    # pack
    t0_overlap_1 = np.hstack([t0_unfixed[: len(l_unfixed)], t0_max])
    l_overlap_1 = np.array(l_unfixed + l_max)
    return t0_overlap_1, l_overlap_1


## HOUR-TO-HOUR CORRELATION MATRIX
def get_equal_intervals_for_all_sessions(df_raw, df_equal, duration):
    # get t0, t1 of each session
    n_sessions = df_raw.session.nunique()
    tt_sessions_realtime = [
        df_raw[df_raw.session == x].realtime.values[[0, -1]] for x in range(n_sessions)
    ]

    # get t0s for each equal intervals
    ttttt_sessions_realtime = [
        np.linspace(x, y, int((y - x) / duration) + 1) for x, y in tt_sessions_realtime
    ]
    ttttt_sessions_equal = [
        [(df_equal[df_equal.session == sess].realtime - y).abs().idxmin() for y in x]
        for sess, x in enumerate(ttttt_sessions_realtime)
    ]

    # trim off last and concatenate
    t_bins_equal = fl([x[:-1] for x in ttttt_sessions_equal]) + [len(df_equal)]  # v1
    t_bins_equal = fl([list(zip(x[:-1], x[1:])) for x in ttttt_sessions_equal])
    return t_bins_equal


def get_l_overlap_hist_for_one_bundle(
    t0_base, t0_base_all, t0_overlap_all, l_overlap_all, t_bins, l_th
):
    # load base
    t0_overlap_list = t0_overlap_all[t0_base]
    l_overlap_list = l_overlap_all[t0_base]

    # filter (this filtering slows things down by 3-fold)
    bool_r = np.isin(t0_overlap_list, t0_base_all)
    t0_overlap_r = t0_overlap_list[bool_r]
    l_overlap_r = l_overlap_list[bool_r]

    # assign t0_overlap to bins
    # bin_overlap_list = np.digitize(t0_overlap_list, t_bins[1:-1]) # v1
    bin_overlap_r = np.digitize(t0_overlap_r, fl(t_bins))

    # keep only odd bins (even bins is out of t_bins)
    bins_kept = np.arange(len(fl(t_bins)))[1::2]
    l_overlap_groups = [l_overlap_r[bin_overlap_r == x] for x in bins_kept]

    # get histogram for each bin
    l_overlap_hist = np.array(
        [x[x >= l_th].sum() for x in l_overlap_groups], dtype=np.int16
    )
    return l_overlap_hist


ternary_bins = [0, 1 / 3, 2 / 3, 1]


def get_corr_for_one_intervals_ij(i, j, p_tensor):
    p_i_sum = p_tensor[i][[i, j]].sum(0)
    p_ii = p_tensor[i][i][p_i_sum > 0] / p_i_sum[p_i_sum > 0]
    count_dict = Counter(np.digitize(p_ii, ternary_bins[1:-1]))
    p_persist = count_dict[1] / len(p_ii)
    p_emerge = count_dict[2] / len(p_ii)  # vanishing if i>j
    return p_persist, p_emerge


def get_corr_matrices(p_tensor):
    persist_arr = np.zeros((len(p_tensor), len(p_tensor)))
    emerge_arr = np.zeros((len(p_tensor), len(p_tensor)))
    for i, j in product(range(len(p_tensor)), repeat=2):
        persist_arr[i, j], emerge_arr[i, j] = get_corr_for_one_intervals_ij(
            i, j, p_tensor
        )
    return persist_arr, emerge_arr
