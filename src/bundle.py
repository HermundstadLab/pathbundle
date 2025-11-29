from src.utils import *
from src.metadata import *


## GET BUNDLE TRAJECTORY FROM DATA
def get_all_base_t0_l(df_equal, l_bundle_max):
    """Extract all timepoints and their upper bounds of path length from equalized trajectory.

    NOTE:
        The upper bound ``l_bundle_max`` should be set to be longer than the longest repeatable path length. Default is 200, or ~50 tile widths.

    Args:
        df_equal (pd.DataFrame): DataFrame containing equalized trajectory data
        l_bundle_max (int): upper bound of the bundle length

    Returns:
        t0_base_all (np.ndarray): all timepoints of equalize trajectory
        bool_small_all (np.ndarray): to specify if a timepoint is small-diameter or not
        l_base_max_all (np.ndarray): upper bound of path length for each timepoint
        xy_all (np.ndarray): all ``(x,y)`` coordinates at each timepoint of equalized trajectory
        xy_kdtree (KDTree): kdtree of ``(x,y)`` coordinates for fast neighbor search
    """
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
    """From equalized trajectory, a path bundle is extracted as paths overlapping with the base path at timepoint ``t0_base``

    Algorithm:
        step 1: Load base path starting from ``(x_base, y_base)`` at ``t0_base``
        step 2: Find all timepoints in the equalized trajectory that are within distance ``d_th`` of ``(x_base, y_base)``
        step 3: Remove ``t0_overlap`` from self-overlap of the base path.
        step 4: Split the overlapping timepoints into either small-diameter or not; to make sure small diameter paths only overlap with small diameter base paths, and vice versa.
        step 5: For each overlapping timepoint, calculate the length of continuous overlap with the base path within distance ``d_th``
    Args:
        t0_base (int): base timepoint
        l_base_max (int): upper bound of the bundle length
        xy_all (np.ndarray): all (x,y) coordinates at each timepoint of equalized trajectory
        xy_kdtree (KDTree): kdtree of (x,y) coordinates for fast neighbor search
        bool_small_all (np.ndarray): to specify if a timepoint is small-diameter or not
        df_equal (pd.DataFrame): DataFrame containing equalized trajectory data
        d_th (float): distance threshold for neighbor search
        l_th (int, optional): a path in a bundle must have at least this length. Defaults to 2.

    Returns:
        t0_overlap.astype(np.ndarray of int32):
        l_overlap.astype(np.ndarray of int16):
    """
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
    """_summary_

    .. deprecated:: 1.0
        This function is for a constrained shuffle control, which is not used in the released version of **pathbundle**.

    Args:
        t0_base (_type_): _description_
        tl_overlap_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """_summary_

    .. deprecated:: 1.0
        This function is for a constrained shuffle control, which is not used in the released version of **pathbundle**.

    Args:
        t0_base (_type_): _description_
        bool_used (_type_): _description_
        pair_fixed_dict (_type_): _description_
        tl_overlap_dict (_type_): _description_
        l_base (_type_): _description_
        t_equal_max (_type_): _description_

    Returns:
        _type_: _description_
    """
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
def get_equal_hourly_intervals_for_all_sessions(df_raw, df_equal, duration=3600):
    """Compute breakpoints with equal-duration intervals for all sessions.

    Algorithm:
        step 1: For each session of the raw trajectory, compute equal-duration timepoints from start to end.
        step 2: Proceed to compute all sessions and concatenate them.

    Args:
        df_raw (pd.DataFrame): Raw trajectory data containing sessions and realtime.
        df_equal (pd.DataFrame): Equalized trajectory data.
        duration (int, optional): Duration (in seconds) for equal intervals. Defaults to 3600.

    Returns:
        hour_bins_equal (list of int): Breakpoints with equal-duration intervals for all sessions.
    """
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
    hour_bins_equal = fl([x[:-1] for x in ttttt_sessions_equal]) + [len(df_equal)]  # v1
    hour_bins_equal = fl([list(zip(x[:-1], x[1:])) for x in ttttt_sessions_equal])
    return hour_bins_equal


def gen_one_column_of_p_array_from_bundle(
    t0_base, t0_base_all, t0_overlap_all, l_overlap_all, hour_bins, l_th
):
    """Given a base path, compute total overlaps in each hourly bin ``hour_bins``.

    Algorithm:
        step 1: Load path bundle of the base path at ``t0_base``.
        step 2: Filter overlapping timepoints to keep only those within ``t0_base_all``; e.g., when considering only timepoints that is not small-diameter: ``t0_base_all = df_equal[~df_equal.is_small].index.values``.
        step 3: Assign each overlapping timepoint to a hourly bin defined in ``hour_bins``

    Args:
        t0_base (int): starting time of the base path
        t0_base_all (list of int): valid timepoints to consider for overlap
        t0_overlap_all (list of np.ndarray): overlapping timepoints for each base path
        l_overlap_all (list of np.ndarray): overlap lengths for each base path
        hour_bins (list of list of tuple): hourly bins for each session
        l_th (int): minimum overlap length threshold

    Returns:
        l_overlap_hist (np.ndarray): total overlaps in each hourly bin
    """
    # load base
    t0_overlap_list = t0_overlap_all[t0_base]
    l_overlap_list = l_overlap_all[t0_base]

    # filter (this filtering slows things down by 3-fold)
    bool_r = np.isin(t0_overlap_list, t0_base_all)
    t0_overlap_r = t0_overlap_list[bool_r]
    l_overlap_r = l_overlap_list[bool_r]

    # assign t0_overlap to bins
    bin_overlap_r = np.digitize(t0_overlap_r, fl(hour_bins))

    # keep only odd bins (even bins is outside of hour_bins)
    bins_kept = np.arange(len(fl(hour_bins)))[1::2]
    l_overlap_groups = [l_overlap_r[bin_overlap_r == x] for x in bins_kept]

    # get histogram for hour_bins
    l_overlap_hist = np.array(
        [x[x >= l_th].sum() for x in l_overlap_groups], dtype=np.int16
    )
    # get p_column
    p_column = l_overlap_hist / (l_overlap_hist.sum() + 1e-12)
    return p_column


def gen_p_tensor(p_arr, t0_base_groups):
    """Reshape p_array into ``p_tensor`` and ``p_pillars``.

    Algorithm:
        step 1: Compute ``group_bins`` based on ``t0_base_groups``
        step 2: split ``p_arr`` into slab groups according to ``group_bins``
        step 3: transpose probability slabs into probability pillars ``p_pillars`` each with shape ``(n_bins, n_base_in_bin)``
        step 4: recast ``p_pillars`` into ``p_tensor`` format, where p_tensor[i, j] loads probabilities of all base paths in bin i that are overlapping with bin j, i.e., ``p(overlap with j | a base in i)``

    Args:
        p_arr (np.ndarray): probability array for ``(base time k, hour bin i)``
        t0_base_groups (list of np.ndarray): base timepoints grouped by hour bins

    Returns:
        p_tensor (list of np.ndarray): probability tensor with index ``(hour bin i, hour bin j, base time k)``
        p_pillars (list of np.ndarray): list of ``n_bins`` probability pillars each with shape ``(n_bins, n_base_in_bin)``

    Example:
        For loading all base timepoints in bin i that overlap with bin j::

            p_list = p_tensor[i,j]
            n_base_in_bin = len(p_list)

        Note that ``p_tensor`` and ``p_pillars`` are related by::

            np.stack(p_tensor[i,j]) == p_pillars[i][j]
    """
    group_bins = np.cumsum([0] + [len(x) for x in t0_base_groups])
    n_bins = len(t0_base_groups)
    p_pillars = [p_arr[x:y].T for x, y in zip(group_bins[:-1], group_bins[1:])]
    p_tensor = np.zeros((n_bins, n_bins), dtype=object)
    for i, j in product(range(n_bins), repeat=2):
        p_tensor[i, j] = p_pillars[i][j]
    return p_tensor, p_pillars


# def get_one_cij_for_corr_matrices(i, j, p_tensor, ternary_bins=[0, 1 / 3, 2 / 3, 1]):
#     """_summary_

#     Used in :py:func:`get_corr_matrices`.

#     Args:
#         i (int): index of the first interval
#         j (int): index of the second interval
#         p_tensor (list of np.ndarray): probability tensor
#         ternary_bins (list, optional): _description_. Defaults to [0, 1 / 3, 2 / 3, 1].

#     Returns:
#         p_persist (float): probability of finding a persist path from interval i to j
#         p_emerge (float): probability of finding an emerging path from interval i to j
#     """
#     p_i_sum = p_tensor[i][[i, j]].sum(0)
#     p_ii = p_tensor[i][i][p_i_sum > 0] / p_i_sum[p_i_sum > 0]
#     count_dict = Counter(np.digitize(p_ii, ternary_bins[1:-1]))
#     p_persist = count_dict[1] / len(p_ii)
#     p_emerge = count_dict[2] / len(p_ii)  # vanishing if i>j
#     return p_persist, p_emerge


def get_one_cij_for_corr_matrices(i, j, p_tensor, ternary_bins=[0, 1 / 3, 2 / 3, 1]):
    """Compute one entry ``c_ij`` of correlation matrices from probability tensor.

    Used in :py:func:`get_corr_matrices`.

    Algorithm:
        step 1: For an hour bin i, isolate all overlapping paths with hour bins i and j with nonzero probability, i.e, ``p_k(i|i) = p(overlap with bin i | base k in bin i)`` and ``p_k(j|i) = p(overlap with bin j | base k in bin i)``.
        step 2: Renormalize the probabilities so that ``p_k(i|i) + p_k(j|i) = 1``.
        step 3: For each base path k, classify the renormalized probabilities into persistent or emerging paths based on ``ternary_bins``; i.e., persistent paths have ``p_k(j|i)`` in ``[1/3, 2/3)``, and emerging paths have ``p_k(j|i)`` in ``[2/3, 1]`` for ``j>i``.
        step 4: Compute the fraction of persistent (``p_persist``) and emerging paths (``p_emerge``) by summing over all base paths k.

    Args:
        i (int): index of the base hour bin
        j (int): index of the target hour bin
        p_tensor (list of np.ndarray): probability tensor (see :py:func:`gen_p_tensor` for details)
        ternary_bins (list, optional): For classify whether a base path is persistent or emerging. Defaults to [0, 1/3, 2/3, 1].

    Returns:
        p_persist (float): probability of finding a persist path from interval i to j
        p_emerge (float): probability of finding an emerging path from interval i to j

    NOTE:
        ``p_emerge`` is ``p_vanish`` if ``i>j``
    """
    p_i_sum = np.stack(p_tensor[i, [i, j]]).sum(0)
    p_ii = (
        p_tensor[i, i][p_i_sum > 0] / p_i_sum[p_i_sum > 0]
    )  # renormalize to bin i and j only
    count_dict = Counter(np.digitize(p_ii, ternary_bins[1:-1]))
    p_persist = count_dict[1] / len(p_ii)
    p_emerge = count_dict[2] / len(p_ii)  # vanishing if i>j
    return p_persist, p_emerge


def get_corr_matrices(p_tensor):
    """Compute entire correlation matrices from probability tensor.

    .. admonition:: Auxiliary functions:

        .. line-block::
            ╙── :py:func:`get_corr_matrices`
                └─╼ :py:func:`get_one_cij_for_corr_matrices`

    .. autosummary::
        get_one_cij_for_corr_matrices

    Args:
        p_tensor (np.ndarray of np.ndarray): probability tensor (see :py:func:`gen_p_tensor` for details)

    Returns:
        persist_arr (np.ndarray): matrix of persistent path probabilities
        emerge_arr (np.ndarray): matrix of emerging path probabilities
    """
    persist_arr = np.zeros((len(p_tensor), len(p_tensor)))
    emerge_arr = np.zeros((len(p_tensor), len(p_tensor)))
    for i, j in product(range(len(p_tensor)), repeat=2):
        persist_arr[i, j], emerge_arr[i, j] = get_one_cij_for_corr_matrices(
            i, j, p_tensor
        )
    return persist_arr, emerge_arr
