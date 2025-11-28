from src.utils import *
from src.metadata import *


## LOAD RAW TRAJECTORY
def read_h5py(load_dir):
    """Convert h5 file to python dictionary.

    Used in: :py:func:`gen_df_raw_for_n_sessions`

    Args:
        load_dir (str): path to h5py file

    Returns:
        keys (list): list of keys in the h5py file
        data_dict (dict): dictionary of data in the h5py file
        tracksize (int): number of frames in the trajectory
    """
    with h5py.File(load_dir, "r") as f:
        # print("Keys: %s" % f.keys())
        keys = list(f.keys())
        data_dict = {x: f[x][()] for x in keys}
    tracksize = data_dict["tracks"].shape[-1]
    return keys, data_dict, tracksize


def get_realtime_all(load_dir, tracksize):
    """Load timestamp csv file.

    Used in: :py:func:`gen_df_raw_for_n_sessions`

    Args:
        load_dir (str): path to timestamp csv file
        tracksize (int): number of frames in the trajectory
    Returns:
        realtime_all (np.ndarray): array of timestamps in seconds
    """
    df_timestamp = pd.read_csv(load_dir, header=None)
    df_timestamp.columns = ["datetime"]
    df_timestamp["datetime"] = pd.to_datetime(
        df_timestamp["datetime"]
    )  # in nano-seconds
    df_timestamp["seconds"] = df_timestamp["datetime"].values.astype(int) / 1e9
    df_timestamp["seconds"] = df_timestamp["seconds"] - df_timestamp["seconds"].iloc[0]
    realtime_all = df_timestamp["seconds"].values
    if len(realtime_all) >= tracksize:
        realtime_all = realtime_all[:tracksize]
    else:
        realtime_all = np.concatenate(
            [realtime_all, np.full(tracksize - len(realtime_all), np.nan)]
        )
    return realtime_all


def gen_dff_raw(data_dict, realtime_all):
    """Conver data_dict to dataframe.

    Used in: :py:func:`gen_df_raw_for_n_sessions`

    Args:
        data_dict (dict): dictionary of data in the h5 file
        realtime_all (np.ndarray): array of timestamps in seconds
    Returns:
        df (pd.DataFrame): dataframe of trajectory in one session
    """
    # load keypoint tracks
    tracks = data_dict["tracks"][0]
    x0, y0 = tracks[:, 0]  # nose
    x1, y1 = tracks[:, 1]  # left ear
    x2, y2 = tracks[:, 2]  # right ear
    x3, y3 = tracks[:, 3]  # bum

    # compute raw traj
    xn, yn = x0, y0
    xb, yb = x3, y3
    xc, yc = (x0 + x3) / 2, (y0 + y3) / 2  # centroid
    xh, yh = x0 - x3, y0 - y3  # head direction

    # linear interp to fill in missing frames (nan)
    df = pd.DataFrame(
        {
            "xc": xc,
            "yc": yc,
            "xh": xh,
            "yh": yh,
            "xn": xn,
            "yn": yn,
            "xb": xb,
            "yb": yb,
            "xr": x2,
            "yr": y2,
            "xl": x1,
            "yl": y1,
            "realtime": realtime_all,
        }
    )
    # df = df.interpolate("linear", limit_direction="both")
    # t_valid = max([np.where(~np.isnan(x))[0][0] for x in df.values.T])
    # df = df[t_valid:]
    # df.index = range(len(df))
    return df


def interpolate_df_raw(df_raw):
    """Interpolate missing values in raw dataframe.

    Used in: :py:func:`gen_df_raw_for_n_sessions`

    Args:
        df_raw (pd.DataFrame): raw dataframe with missing values

    Returns:
        df (pd.DataFrame): dataframe with interpolated values
    """
    df = df_raw.interpolate("linear", limit_direction="both")
    t_valid = max([np.where(~np.isnan(x))[0][0] for x in df.values.T])
    df = df[t_valid:]
    df.index = range(len(df))
    return df


def align_xy_with_perspective_transform(xy_list, M):
    """Apply perspective transform to xy coordinates.

    Used in: :py:func:`gen_df_raw_for_one_session`

    Args:
        xy_list (np.ndarray): array of xy coordinates
        M (np.ndarray): perspective transform matrix

    Returns:
        xy_list_transform (np.ndarray): transformed xy coordinates
    """
    xy_list_transform = cv2.perspectiveTransform(xy_list[None], M)[0]
    return xy_list_transform


def gen_df_raw_for_one_session(mouse_id, session, xy_removed_list, sigma_xy, M):
    """Load h5, timestamp, and generate df_raw for one session

    Used in: :py:func:`gen_df_raw_for_n_sessions`

    Args:
        mouse_id (int): mouse identifier
        session (int): session number
        xy_removed_list (list): list of xy coordinates to be removed
        sigma_xy (float): standard deviation for Gaussian smoothing
        M (np.ndarray): perspective transform matrix

    Returns:
        dff (pd.DataFrame): dataframe of raw trajectory for one session
    """
    # load h5py file
    subdir = h5_dict[mouse_id][session]
    load_dir = str(bigdata_dir / subdir)
    keys, data_dict, tracksize = read_h5py(load_dir)

    # load realtime
    subdir = csv_timestamp_dict[mouse_id][session]
    load_dir = str(bigdata_dir / subdir)
    realtime_all = get_realtime_all(load_dir, tracksize)

    # initialize df_raw
    dff = gen_dff_raw(data_dict, realtime_all)

    # filter out 4 corners
    xy_traj = dff[["xb", "yb"]].values
    if len(xy_removed_list) > 0:
        d_removed_traj_list = [
            ((xy_traj - np.array(xy_removed)[None]) ** 2).sum(1) ** 0.5
            for xy_removed in xy_removed_list
        ]
        is_kept_traj_list = [
            d_removed_traj > d_tile for d_removed_traj in d_removed_traj_list
        ]
        is_kept_traj = np.array(is_kept_traj_list).prod(0) > 0
        dff = dff[is_kept_traj]

    # interpolate dff
    dff = interpolate_df_raw(dff)

    # attr: gaussian smooth xb, yb
    xyb_all = dff[["xb", "yb"]].values
    xb, yb = xyb_all.T
    xg, yg = gaussian_filter1d(xb, sigma=sigma_xy), gaussian_filter1d(
        yb, sigma=sigma_xy
    )
    xyg_all = np.stack([xg, yg], axis=1)

    # align trajectory with perspective transform
    xyb_transform = align_xy_with_perspective_transform(xyb_all, M)
    xyg_transform = align_xy_with_perspective_transform(xyg_all, M)

    # attr
    dff = dff[["realtime"]].copy()
    dff["session"] = session
    dff["xb"] = xyb_transform[:, 0]
    dff["yb"] = xyb_transform[:, 1]
    dff["xg"] = xyg_transform[:, 0]
    dff["yg"] = xyg_transform[:, 1]
    return dff


def gen_df_raw_for_n_sessions(mouse_id, n_sessions, xy_removed_list, sigma_xy, M_dict):
    """Run :py:func:`gen_df_raw_for_one_session` for n_sessions and concatenate the results.

    .. admonition:: Auxiliary functions:

        .. line-block::
            ╙── :py:func:`gen_df_raw_for_n_sessions`
                └─╼ :py:func:`gen_df_raw_for_one_session`
                    ├─╼ :py:func:`read_h5py`
                    ├─╼ :py:func:`get_realtime_all`
                    ├─╼ :py:func:`gen_dff_raw`
                    ├─╼ :py:func:`interpolate_df_raw`
                    └─╼ :py:func:`align_xy_with_perspective_transform`

    .. autosummary::
        gen_df_raw_for_one_session
        read_h5py
        get_realtime_all
        gen_dff_raw
        interpolate_df_raw
        align_xy_with_perspective_transform

    Args:
        mouse_id (int): mouse identifier
        n_sessions (int): number of sessions
        xy_removed_list (list): list of xy coordinates to be removed
        sigma_xy (float): standard deviation for Gaussian smoothing
        M_dict (dict): dictionary of perspective transform matrices

    Returns:
        df (pd.DataFrame): concatenated dataframe of raw trajectories for all sessions
    """
    dff_list = []
    for session in range(n_sessions):
        print("generating dff_raw for session", session)
        M = M_dict[(mouse_id, session)]
        dff = gen_df_raw_for_one_session(
            mouse_id, session, xy_removed_list, sigma_xy, M
        )
        dff_list.append(dff)
    df_raw = pd.concat(dff_list, axis=0)
    df_raw.index = range(len(df_raw))
    return df_raw


def get_one_tile_id(xy, xy_pixel_all, tile_id_all):
    """Assign tile id to one coordinate based on nearest tile pixel.

    Args:
        xy (np.ndarray): coordinates of a point
        xy_pixel_all (np.ndarray): array of pixel coordinates
        tile_id_all (np.ndarray): array of tile IDs corresponding to pixels

    Returns:
        tile_id (int): ID of the tile closest to the given point
    """
    idx_pixel = ((xy_pixel_all - xy) ** 2).sum(1).argmin()
    tile_id = tile_id_all[idx_pixel]
    return tile_id


## EQUALIZE TRAJECTORY
def upsample_df_raw(df_raw, upsampl_fac):
    """Upsample raw dataframe by linear interpolation.

    Used in: :py:func:`gen_df_equal`

    Args:
        df_raw (pd.DataFrame): raw trajectory dataframe
        upsampl_fac (int): upsampling factor
    Returns:
        df_raw_upsampl (pd.DataFrame): upsampled raw trajectory dataframe
    """
    assert upsampl_fac in [1, 2, 4, 8]

    # prep
    df_raw_upsampl = df_raw.copy()
    df_raw_upsampl["t_raw"] = df_raw.index
    df_raw_upsampl.index = pd.date_range("1/1/2000", periods=len(df_raw), freq="64s")

    # interpolate
    dt = {1: "64s", 2: "32s", 4: "16s", 8: "8s"}[upsampl_fac]
    df_raw_upsampl = (
        df_raw_upsampl.resample(dt).asfreq().interpolate("linear").copy()
    )  ###
    df_raw_upsampl.index = range(len(df_raw_upsampl))

    # add vg
    xyg_traj = df_raw_upsampl[["xg", "yg"]].values
    vg_traj = (np.diff(xyg_traj, axis=0) ** 2).sum(1) ** 0.5
    vg_traj = np.hstack([vg_traj, [0]])
    df_raw_upsampl["vg"] = vg_traj
    return df_raw_upsampl


def get_geod_equal_downsampl(df_raw_upsampl, d_downsampl):
    """Get geodesic equal downsampling attributes.

    Used in: :py:func:`gen_df_equal`

    Args:
        df_raw_upsampl (pd.DataFrame): upsampled raw trajectory dataframe
        d_downsampl (float): target distance for downsampling

    Returns:
        t_sum (int): total number of time points
        geod_diff_all (np.ndarray): difference between cumulative geodesic distance and equalized geodesic distance
        geod_cumsum_all (np.ndarray): cumulative geodesic distance
        geod_equal_downsampl (np.ndarray): equalized geodesic distance for downsampling
    """
    vg_traj = df_raw_upsampl["vg"].values
    geod_cumsum_all = vg_traj.cumsum() / vg_traj.sum()
    t_sum = len(geod_cumsum_all)
    #
    geod_equal = np.linspace(0, 1, t_sum)
    geod_diff_all = geod_cumsum_all - geod_equal
    geod_equal_downsampl = np.linspace(0, 1, int(vg_traj.sum() / d_downsampl))
    return t_sum, geod_diff_all, geod_cumsum_all, geod_equal_downsampl


def find_one_t_downsampl(prg, t_sum, geod_diff_all, geod_cumsum_all):
    """Find one downsampled time point closest to the target geodesic progression.

    Used in: :py:func:`gen_df_equal`

    Algorithm:
        Step 1: find closest time t_min to the linear progression geod_equal, or minimize geod_diff
        Step 2: progress of the geodesic distance is from 0 to 1

    Args:
        prg (float): target geodesic progression
        t_sum (int): total number of time points
        geod_diff_all (np.ndarray): difference between cumulative geodesic distance and equalized geodesic distance
        geod_cumsum_all (np.ndarray): cumulative geodesic distance

    Returns:
        t_min (int): downsampled time point closest to the target geodesic progression
    """
    # param
    # search_range = np.abs(geod_diff_all).max()

    # get target time (guessed t_min)
    geod_diff = geod_diff_all[int((prg - 1e-16) * t_sum)]
    t_tar = int((prg - geod_diff) * t_sum)

    # limit the search range between x% of the total time
    search_range = 0.05 + np.abs(geod_diff) * 1.5
    t_low, t_up = max(0, t_tar - int(t_sum * search_range)), min(
        t_tar + int(t_sum * search_range), t_sum - 1
    )
    y_filtered = geod_cumsum_all[t_low:t_up]
    t_min = t_low + ((prg - y_filtered) ** 2).argmin()
    return t_min


def get_attr_df_raw_upsampl(
    df_raw_upsampl, geod_cumsum_all, t_downsampl_all, geod_equal_downsampl, d_downsampl
):
    """Get attributes for upsampled raw dataframe that will be inherited by df_equal.

    Used in: :py:func:`gen_df_equal`

    Args:
        df_raw_upsampl (pd.DataFrame): upsampled raw trajectory dataframe
        geod_cumsum_all (np.ndarray): cumulative geodesic distance
        t_downsampl_all (list): list of downsampled time points
        geod_equal_downsampl (np.ndarray): equalized geodesic distance for downsampling
        d_downsampl (float): target distance for downsampling

    Returns:
        is_equal_all (np.ndarray): array indicating equalized points
        t_equal_all (np.ndarray): cumulative count of equalized points
        xye_traj (np.ndarray): corrected trajectory with equal spacing
    """
    # load
    xyg_traj = df_raw_upsampl[["xg", "yg"]].values
    vg_traj = df_raw_upsampl["vg"].values

    # t_equal leaving a trace to connect timestamp of df_equal and df_raw
    is_equal_all = np.zeros(len(df_raw_upsampl))
    for x in t_downsampl_all:
        is_equal_all[x] += 1
    t_equal_all = np.cumsum(is_equal_all).astype("int")

    # correct xy_downsampl with linear interp for equal spacing
    err = geod_cumsum_all[np.array(t_downsampl_all)] - geod_equal_downsampl
    # err in progress (0 to 1)
    err_step = err * round(vg_traj.sum() / d_downsampl)  # err * n_steps
    xyg_equal = df_raw_upsampl.loc[t_downsampl_all][["xg", "yg"]].values
    xyg_equal_correct = (
        xyg_equal[:-1] + np.diff(xyg_equal, axis=0) * err_step[:-1, None] * -1.0
    )

    # df
    xye_traj = xyg_traj.copy()
    xye_traj[t_downsampl_all[:-1]] = xyg_equal_correct
    return is_equal_all, t_equal_all, xye_traj


def gen_df_equal(df_raw, d_downsampl, upsampl_fac=2):
    """Take raw trajectory and equalized trajectory.

    .. admonition:: Auxiliary functions:

        .. line-block::
            ╙── :py:func:`gen_df_equal`
                ├─╼ :py:func:`upsample_df_raw`
                ├─╼ :py:func:`get_geod_equal_downsampl`
                ├─╼ :py:func:`get_attr_df_raw_upsampl`
                └─╼ :py:func:`find_one_t_downsampl`

    .. autosummary::
        upsample_df_raw
        get_geod_equal_downsampl
        get_attr_df_raw_upsampl
        find_one_t_downsampl

    Algorithm:
        Step 1: upsampling
        Step 2: removing intermediate points so that inter-point distances are roughly ``d_tile/4``
        Step 3: correcting the remaining points with linear interpolation to ensure equal spacing

    Args:
        df_raw (pd.DataFrame): raw trajectory dataframe
        d_downsampl (float): target distance for downsampling, e.g., ``d_tile/4``
        upsampl_fac (int, optional): upsampling factor. Defaults to 2.

    Returns:
        df_equal (pd.DataFrame): equalized trajectory dataframe
    """
    # upsample the raw data x2 (x8, 45s)
    df_raw_upsampl = upsample_df_raw(df_raw, upsampl_fac)

    # downsample by removing intermediate points so that inter-point distances are roughly d_tile/4 (45s)
    t_sum, geod_diff_all, geod_cumsum_all, geod_equal_downsampl = (
        get_geod_equal_downsampl(df_raw_upsampl, d_downsampl)
    )

    # mp (3m45s)
    def run_mp(prg):
        return find_one_t_downsampl(prg, t_sum, geod_diff_all, geod_cumsum_all)

    with multiprocess.Pool(num_cpus) as p:
        t_downsampl_all = p.map(run_mp, geod_equal_downsampl)

    # get attributes to df_raw_upsampl that will be inherited by df_equal
    is_equal_all, t_equal_all, xye_traj = get_attr_df_raw_upsampl(
        df_raw_upsampl,
        geod_cumsum_all,
        t_downsampl_all,
        geod_equal_downsampl,
        d_downsampl,
    )

    # attr to df_raw_upsampl
    df_raw_upsampl["is_equal"] = is_equal_all
    df_raw_upsampl["t_equal"] = t_equal_all
    df_raw_upsampl["geod_cumsum"] = geod_cumsum_all
    df_raw_upsampl["xe"] = xye_traj[:, 0]
    df_raw_upsampl["ye"] = xye_traj[:, 1]

    # generate df_equal (1m)
    df_equal = df_raw_upsampl.loc[
        t_downsampl_all
    ].copy()  # remove unwanted upsampled points
    # df_equal["tile"] = df_equal["tile"].astype("int")
    df_equal["session"] = df_equal["session"].astype("int")
    df_equal["t_raw"] = df_equal["t_raw"].astype("int")
    df_equal.index = range(len(df_equal))

    # remove zero velocity from upsampling
    vg_traj = (np.diff(df_equal[["xg", "yg"]].values, axis=0) ** 2).sum(-1) ** 0.5
    is_zero = np.zeros(len(df_equal))
    is_zero[np.where(vg_traj == 0)[0] + 1] = 1
    df_equal = df_equal.loc[is_zero == 0]

    # final update index of df_equal
    df_equal.index = range(len(df_equal))
    return df_equal


## LABEL TRAJECTORY
def get_all_diameter(
    t0_base_all,
    df_equal,
    l_th=2 * fs_equal,
    d_th=d_tile,
    n_samples_per_path=20 * fs_equal,
):
    """Get diameter of a short segment at each timepoint (``t_equal``).

    Algorithm:
        Step 1: for each timepoint ``t``, sweep forwards a short segment e.g., ``l_th=2*4``, or 2 tile width by default.
        Step 2: for each segment, find the maximum distance between any two points in the segment, i.e., diameter.
        Step 3: if the diameter is smaller than ``d_th``, label ``is_small=True`` for that timepoint.

    Args:
        t0_base_all (np.ndarray): array of timepoints in ``t_equal``
        df_equal (pd.DataFrame): equalized trajectory dataframe
        l_th (int, optional): length threshold for segment. Defaults to 2*fs_equal.
        d_th (float, optional): diameter threshold. Defaults to d_tile.
        n_samples_per_path (int, optional): number of samples per path. Defaults to 20*fs_equal.

    Returns:
        d_max_all (np.ndarray): array of diameters for each timepoint
        bool_small_trim_all (np.ndarray): boolean array indicating small-diameter segments
    """
    # trim to fit df_equal
    t0_base_all_1 = t0_base_all[:-l_th]
    t0_base_all_2 = t0_base_all[-l_th:]

    # get ttttt array
    progression = np.linspace(0, 1, n_samples_per_path)
    ttttt_all = [(t0_base_all_1 + l_th * x).astype(int) for x in progression]

    # get xy_arr
    xy_arr = np.stack([df_equal.loc[x, ["xe", "ye"]].values for x in ttttt_all], axis=1)

    # get diameter
    d_arr = ((xy_arr[:, :, None] - xy_arr[:, None]) ** 2).sum(-1) ** 0.5
    d_max_all_1 = d_arr.max(-1).max(-1)
    bool_small_all_1 = d_max_all_1 < d_th

    # compute t0_base_all_2 separately
    d_max_all_2, bool_small_all_2 = [], []
    for t0 in t0_base_all_2:
        xy_ = df_equal.loc[t0:, ["xe", "ye"]].values
        d_ = ((xy_[:, None] - xy_[None, :]) ** 2).sum(-1) ** 0.5
        d_max = d_.max()
        d_max_all_2.append(d_max)
        bool_small_all_2.append(d_max < d_th)

    # concat
    d_max_all = np.concatenate([d_max_all_1, d_max_all_2])
    bool_small_all = np.concatenate([bool_small_all_1, bool_small_all_2])

    # trim head and tail of each segment
    tt_small_all = get_ii_segm_from_bool_list(bool_small_all)
    tt_small_trim = tt_small_all.copy()
    tt_small_trim[:, 0] += fs_equal
    tt_small_trim[:, 1] -= fs_equal
    tt_small_trim[0, 0] = tt_small_all[0, 0]
    tt_small_trim[-1, 1] = tt_small_all[-1, 1]

    bool_small_trim_all = np.zeros_like(bool_small_all, dtype=bool)
    for x, y in tt_small_trim:
        if y <= x:
            continue
        bool_small_trim_all[x:y] = True

    return d_max_all, bool_small_trim_all


def get_xy_wall(x0=1148, y0=1030, r=990, n_points=200):
    """This function draws a circle near the wall boundary.

    Args:
        x0 (float): x-coordinate of the center of the wall boundary circle.
        y0 (float): y-coordinate of the center of the wall boundary circle.
        r (float, optional): radius of the wall. Defaults to 990 for the standard maze (mouse 10 session 6)
        n_points (int, optional): number of points to generate along the circle. Defaults to 200.

    Returns:
        xy_wall: np.ndarray of shape (200, 2) representing the x and y coordinates of the wall boundary
    """
    theta_ = np.linspace(0, 2 * np.pi, n_points)
    x_ = np.cos(theta_) * r + x0
    y_ = np.sin(theta_) * r + y0
    xy_wall = np.stack([x_, y_], axis=1)
    return xy_wall


def label_raw_traj_from_equalized_traj(bool_label_all, label, df_equal, df_raw):
    """Given an attribute in df_equal, map it to df_raw based on ``df_equal.t_raw``

    Args:
        bool_label_all (np.ndarray): boolean array indicating labeled segments in df_equal
        label (str): name of the new attribute to add to df_raw
        df_equal (pd.DataFrame): equalized trajectory dataframe
        df_raw (pd.DataFrame): raw trajectory dataframe

    Returns:
        df_raw (pd.DataFrame): df_raw with a newly added attribute
    """
    # get intervals in t_equal
    tt_label_all = get_ii_segm_from_bool_list(bool_label_all)
    tt_label_all[tt_label_all == len(df_equal)] = (
        len(df_equal) - 1
    )  # replace last index

    # get corresponding intervals in t_raw
    tt_raw_label_all = np.stack(
        [df_equal.loc[[x, y]].t_raw.values for x, y in tt_label_all]
    )

    # attr to df_raw
    bool_label_all_raw = np.zeros(len(df_raw), dtype=bool)
    for x, y in tt_raw_label_all:
        bool_label_all_raw[x:y] = True
    df_raw[label] = bool_label_all_raw
    return df_raw


def get_all_jumps_as_tile_123(df_equal):
    """Get (tile_last, tile_current, tile_next) for each timepoint in df_equal.

    Algorithm:
        Step 1: get tile_id sequence without consecutive duplicates
        Step 2: for each tile_id, get (tile_last, tile_current, tile_next)

    Args:
        df_equal (pd.DataFrame): equalized trajectory dataframe

    Returns:
        tile_123_all (list of tuples): each tuple contains (tile_last, tile_current, tile_next) for each timepoint
    """
    tile_traj = df_equal.tile_id.values.astype(int)
    tile_traj_r = [x for x, y in groupby(tile_traj)]
    tile_last_all = [tile_traj_r[0]] + tile_traj_r[:-1]
    tile_next_all = tile_traj_r[1:] + [tile_traj_r[-1]]
    tile_123_all = fl(
        [
            len(list(y)) * [(z, x, w)]
            for (x, y), z, w in zip(groupby(tile_traj), tile_last_all, tile_next_all)
        ]
    )
    return tile_123_all
