from pathlib import Path
import numpy as np


## SET DIRECTORIES
project_dir = Path(__file__).parent.parent
bigdata_dir = project_dir / "bigdata"

## BASH PARAMETERS
mouse_ids = [3,4,5,16,18,19,20]

## GLOBAL PARAMETERS
# UTILS
q1, q2, q3 = 0.25, 0.50, 0.75  # quartiles

# MINIMAP
maze = np.array(
    [
        [-1] * 17,
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 6, -1, -1],
        [-1, -1, -1, -1, -1, -1, 8, 1, 1, 3, 1, 1, 1, 1, 3, 8, -1],
        [-1, -1, -1, -1, -1, 8, 1, 1, 1, 8, 9, 1, 1, 1, 8, 8, -1],
        [-1, -1, -1, -1, 8, 5, 1, 1, 2, 1, 1, 9, 2, 12, 6, 8, -1],
        [-1, -1, -1, 7, 5, 3, 6, 10, 1, 2, 2, 3, 1, 5, 5, 8, -1],
        [-1, -1, -1, 8, 5, 2, 1, 1, 1, 5, 8, 1, 10, 6, 7, -1, -1],
        [-1, -1, 8, 4, 8, 3, 3, 2, 8, 3, 3, 5, 4, 7, 8, -1, -1],
        [-1, -1, 5, 3, 2, 1, 1, 1, 1, 2, 3, 2, 6, 6, -1, -1, -1],
        [-1, 8, 3, 3, 3, 1, 2, 9, 1, 2, 2, 9, 6, 8, -1, -1, -1],
        [-1, 5, 8, 3, 3, 2, 1, 1, 2, 9, 4, 8, 8, -1, -1, -1, -1],
        [-1, 6, 5, 2, 8, 8, 2, 1, 3, 3, 4, 8, -1, -1, -1, -1, -1],
        [-1, 8, 3, 1, 1, 1, 1, 2, 3, 2, 8, -1, -1, -1, -1, -1, -1],
        [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1] * 17,
    ]
)[::-1, :]
n_tiles = (maze != -1).sum()
ringsize = 6

# EQUAL
use_tile_data = True
d_tile = 170  # width of center tile of mouse 10 session 6 in pixels
# 3 for 25Hz, 5for 40Hz
sigma_smooth_dict = {
    3: 3,
    10: 5,
}  # sigma for gaussian smoothing xb, yb (keypoint of mouse bum)
fs_equal = 4  # sample rate (points per tile) of equalized trajectory xe, ye
fs_pair_th = 2  # criterion of a pair of matching paths: d_p2p < 1/2*d_tile
vg_jump_th = 10  # above which a point is considered as a candidate jump
fwd_jump_dict = {
    (27, 39, 52): (27, 39, 52),
    #
    (52, 64, 77): (52, 64, 77),
    (52, 65, 77): (52, 64, 77),
    #
    (77, 78, 91): (77, 78, 91),
    (77, 90, 91): (77, 78, 91),
    #
    (90, 103, 116): (90, 103, 116),
    #
    (91, 103, 116): (91, 103, 116),
    (91, 104, 116): (91, 103, 116),
    #
    (91, 92, 93): (91, 92, 93),
    #
    (93, 105, 118): (93, 105, 118),
    (93, 106, 118): (93, 105, 118),
    #
    (116, 117, 118): (116, 117, 118),
    (116, 117, 129): (116, 117, 118),
    (116, 129, 118): (116, 117, 118),
    (116, 128, 129): (116, 117, 118),
    (117, 129, 118): (116, 117, 118),
    (128, 129, 118): (116, 117, 118),
    (128, 129, 118): (116, 117, 118),
    #
    (105, 104, 116): (105, 104, 116),
    (105, 117, 116): (105, 104, 116),
    #
    (116, 127, 126): (116, 127, 126),
    #
    (100, 112, 124): (100, 113, 125),
    (100, 112, 113): (100, 113, 125),
    (100, 112, 125): (100, 113, 125),
    (100, 113, 112): (100, 113, 125),
    (100, 113, 125): (100, 113, 125),
    (100, 113, 124): (100, 113, 125),
    (100, 124, 125): (100, 113, 125),
    (112, 124, 125): (100, 113, 125),
    #
    (41, 40, 52): (42, 41, 40, 52),
    (54, 40, 52): (42, 41, 40, 52),
    (41, 53, 52): (42, 41, 40, 52),
    (54, 53, 52): (42, 41, 40, 52),
    (42, 41, 40): (42, 41, 40, 52),
    (42, 41, 53): (42, 41, 40, 52),
    (42, 54, 40): (42, 41, 40, 52),
    (42, 54, 53): (42, 41, 40, 52),
    #
    (42, 43, 56): (42, 43, 56),
    (42, 55, 56): (42, 43, 56),
    #
    (73, 86, 99): (73, 86, 99),
    (73, 98, 99): (73, 86, 99),
}

# BUNDLE
l_bundle_max = 200
l_bundle_min = 4

# PAIR
fs_pair_th = 2  # criterion of a pair of matching paths: d_p2p < 1/2*d_tile
dv_pair_th = np.pi / 2  # angle threshold (consider matching if better than orthogonal)
q_v_low, q_v_high = q2, q3  # for ternarize velocity
l_pair_th = 6  # toklen threshold (minimum is a one tile jump ~1.5 tiles)
# l_pair_th = 4

# ROOTPAIR

# TREESEGM
soft_exclusion = q1
soft_inclusion = q3


# LEAFSEGM


## LOAD: align
align_data_dir = project_dir / "data" / "anchors_all_mice"
snapshot_dir = project_dir / "data" / "cam_all_mice"


## LOAD: tile
tile_data_dir = project_dir / "data" / "mouse10_sess6_annotated_tiles"


## LOAD: day
csv_timestamp_dict = {
    3: ["centroid_timestamps_2023-08-02T10_10_43.csv"],
    10: ["centroid_timestamps_2025-09-17T09_57_09.csv"],
}

h5_dict = {
    3: [
        "top_cam_video_coarse2023-08-02T10_10_44.avi.000_top_cam_video_coarse2023-08-02T10_10_44.analysis.h5",
    ],
    10: ["v2_compressed_video_2025-09-17T09_57_10.analysis.h5"],
}

n_sessions_dict = {x: len(y) for x, y in h5_dict.items()}
