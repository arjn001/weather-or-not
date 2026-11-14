"""
Estimating the car's motion from the front camera alone.

Two separate estimates come out of each pair of consecutive frames, and they
are made in two different ways.

TURN RATE (yaw)
  Follow a few hundred corner points from one frame to the next with
  Lucas-Kanade optical flow, then ask what camera rotation explains how they
  all moved. Rotation shifts every point the same way no matter how far
  away it is, so this needs no knowledge of distance and comes out in real
  units.

FORWARD SPEED
  Speed does need distance, and one camera cannot measure distance. We get
  around that with a single assumption: the road is flat and the camera
  sits at a known height above it. Under that assumption we can predict
  exactly how the road surface should look in the second frame if the car
  moved forward by, say, 0.5 m. So we try many candidate distances and keep
  the one whose prediction lines up best with the real second frame.

  When the assumption is wrong (a hill, a vehicle filling the road ahead) or
  the road shows nothing to line up (dark, wet, blurred), the speed estimate
  degrades. Yaw does not share this weakness.
"""

import cv2
import numpy as np

from camera_health import frame_appearance

# --- Point tracking, used for rotation -------------------------------------
MAX_CORNERS = 1500
CORNER_QUALITY = 0.01
CORNER_MIN_DISTANCE = 8

# The pyramid lets Lucas-Kanade follow points that move a long way between
# frames, which happens at speed.
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# A track is thrown away if following it forward and then backward does not
# land within this many pixels of where it started.
MAX_ROUND_TRIP_ERROR = 1.0

# Below this many tracks we do not trust a frame pair and report NaN.
MIN_TRACKS = 30

# --- Road matching, used for speed -----------------------------------------
# Work at half resolution. It is four times less work and loses nothing that
# matters here.
ROAD_SCALE = 0.5
# Before comparing frames we subtract a heavily blurred copy of each image.
# That removes overall brightness (shadows, exposure changes) and keeps the
# fine texture, which is what actually tells us the road has moved.
DETAIL_BLUR_SIGMA = 8
# The road region is a trapezoid below the horizon that widens toward the
# bottom of the image. Numbers are in full-resolution pixels.
ROAD_ROWS_BELOW_HORIZON = 90
ROAD_HALF_WIDTH_AT_TOP = 100
ROAD_WIDENING_PER_ROW = 1.2
# Candidate forward distances per frame pair, in metres. First a coarse scan,
# then a fine scan around the best coarse answer. 3 m in one 0.1 s frame gap
# is 30 m/s (about 67 mph), well above anything in city driving. The scan
# starts slightly below zero so a stopped car is not pinned at the edge.
COARSE_DISTANCES = np.arange(-0.3, 3.01, 0.1)
FINE_HALF_RANGE = 0.1
FINE_STEP = 0.02
# Need at least this many usable road pixels to score a candidate.
MIN_ROAD_PIXELS = 1000


def track_points(gray_a, gray_b):
    """
    Follow corner points from one grayscale frame into the next.

    Returns two (N, 2) arrays: where each point was, and where it went.
    Only points that pass the round-trip check are returned.
    """
    corners = cv2.goodFeaturesToTrack(gray_a, MAX_CORNERS, CORNER_QUALITY, CORNER_MIN_DISTANCE)
    if corners is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    moved, ok_fwd, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, corners, None, **LK_PARAMS)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, moved, None, **LK_PARAMS)

    # Round-trip check: a point that was tracked correctly should come back
    # to where it started when tracked in reverse.
    round_trip = np.linalg.norm((corners - back).reshape(-1, 2), axis=1)
    keep = (ok_fwd.ravel() == 1) & (ok_back.ravel() == 1) & (round_trip < MAX_ROUND_TRIP_ERROR)
    return corners.reshape(-1, 2)[keep], moved.reshape(-1, 2)[keep]


def rotation_between(pts_a, pts_b, K):
    """
    Work out how the camera rotated between two frames.

    Returns (R, n_inliers). R is a 3x3 rotation matrix, or None if the
    geometry could not be solved.
    """
    if len(pts_a) < MIN_TRACKS:
        return None, 0

    # The essential matrix describes how two views of a rigid scene relate.
    # RANSAC fits it to the majority of tracks and flags the rest (other
    # cars, pedestrians, bad tracks) as outliers.
    E, inlier_mask = cv2.findEssentialMat(
        pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None or E.shape != (3, 3):
        return None, 0

    # An essential matrix is consistent with two different rotations. One is
    # the real one, the other is that plus a half turn. A car cannot spin
    # half way round in a tenth of a second, so we take the smaller rotation.
    # (A bigger trace means a rotation closer to "no rotation at all".)
    R1, R2, _ = cv2.decomposeEssentialMat(E)
    R = R1 if np.trace(R1) > np.trace(R2) else R2
    return R, int(inlier_mask.sum())


def yaw_from_rotation(R):
    """
    Pull the yaw angle (radians) out of a rotation matrix.

    The camera's y axis points down at the road, so rotation about it is
    the car turning. With OpenCV's conventions a left turn comes out
    positive, the same sign the CAN bus uses. Checked against CAN on
    scene-0916, which has hard turns in both directions.
    """
    rotation_vector, _ = cv2.Rodrigues(R)
    return float(rotation_vector[1, 0])


class RoadMatcher:
    """
    Measures forward distance by lining up the road between two frames.

    Built once per scene, because the camera geometry and the road region
    do not change from frame to frame.
    """

    def __init__(self, image_shape, camera):
        self.camera = camera
        self.size = (int(image_shape[1] * ROAD_SCALE), int(image_shape[0] * ROAD_SCALE))

        # Intrinsics for the half-size image.
        self.K = camera["K"].copy()
        self.K[:2] *= ROAD_SCALE
        self.K_inv = np.linalg.inv(self.K)

        # The trapezoid of pixels we treat as road.
        rows, cols = np.mgrid[0 : image_shape[0], 0 : image_shape[1]]
        below = rows - camera["ahead_y"]
        sideways = np.abs(cols - camera["ahead_x"])
        full = (below > ROAD_ROWS_BELOW_HORIZON) & (
            sideways < ROAD_HALF_WIDTH_AT_TOP + ROAD_WIDENING_PER_ROW * below
        )
        small = cv2.resize(full.astype(np.uint8), self.size, interpolation=cv2.INTER_NEAREST)
        self.road = small > 0

    def detail(self, gray):
        """Shrink a frame and keep only its fine texture."""
        small = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        return small - cv2.GaussianBlur(small, (0, 0), DETAIL_BLUR_SIGMA)

    def mismatch(self, detail_a, detail_b, R, distance):
        """
        How badly the road fails to line up if the car moved `distance` m.

        For a flat surface there is an exact formula (a homography) for
        where each road pixel of frame A shows up in frame B, given the
        camera's rotation and how far it moved. We use it to pull frame B
        back onto frame A and average the difference over the road region.
        Lower is better.
        """
        cam = self.camera
        shift = distance * np.outer(cam["forward"], cam["up"]) / cam["height"]
        H = self.K @ R @ (np.eye(3) + shift) @ self.K_inv
        pulled_back = cv2.warpPerspective(
            detail_b, H, self.size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=np.nan,
        )
        difference = np.abs(pulled_back - detail_a)[self.road]
        difference = difference[np.isfinite(difference)]
        if difference.size < MIN_ROAD_PIXELS:
            return np.inf
        return float(difference.mean())

    def forward_distance(self, detail_a, detail_b, R):
        """
        Find the forward distance (metres) that lines the road up best.

        Returns (distance, match_quality). match_quality compares the best
        candidate against a typical one: near 1 means one distance stood
        out clearly, near 0 means every candidate looked about equally bad
        and the answer should not be trusted.
        """
        coarse = np.array([self.mismatch(detail_a, detail_b, R, d) for d in COARSE_DISTANCES])
        if not np.isfinite(coarse).any():
            return np.nan, 0.0
        best = COARSE_DISTANCES[np.argmin(coarse)]

        fine_distances = np.arange(best - FINE_HALF_RANGE, best + FINE_HALF_RANGE + 1e-9, FINE_STEP)
        fine = np.array([self.mismatch(detail_a, detail_b, R, d) for d in fine_distances])
        winner = int(np.argmin(fine))

        typical = np.median(coarse[np.isfinite(coarse)])
        quality = 1.0 - fine[winner] / typical if typical > 0 else 0.0
        return float(fine_distances[winner]), float(quality)


def estimate_scene_motion(utimes, paths, camera):
    """
    Run both estimators over every consecutive frame pair of a scene.

    Returns a dict of arrays, one entry per frame pair:
      utime       : timestamp of the midpoint between the two frames
      dt          : seconds between the two frames
      yaw_rate    : rad/s, left positive
      speed       : m/s, valid only as far as the flat road assumption is
      n_tracks    : corner points followed across the image
      n_inliers   : of those, how many agreed with the rigid-scene fit
      road_match  : 0 to 1, how clearly one speed stood out (see RoadMatcher)
      brightness, contrast, sharpness : appearance of the second frame of
                    the pair (see camera_health.py)
    """
    keys = ("utime", "dt", "yaw_rate", "speed", "n_tracks", "n_inliers", "road_match",
            "brightness", "contrast", "sharpness")
    out = {k: [] for k in keys}

    prev_gray = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    matcher = RoadMatcher(prev_gray.shape, camera)
    prev_detail = matcher.detail(prev_gray)

    for i in range(1, len(paths)):
        gray = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
        detail = matcher.detail(gray)
        dt = (utimes[i] - utimes[i - 1]) / 1e6

        pts_a, pts_b = track_points(prev_gray, gray)
        R, n_inliers = rotation_between(pts_a, pts_b, camera["K"])

        if R is None:
            yaw_rate, speed, quality = np.nan, np.nan, 0.0
        else:
            yaw_rate = yaw_from_rotation(R) / dt
            distance, quality = matcher.forward_distance(prev_detail, detail, R)
            speed = distance / dt

        out["utime"].append((utimes[i] + utimes[i - 1]) // 2)
        out["dt"].append(dt)
        out["yaw_rate"].append(yaw_rate)
        out["speed"].append(speed)
        out["n_tracks"].append(len(pts_a))
        out["n_inliers"].append(n_inliers)
        out["road_match"].append(quality)
        for name, value in frame_appearance(gray).items():
            out[name].append(value)
        prev_gray, prev_detail = gray, detail

    return {k: np.array(v) for k, v in out.items()}
