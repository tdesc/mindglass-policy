"""Scripted Franka pick(-and-place) policy served over the Positronic wire.

No neural network: a deterministic per-session state machine driven by a wrist-camera
blob detector (HSV segmentation via OpenCV) and the rig's own end-effector pose. Served
with the official ``positronic-model-server`` wrapper (protocol v3), so transport,
serialization, sessions and auth are their tested code, not ours.

Wire contract (verified against /tmp/positronic_repo, model_server 0.3.7 == clone 0.3.8
minus timing-telemetry fields):
  - Session.infer gets the observation dict the client stack selects (flat keys, images
    already decoded to uint8 (H, W, 3) RGB by ``serialization.deserialise``).
  - Session.infer returns a list of per-step dicts ``{'action': float32(8)}`` where the
    action vector is ``[x, y, z, qw, qx, qy, qz, grip]`` (QUAT rep = wxyz scalar-first,
    same convention as ``robot_state.ee_pose``). The declared client stack turns each
    vector into a ``CartesianPosition`` command plus ``target_grip`` via
    ``absolute_position_action`` (positronic/policy/codecs/action.py:161-190).
  - Once the episode decision is made, every element also carries ``'end'``:
    ``'complete'`` or ``'give_up'`` (EndCommand codec, positronic/policy/codecs/end.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY
from positronic_model_server.server import Model, Session
from positronic_model_server.spec import component, parallel, sequence

from task_parser import CONTAINERS, SKILL_MOVE_DIR, SKILL_PLACE, SURFACES, parse_task_text

logger = logging.getLogger(__name__)

# Observation keys (flat strings; dots are literal — docs/connect-your-model.md:114-131).
# Only guaranteed-present keys are selected: ee_wrench/dq can be absent per contract.
EE_POSE = 'robot_state.ee_pose'  # float32(7): x, y, z, qw, qx, qy, qz (wxyz scalar first)
STATUS = 'robot_state.status'  # int scalar: 0 available, 1 busy, 3 error
GRIP = 'grip'  # float scalar in [0, 1]: 0 open, 1 closed
WRIST_IMAGE = 'image.wrist'  # uint8 (H, W, 3) RGB
EXTERIOR_IMAGE = 'image.exterior'  # uint8 (H, W, 3) RGB
EXTERIOR2_IMAGE = 'image.exterior_2'  # uint8 (H, W, 3) RGB
TASK = 'task'  # str
DESCRIPTOR = 'descriptor'  # str, e.g. 'mujoco.franka'

# Session parameter names (same names the GR00T vendor server declares — the platform
# may send them as query params; FIX-1: nothing is refused — unknown names are ignored
# by the tolerant wire resolution in policy_server, unusable values fall back to the
# defaults in PickAndPlaceModel.prepare_session).
PARAM_FPS = 'fps'
PARAM_HORIZON_SEC = 'horizon_sec'
PARAM_COMPRESS_IMAGES = 'compress_images'
PARAM_JPEG_QUALITY = 'jpeg_quality'
PARAM_INFERENCE_LEAD = 'inference_lead'
PARAM_RTC = 'rtc'

# The model's native action field: AbsolutePositionAction._decode_single reads
# data['action'] (metadata.ACTION) — hardcoded, not a wire argument.
ACTION_FIELD = 'action'

# Frame conventions
XYZW = 4  # quat occupies action[3:7], wxyz scalar first
ACTION_WIDTH = XYZW + 3 + 1  # translation(3) + quat(4) + grip(1)

DEFAULT_FPS = 10.0
DEFAULT_HORIZON_SEC = 1.0

# ---------------------------------------------------------------------------
# State-machine constants. The wrist camera is uncalibrated on purpose: the policy is
# closed-loop over the blob centroid, so an unknown pixels-to-metres gain converges;
# a wrong axis SIGN diverges, so the servo auto-flips a sign when its error grows.
# ---------------------------------------------------------------------------

SERVO_GAIN = 0.25  # metres of EE motion per unit of normalized image offset ([-1, 1])
MAX_XY_STEP = 0.10  # metres per chunk (~1 s) on the horizontal plane
MAX_Z_STEP = 0.07  # metres per chunk vertically
CENTER_TOL = 0.08  # normalized |offset| that counts as "object centered"
CONVERGE_CHUNKS = 2  # consecutive centered chunks before descending
SEARCH_BUDGET = 18  # chunks without any blob detection before give_up
SPIRAL_STEP = 0.02  # metres of radius growth per search chunk
SPIRAL_MAX_RADIUS = 0.16
AREA_GRASP_FRAC = 0.45  # blob area / image area that means "camera is right on the object"
FLOOR_DROP = 0.32  # metres below the first observed cruise height: blind descend limit
DESCEND_DZ = 0.045  # metres per chunk while descending
GRASP_HOLD_CHUNKS = 2  # chunks with the gripper commanded closed before verifying
GRASP_RETRIES = 2  # verified-failed grasp attempts before give_up
LIFT_HEIGHT = 0.14  # metres above the grasp height
PLACE_DEPTH = 0.05  # metres to lower the object before releasing
CARRY_OFFSET = (0.25, 0.0)  # metres from the grasp point; axis meaning unknown without extrinsics
RELEASE_HOLD_CHUNKS = 2
RETRACT_HEIGHT = 0.08
HOLD_AFTER_TASK_BUDGET = 30  # chunks to keep serving a finished task before ending 'complete'
ERROR_BUDGET = 6  # consecutive error-status chunks before give_up
EPISODE_BUDGET = 400  # chunks total; the drawer pipeline (sweep + probe walk + grasp) needs
                      # ~240-300 — 128 starved it mid-grasp (submission #4 log: sweep 110 + probe 7 +
                      # grasp 2, then the safety net fired). Picks finish in ~25 chunks, harmless.
                      # the open-mode blind attempt + vision search needs the headroom)

# --- closed-loop gain-sign calibration (replaces the flip heuristic, FIX-2) ----------
# The wrist camera is uncalibrated: the pixels-to-metres gain and the axis signs are
# unknown. The old rule flipped a sign whenever the image error grew after a move; the
# submission-3 log shows it oscillated (803 flips, all 367 second flips within 120 s of
# the first, median 3 s, no convergence over 84 min). Now each axis gets ONE deliberate
# probe: command a fixed step along that world axis, then divide the measured image-
# offset change by the measured EE motion (robot_state.ee_pose arrives every chunk):
# g = Δoffset/Δee, sign = -sign(g). Deterministic in one chunk per axis, no series.
PROBE_STEP = 0.05          # metres of the deliberate probe move along one axis
PROBE_MIN_MOVE = 0.008     # metres: less actual motion along the axis = inconclusive probe
MOTION_VERIFY_STEP = 0.015  # metres of actual EE motion after which a move is judgeable (the
                           # local gym tracks ~0.018 m per chunk — 0.02 never judged there)
CAMERA_FIXED_TOL = 0.015   # image offset unmoved while the EE moved >= verify step: feature is
                           # rigid with the camera (the gripper itself, its shadow, scenery)
EXCLUDE_RADIUS = 0.30      # normalized image radius around a rejected feature
MAX_EXCLUDE_ZONES = 4      # bounded blacklist of camera-fixed regions
PROBE_RETRIES = 2          # inconclusive probe attempts before proceeding with the current sign
PROBE_JUMP_TOL = 0.5       # image offset jump that means the tracked blob changed: reset the
                           # reference instead of accumulating across different features
GAIN_CLIP = (1.5, 60.0)    # measured |doffset/dEE| bounds for the closed-loop servo gain
GAIN_SAFETY = 0.8          # per-chunk error decay factor: delta = min(SERVO_GAIN, 0.8/|gain|)
GAIN_MIN_OBJECT = 2.5      # a tracked blob whose measured |gain| is below this moves with the
                           # EE (the arm's shadow) or sits so far that grasping is pointless —
                           # reject it instead of chasing
DRAWER_CAMERA_FIXED_TOL = 0.03  # the drawer search tolerates more image drift before calling
                                # a blob camera-fixed (the arm shadow shifts non-rigidly)

# --- drawer open-search (FIX-3, open mode only; close mode untouched) ----------------
# The blind nominal handle point misses when the cabinet is off to the side (the local
# gym drawer sits 0.30 m lateral; the nominal point is unreachable short of the reach
# limit and false-grasps the place box). Open mode then searches by CONTACT: re-home,
# transit at the cruise height, then drag the closed gripper along a lateral line 6 cm
# below it — the pad zone spans TCP..TCP+0.034 and only the raised cabinet body reaches
# into it (measured on the gym: the bin top at 0.425 and its walls at 0.42 vs the table
# 0.31, the place box 0.385 and the cubes 0.342), so the actual TCP riding above the
# commanded drag height or a grip stopping while dragging is a solid contact signal that
# camera artifacts cannot fake. The handle then sits TOWARD THE ROBOT of the contact
# point within about one cabinet depth (a drawer front faces the robot), so a short
# blind PROBE GRID of grasp attempts around the contact point catches it; the pull
# starts toward the robot and the drawer_act stall reverses it once if the slide axis
# sign was guessed wrong. The wrist camera is not used for the search: the live runs
# showed its dark blobs are dominated by the gripper's own crevices and the arm's
# shadow, which no area/edge/gain filter separated reliably.
SWEEP_CONTACT_DROP = 0.06  # metres below the cruise height for the contact sweep
SWEEP_BACK = 0.06          # metres behind the start pose: the sweep line's x offset (a deeper
                           # line folds the arm hard enough that the rotated gripper collides
                           # with the forearm and the drag stalls in a local minimum)
SWEEP_Y_START = 0.08       # metres: the drag lines start this far off the y=0 axis, where the
                           # folded-and-rotated gripper self-presses the forearm (grip reads
                           # 0.04-0.16 in free space and would fake a contact signal)
SWEEP_STEP = 0.06          # metres between sweep waypoints
SWEEP_HALF_SPAN = 0.42     # sweep this far to each side before giving up
SWEEP_RIDE_Z = 0.005       # metres of actual TCP above the commanded drag height = riding the
                           # cabinet body = contact
GRIP_AIR_MARGIN_STOP = 0.15  # grip below (measured air closure - this) while dragging = contact
GRIP_AIR_MARGIN_HELD = 0.08  # grip below (air - this) at the hold check = fingers stopped on object
GRIP_AIR_MARGIN_LOST = 0.05  # grip within (air - this) of full closure = slipped/empty
PROBE_DX_OPEN = (-0.15, -0.12, -0.20, -0.25)  # open mode: the drawer starts shut; the
                           # handle sits at the front face, ~0.15 toward the robot of the
                           # contacted body top (polygon probe round 1: the grid stopped
                           # 0.04-0.06 m short of the handle truth (0.365, -0.30) — extended)
PROBE_DX_CLOSE = (-0.25, -0.20, -0.15)  # close mode: the drawer starts pulled OUT, so the
                           # handle slides this much further toward the robot (the slide
                           # travel), and the probe grid must span the whole travel
PROBE_DY = (0.08, 0.11, 0.05, 0.16, 0.21)  # …and BEYOND the contact along the drag direction: the first
                           # contact is the cabinet's side wall, and the handle sits just
                           # past that wall (the bin half-width), +-0.03 of uncertainty
DRAWER_PROBE_Z_DROP = 0.02  # metres BELOW the nominal handle height the probe grasps at:
                           # the pad zone spans TCP-0.005..TCP+0.034 and the gym handle rod
                           # centre sits 0.02 above the nominal point's pad zone bottom
FINGERTIP_REACH = 0.05     # metres: the fingers reach past the pads, so the drag contact fires
                           # this far BEFORE the camera-side cabinet edge; the probe grid is
                           # built from the corrected contact point

# --- colorless object candidates (FIX-2) ----------------------------------------------
# 90.3% of the submission-3 tasks name no colour ('pick up the bottle/tissue/fork/…'),
# and an HSV-only scan of the four vision colours found nothing on them (27x
# 'no object detected in 18 chunks'). When the colour scan finds nothing, the largest
# contrast candidate against the estimated table background is tracked instead: pixels
# whose saturation or brightness deviates from the frame median, plus strong gradient
# response; compactness filters drop thin streaks (shadow bars) and whole-frame clutter.
CAND_MIN_PIXELS = 30       # same bar as the coloured path (_MIN_BLOB_PIXELS); compactness
                           # filters below do the junk rejection
CAND_SAT_MARGIN = 30       # saturation above the background median by this
CAND_VAL_MARGIN = 36       # |brightness - background median| by this
CAND_GRAD_THRESH = 80.0    # absolute Sobel-response floor …
CAND_GRAD_K = 2.0          # …and above mean + K*std of the frame (adapts to texture)
CAND_FILL_MIN = 0.30       # compactness: area / bbox area (rejects elongated shadow streaks)
CAND_SPAN_MAX = 0.85       # reject blobs spanning the frame in both axes (scene clutter)

# --- task-grammar extensions (task_parser.parse_task_text drives these) -------------
DQ = 'robot_state.dq'  # optional per contract; when absent the FSM falls back to position stall
# Drawer skill: the cabinet is assumed in front of the start pose (+X — the same blind
# guess as CARRY_OFFSET below); the wrist camera may refine the handle if it sees one.
HANDLE_FORWARD = 0.28  # metres from the start xy to the nominal drawer-handle point
HANDLE_HEIGHT_DROP = 0.05  # metres below the cruise height
DRAWER_STEP = 0.03  # metres per chunk along the drawer axis
DRAWER_ACT_BUDGET = 14  # chunks of pulling/pushing before stopping without a clear stall
DQ_STALL_NORM = 0.06  # L2 |dq| (rad/s) below which the arm counts as not moving
EE_STALL_MOVE = 0.004  # metres: less than this per chunk means stalled (when dq is absent)
STALL_CHUNKS = 3  # consecutive stalled chunks = end-of-travel signal (min. 2 motion chunks first)
DRAWER_REGRASP = 1  # grip-lost / failed-closure retries inside the drawer skill
DOOR_LIFT = 0.08  # metres of upward pull for hinged fronts (open: up-and-back diagonal)
DRAWER_REGRASP_DZ = 0.04  # open mode: re-grasp retries probe this much lower (blind point misses)
# Fronts that open by hinge (pull up/back) rather than by slide (pull toward the robot).
# Slide: drawer/dresser/desk/table/shelf (drawers). Hinge: everything with a door.
HINGED_OBJECTS = {'door', 'cabinet', 'cupboard', 'refrigerator', 'oven', 'dishwasher',
                  'wardrobe', 'freezer', 'microwave'}
# Place targets by relation; all horizontal axis meanings carry the CARRY_OFFSET caveat
# (screen left/right cannot be resolved to a world axis without camera extrinsics).
PLACE_LATERAL = 0.15  # metres for 'left'/'right' targets
BIN_RELEASE_DROP = 0.10  # into-container: release this far below the cruise height
PLATE_HEIGHT = 0.02  # onto-plate/onto-pot: table height + this
# HSV ranges (OpenCV H in [0, 180]). Red wraps, hence two ranges.
COLOR_RANGES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    'red': (np.array([0, 70, 50]), np.array([10, 255, 255])),
    'red2': (np.array([170, 70, 50]), np.array([180, 255, 255])),
    'green': (np.array([35, 70, 50]), np.array([85, 255, 255])),
    'blue': (np.array([95, 70, 50]), np.array([135, 255, 255])),
    'yellow': (np.array([18, 70, 50]), np.array([35, 255, 255])),
}
_MIN_BLOB_PIXELS = 30


def detect_blob(image: np.ndarray, color: str | None) -> tuple[float, float, float] | None:
    """Largest colour blob centroid in a wrist RGB frame.

    Returns ``(u, v, area_frac)`` with the centroid normalized to ``[-1, 1]`` from the
    image centre, or ``None`` when nothing matching is found. With ``color=None`` (task
    names no known colour) the largest blob of any known colour wins.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'image.wrist must be (H, W, 3), got {image.shape}')
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if color == 'red':
        ranges = [COLOR_RANGES['red'], COLOR_RANGES['red2']]
    elif color in COLOR_RANGES:
        ranges = [COLOR_RANGES[color]]
    else:
        ranges = [COLOR_RANGES[name] for name in ('red', 'red2', 'green', 'blue', 'yellow')]
    best: tuple[int, int, int] | None = None  # (area, cx, cy) in pixels
    for low, high in ranges:
        mask = cv2.inRange(hsv, low, high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, count):  # 0 is the background
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < _MIN_BLOB_PIXELS or (best is not None and area <= best[0]):
                continue
            best = (area, int(centroids[i, 0]), int(centroids[i, 1]))
    if best is None:
        return None
    area, cx, cy = best
    h, w = image.shape[:2]
    return (2.0 * cx / w - 1.0, 2.0 * cy / h - 1.0, area / float(h * w))


def parse_task(task: str) -> tuple[str | None, bool]:
    """Legacy colour + place-intent probe, now a thin wrapper over the task grammar.

    The FSM itself uses :func:`task_parser.parse_task_text` (skill/object/target); this
    helper only preserves the old two-value contract for any external caller.
    """
    intent = parse_task_text(task)
    return intent['color'], intent['skill'] in (SKILL_PLACE, SKILL_MOVE_DIR)


def detect_blob_soft(image: np.ndarray | None, color: str | None) -> tuple[float, float, float] | None:
    """``detect_blob`` that never raises: drawer vision is a refinement, not a requirement.

    Returns ``None`` when the image is unusable (missing key, detector failure — e.g. a
    stubbed cv2 in unit tests) or no blob matches; the drawer skill then falls back to
    the blind nominal handle point.
    """
    if image is None:
        return None
    try:
        return detect_blob(image, color)
    except Exception:  # noqa: BLE001 - any detector failure degrades to the blind path
        logger.debug('scripted policy: wrist blob detection failed, using the blind path', exc_info=True)
        return None


def _in_excluded(u: float, v: float, exclude: list | tuple) -> bool:
    return any((u - eu) ** 2 + (v - ev) ** 2 < er * er for eu, ev, er in exclude)


def detect_candidates(image: np.ndarray, exclude: list | tuple = ()) -> list[tuple[float, float, float]]:
    """Colorless object candidates: blobs that contrast with the estimated table background.

    The wrist camera mostly sees the tabletop, so the frame median saturation and
    brightness estimate the background; pixels that deviate (more saturated, darker or
    brighter) plus a strong gradient response form the candidate mask. Connected
    components above :data:`CAND_MIN_PIXELS` survive, ordered largest first; thin
    elongated regions (shadow bars) and whole-frame clutter are dropped by the
    compactness/span filters, and centroids inside ``exclude`` zones are skipped (those
    are camera-fixed features — the gripper itself and its shadow — rejected earlier by
    the servo's closed-loop check).

    Returns ``(u, v, area_frac)`` tuples like :func:`detect_blob`, normalized to [-1, 1].
    Raises like ``detect_blob`` on a malformed image (callers either want that contract
    or wrap it in their own soft path).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'image.wrist must be (H, W, 3), got {image.shape}')
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    # The background is estimated from the BORDER ring of the frame, not the whole
    # frame: at grasp height the object covers the centre, and a whole-frame median
    # would flip onto the object and make the tabletop the "outlier" — detection died
    # exactly one step before the grasp area trigger (found in the fix regression run).
    band = max(2, min(h, w) // 10)
    border = np.ones((h, w), dtype=bool)
    border[band:h - band, band:w - band] = False
    bg_s = float(np.median(sat[border]))
    bg_v = float(np.median(val[border]))
    sat_mask = sat >= min(255.0, bg_s + CAND_SAT_MARGIN)
    val_mask = np.abs(val - bg_v) >= CAND_VAL_MARGIN
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    grad_mask = grad >= max(CAND_GRAD_THRESH, float(grad.mean()) + CAND_GRAD_K * float(grad.std()))
    mask = ((sat_mask | val_mask | grad_mask).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs: list[tuple[float, float, float]] = []
    for i in range(1, count):  # 0 is the background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < CAND_MIN_PIXELS:
            continue
        bbox_w = int(stats[i, cv2.CC_STAT_WIDTH])
        bbox_h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bbox_w > CAND_SPAN_MAX * w and bbox_h > CAND_SPAN_MAX * h:
            continue  # clutter spanning the whole frame
        if area < CAND_FILL_MIN * bbox_w * bbox_h:
            continue  # elongated streak, e.g. a shadow bar or an edge line
        u = 2.0 * float(centroids[i, 0]) / w - 1.0
        v = 2.0 * float(centroids[i, 1]) / h - 1.0
        if _in_excluded(u, v, exclude):
            continue
        blobs.append((u, v, area / float(h * w)))
    blobs.sort(key=lambda blob: -blob[2])
    return blobs


def detect_candidates_soft(image: np.ndarray | None, exclude: list | tuple = ()) -> list[tuple[float, float, float]]:
    """``detect_candidates`` that never raises: open-mode handle refinement is optional."""
    if image is None:
        return []
    try:
        return detect_candidates(image, exclude)
    except Exception:  # noqa: BLE001 - any detector failure degrades to the blind path
        logger.debug('scripted policy: colorless candidate detection failed', exc_info=True)
        return []


def detect_object(image: np.ndarray, color: str | None, exclude: list | tuple = (),
                  anchor: tuple[float, float] | None = None) -> tuple[float, float, float] | None:
    """The pick path's detector: task colour when named, then any vision colour, then a
    colorless contrast candidate (FIX-2).

    A NAMED colour is authoritative, with one anchored exception: when the colour scan
    has gone dark but a candidate sits near the LAST coloured fix (``anchor``), the
    object is probably still there and its colour was washed out by the JPEG chroma
    subsampling (the mock's tiny 8x8 red square flickers with the subpixel phase) — that
    candidate is tracked. Candidates far from the anchor are refused: the gate-round-1
    polygon run showed the unanchored fallback hijacking the servo with wood-texture
    garbage blobs while the colour scan was correctly reporting None (the object was
    simply out of view). The colorless fallback without an anchor exists for the 90% of
    tasks that name NO colour. Blobs inside an ``exclude`` zone are skipped in all
    branches (camera-fixed features — the gripper and its shadow). Raises like
    ``detect_blob`` on a malformed image.
    """
    found = detect_blob(image, color)
    if found is not None and not _in_excluded(found[0], found[1], exclude):
        return found
    try:
        blobs = detect_candidates(image, exclude)
    except Exception:  # noqa: BLE001 - the fallback must not break the colour path
        return None
    if color is not None:
        if anchor is None:
            return None
        for blob in blobs:
            if abs(blob[0] - anchor[0]) <= 0.4 and abs(blob[1] - anchor[1]) <= 0.4:
                return blob
        return None
    return blobs[0] if blobs else None


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    """Normalize a wxyz quaternion; fall back to identity when it is degenerate."""
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


def quat_mul_z90(quat: np.ndarray) -> np.ndarray:
    """Rotate a wxyz quaternion by +90 deg about the world Z axis (world-frame
    pre-multiplication: ``qz90 @ q``).

    The drawer open skill re-orients the gripper because the pads must close ACROSS the
    pull axis: on the local gym the pads sit at +-x of the TCP at the home orientation
    (measured: pad geom world offsets) while the handle rod lies along x — closing along
    the rod cannot grip it.
    """
    half = math.sqrt(0.5)
    w1, x1, y1, z1 = half, 0.0, 0.0, half  # +90 deg about z, wxyz
    w2, x2, y2, z2 = normalize_quat(quat)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class PickStateMachine:
    """One episode's deterministic pick state machine, advanced once per chunk.

    Each ``step`` call corresponds to one client inference round (about
    ``horizon_sec`` of wall clock); the chunk it returns is played at ``fps`` by the
    client's TrajectoryPlayer, and the next observation arrives when the chunk ends.
    """

    def __init__(self) -> None:
        self.state = 'init'
        # Anchors from the first observation.
        self.rotation = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz, captured once and reused
        self.origin_xy = np.zeros(2)
        self.origin_z = 0.35
        # Servo bookkeeping: closed-loop gain calibration (one deliberate probe per axis,
        # measured against the actual EE motion) plus the camera-fixed-feature blacklist —
        # the wrist camera sees the gripper and its shadow, and a candidate that does not
        # move in the image while the EE moves is rigid with the camera.
        self.gain_sign = np.array([1.0, 1.0])  # per-axis sign of image-offset -> metres
        self.gain_vec = np.zeros(2)  # measured signed doffset/dEE per axis (0 = unmeasured)
        self.probed = [False, False]  # axis got a conclusive probe measurement
        self.probe_retries = 0
        self.motion_ref: dict[str, Any] | None = None  # {'offset0','xy0','probe_axis'} of the pending move
        self.exclude_zones: list[tuple[float, float, float]] = []
        self.centered_run = 0
        self.search_count = 0
        self.spiral_i = 0
        # Grasp/place bookkeeping.
        self.grasp_xy = np.zeros(2)
        self.grasp_z = 0.0
        self.grasp_area = 0.0
        self.grasp_grip_stopped = False
        self.hold_run = 0
        self.retries = 0
        self.error_run = 0
        self.chunk_count = 0
        self.color_anchor: tuple[float, float] | None = None  # the last coloured fix (u, v)
        self.last_grip_cmd: float | None = None  # for the safe-hold path (never drop a held object)
        self.grip_air: float | None = None  # measured free-air closure level (venv-dependent)
        self.prev_grip_obs: float | None = None  # the previous chunk's grip, for the settle test
        self.want_place = False
        self.color: str | None = None
        # Task-grammar intent (task_parser.parse_task_text): skill/target for place and
        # the drawer bookkeeping below.
        self.place_target: str | None = None
        self.drop_xy: np.ndarray | None = None
        self.place_z: float | None = None
        self.drawer_mode: str | None = None  # 'open' | 'close'
        self.drawer_kind = 'slide'  # open-pull geometry: 'slide' (drawer) | 'hinge' (door-like)
        self.drawer_search: dict[str, Any] | None = None  # open-mode vision search state
        self.pull_dir_vision: np.ndarray | None = None  # measured cabinet->handle pull direction
        self._center_offset = np.zeros(2)  # last offset seen by the drawer centring loop
        self.drawer_phase = ''  # 'lateral' | 'vertical' inside drawer_approach
        self.handle_xy = np.zeros(2)
        self.handle_z = 0.0
        self.pull_dir = np.zeros(2)
        self.pull_start = np.zeros(2)
        self.drawer_regrasps = 0
        self.drawer_steps = 0
        self.pull_reversed = False  # open mode: the slide-axis sign was corrected once
        self.safety_fallbacks = 0
        self.last_good_ee = None
        self.stall_run = 0
        self.approach_stall_run = 0
        self.prev_drawer_xy: np.ndarray | None = None

    # -- helpers ----------------------------------------------------------

    def _clamped(self, current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        """Move ``target`` onto the per-chunk step budget around ``current``.

        The budget applies to the TRANSLATION only: the quaternion is passed through
        untouched (the rig's IK handles the re-orientation). Clamping the full pose
        vector let a large orientation delta — the drawer skill's 90-degree gripper
        rotation — crush the commanded translation to a few percent, so every search
        move crawled at ~4% speed (the whole open-drawer contact search stalled on
        that)."""
        delta = np.asarray(target[:3], dtype=np.float64) - np.asarray(current[:3], dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        if norm <= max_step:
            return target
        out = np.array(current, dtype=np.float64)
        out[:3] = current[:3] + delta * (max_step / norm)
        return out

    def _pose_target(self, xyz: np.ndarray) -> np.ndarray:
        return np.concatenate([xyz, self.rotation]).astype(np.float64)

    def _chunk(self, ee: np.ndarray, target: np.ndarray, grip: float, steps: int, end: str | None) -> list[dict[str, Any]]:
        """Interpolate EE translation from ``ee`` to ``target`` across the chunk; rotation is constant."""
        self.last_grip_cmd = float(np.clip(grip, 0.0, 1.0))
        elements: list[dict[str, Any]] = []
        for i in range(steps):
            t = i / max(1, steps - 1)
            xyz = ee[:3] + (target[:3] - ee[:3]) * t
            vec = np.concatenate([xyz, target[3:], [grip]]).astype(np.float32)
            # any non-finite component makes the client discard the whole chunk
            if not np.isfinite(vec).all():
                vec = np.nan_to_num(vec, nan=0.0, posinf=6.0, neginf=-6.0)
            element: dict[str, Any] = {ACTION_FIELD: vec}
            if end is not None:
                element['end'] = end
            elements.append(element)
        if not elements:
            # defensive: always emit at least one hold element
            vec = np.concatenate([ee[:3], target[3:], [grip]]).astype(np.float32)
            elements.append({ACTION_FIELD: np.nan_to_num(vec, nan=0.0, posinf=6.0, neginf=-6.0)})
        return elements

    # -- main entry -------------------------------------------------------

    def step(self, observation: dict[str, Any], fps: float, steps: int) -> list[dict[str, Any]]:
        # Platform hardening (submission 6482ea4f… robolab episode): a chunk with
        # zero elements is unusable for the client — never emit one.
        steps = max(1, int(steps))
        observation = self._sanitise(observation)
        try:
            return self._step_inner(observation, fps, steps)
        except Exception as exc:  # never let the client see an error response
            logger.warning('policy step failed (%s: %s), serving safe hold', type(exc).__name__, exc)
            self.safety_fallbacks += 1
            ee = self._safe_ee(observation)
            end = 'give_up' if self.safety_fallbacks > 10 else None
            return self._chunk(ee, ee, self._safe_grip(observation), steps, end)

    def _safe_ee(self, observation: dict[str, Any]) -> np.ndarray:
        # last good pose, else the deterministic home pose
        if getattr(self, 'last_good_ee', None) is not None:
            return self.last_good_ee
        return np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])

    def _safe_grip(self, observation: dict[str, Any]) -> float:
        # the safe-hold path must not raise on the very observation that broke the step;
        # prefer the last commanded grip (a held object must not be dropped), else open
        if self.last_grip_cmd is not None:
            return float(self.last_grip_cmd)
        try:
            return self._grip_of(observation)
        except Exception:  # noqa: BLE001 - nothing commanded yet: keep the gripper open
            return 0.0

    def _sanitise(self, observation: dict[str, Any]) -> dict[str, Any]:
        # repair observations a sim can deliver mid-episode (NaN pose, empty frame)
        obs = dict(observation)
        ee = obs.get(EE_POSE)
        if ee is not None:
            ee = np.asarray(ee, dtype=np.float64).reshape(-1)
            if not np.isfinite(ee).all():
                obs[EE_POSE] = self._safe_ee(obs).astype(np.float32)
            else:
                self.last_good_ee = ee.astype(np.float64)
        img = obs.get(WRIST_IMAGE)
        if img is None or not hasattr(img, 'shape') or len(getattr(img, 'shape', ()) or ()) != 3 or img.shape[0] < 8 or img.shape[1] < 8:
            obs[WRIST_IMAGE] = np.zeros((360, 640, 3), dtype=np.uint8)
        return obs

    def _step_inner(self, observation: dict[str, Any], fps: float, steps: int) -> list[dict[str, Any]]:
        self.chunk_count += 1
        if self.chunk_count > EPISODE_BUDGET:
            logger.warning('scripted policy hit the episode chunk budget, giving up')
            return self._chunk(
                self._ee_of(observation), self._ee_of(observation), self._grip_of(observation),
                steps, 'give_up',
            )
        # the grip's free-air closure level, measured as the running max while the
        # gripper is commanded fully closed: the adapter normalizes air to 1.0 via its
        # venv-specific DRIVER_CLOSED calibration, and the gate venv's travel differs —
        # every grip threshold below is relative to this measured level, never absolute
        try:
            grip_now = self._grip_of(observation)
            if (self.last_grip_cmd or 0.0) >= 0.98:
                self.grip_air = max(self.grip_air or 0.0, float(grip_now))
        except Exception:  # noqa: BLE001 - a missing grip must not break the step
            pass
        ee = self._ee_of(observation)
        if self.state == 'init':
            return self._init(observation, ee, steps)
        status = int(np.asarray(observation.get(STATUS, 0)).item())
        if status == 3:
            self.error_run += 1
            if self.error_run >= ERROR_BUDGET:
                logger.warning('scripted policy: arm error status persisted, giving up')
                return self._chunk(ee, ee, self._grip_of(observation), steps, 'give_up')
            return self._chunk(ee, ee, self._grip_of(observation), steps, None)  # hold and wait for recovery
        self.error_run = 0
        handler = {
            'servo': self._servo,
            'descend': self._descend,
            'grasp': self._grasp,
            'lift': self._lift,
            'hold_after_task': self._hold_after_task,
            'carry': self._carry,
            'place': self._place,
            'release': self._release,
            'retract': self._retract,
            'drawer_approach': self._drawer_approach,
            'drawer_grasp': self._drawer_grasp,
            'drawer_act': self._drawer_act,
            'drawer_sweep': self._drawer_sweep,
            'drawer_probe': self._drawer_probe,
            'done': self._done,
        }[self.state]
        return handler(observation, ee, steps)

    # -- states -----------------------------------------------------------

    def _init(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        self.rotation = normalize_quat(ee[3:])
        self.origin_xy = ee[:2].copy()
        self.origin_z = float(ee[2])
        task = str(observation.get(TASK, ''))
        intent = parse_task_text(task)
        self.color = intent['color']  # None unless the task names a vision-known colour
        logger.info('scripted policy: task=%r intent=%s descriptor=%r',
                    task, intent, str(observation.get(DESCRIPTOR, '')))
        skill = intent['skill']
        if skill == 'give_up':
            logger.warning('scripted policy: task %r is outside the task grammar, giving up', task)
            return self._chunk(ee, ee, self._grip_of(observation), steps, 'give_up')
        if skill in ('open_drawer', 'close_drawer'):
            self.drawer_mode = 'open' if skill == 'open_drawer' else 'close'
            # FIX-3: open mirrors close with the pull geometry of the target furniture —
            # drawers slide along the axis toward the robot, hinged fronts (fridge/oven/
            # dishwasher/cabinet doors) pull up-and-back (submission-3 log families:
            # OpenDataGen/OpenHard 0/30 on dresser/drawer/table/refrigerator/oven/…).
            self.drawer_kind = 'hinge' if self.drawer_mode == 'open' and intent['object'] in HINGED_OBJECTS \
                else 'slide'
            # Open mode re-orients the gripper +90 deg about world Z so the pads close
            # across the pull axis (close mode keeps the captured orientation untouched).
            self.rotation = quat_mul_z90(self.rotation)
            self.want_place = False
            self.place_target = intent['target']
            self.handle_xy = self.origin_xy + np.array([HANDLE_FORWARD, 0.0])
            self.handle_z = self.origin_z - HANDLE_HEIGHT_DROP
            self.drawer_phase = 'lateral'
            self.state = 'drawer_approach'
            logger.info('scripted policy: drawer skill mode=%s kind=%s handle=%s z=%.3f',
                        self.drawer_mode, self.drawer_kind, self.handle_xy, self.handle_z)
            return self._drawer_approach(observation, ee, steps)
        self.want_place = skill in (SKILL_PLACE, SKILL_MOVE_DIR)
        self.place_target = intent['target']
        self.state = 'servo'
        return self._servo(observation, ee, steps)

    def _detect(self, image: np.ndarray) -> tuple[float, float, float] | None:
        """Pick-path detection with the episode's camera-fixed exclusion zones applied;
        a colour-named task also carries the last coloured fix as the candidate anchor."""
        if self.color is None:
            return detect_object(image, None, self.exclude_zones)
        found = detect_blob(image, self.color)
        if found is not None and not _in_excluded(found[0], found[1], self.exclude_zones):
            self.color_anchor = (float(found[0]), float(found[1]))  # a fresh coloured fix
            return found
        return detect_object(image, self.color, self.exclude_zones, self.color_anchor)

    def _spiral_chunk(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """One chunk of the growing search spiral at the cruise height (or a give_up)."""
        self.centered_run = 0
        self.search_count += 1
        if self.search_count > SEARCH_BUDGET:
            logger.warning('scripted policy: no object detected in %d chunks, giving up', SEARCH_BUDGET)
            return self._chunk(ee, ee, self._grip_of(observation), steps, 'give_up')
        radius = min(SPIRAL_MAX_RADIUS, SPIRAL_STEP * self.spiral_i)
        angle = 2.39996 * self.spiral_i  # golden-angle spacing
        self.spiral_i += 1
        offset_xy = self.origin_xy + radius * np.array([math.cos(angle), math.sin(angle)])
        xy = self._clamped(ee[:2], offset_xy, MAX_XY_STEP)
        target = self._pose_target(np.array([*xy, self.origin_z]))
        return self._chunk(ee, target, 0.0, steps, None)

    def _servo(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """Hover at the cruise height and centre the object in the wrist image.

        Gain signs are calibrated closed-loop: each axis gets one deliberate probe move
        (:data:`PROBE_STEP` along that world axis); the next observation gives the actual
        EE motion and the actual image-offset change, and ``sign = -sign(Δoffset/Δee)``
        fixes the axis deterministically. A tracked feature that does not move in the
        image while the EE moved is rigid with the camera (the gripper itself, its
        shadow, background scenery) — it is blacklisted and the search continues.
        """
        blob = self._detect(observation[WRIST_IMAGE])
        if blob is None:
            self.motion_ref = None
            return self._spiral_chunk(observation, ee, steps)

        u, v, area = blob
        offset = np.array([u, v])
        # Verify the previously commanded move against the measured reality. A move that
        # has not yet produced enough actual EE motion leaves the reference in place: the
        # next chunk accumulates onto it (the local gym tracks only ~0.018 m per chunk,
        # so a per-chunk threshold never judged anything there).
        if self.motion_ref is not None:
            moved = ee[:2] - self.motion_ref['xy0']
            shifted = offset - self.motion_ref['offset0']
            if float(np.linalg.norm(shifted)) > PROBE_JUMP_TOL:
                self.motion_ref = None  # the tracked blob changed: restart the measurement
            elif float(np.linalg.norm(moved)) >= MOTION_VERIFY_STEP:
                if float(np.linalg.norm(shifted)) < CAMERA_FIXED_TOL:
                    # Feature rigid with the camera: reject the region and re-search.
                    logger.warning('scripted policy: blob at (%.2f, %.2f) did not move with the EE '
                                   '(%.2f m); rejecting it as camera-fixed (gripper/shadow/scenery)',
                                   u, v, float(np.linalg.norm(moved)))
                    self.exclude_zones.append((u, v, EXCLUDE_RADIUS))
                    if len(self.exclude_zones) > MAX_EXCLUDE_ZONES:
                        self.exclude_zones.pop(0)
                    self.motion_ref = None
                    return self._spiral_chunk(observation, ee, steps)
                axis = self.motion_ref['probe_axis']
                if axis is not None:
                    actual_move = float(moved[axis])
                    if abs(actual_move) >= PROBE_MIN_MOVE and abs(float(shifted[axis])) > 1e-6:
                        gain = float(shifted[axis]) / actual_move
                        if abs(gain) < GAIN_MIN_OBJECT:
                            # the blob moves with the EE (shadow) or is too far to grasp:
                            # blacklist the region and keep searching
                            logger.warning('scripted policy: probe rejects a shadow-like blob at '
                                           '(%.2f, %.2f) (doffset/dee=%.2f)', u, v, gain)
                            self.exclude_zones.append((u, v, EXCLUDE_RADIUS))
                            if len(self.exclude_zones) > MAX_EXCLUDE_ZONES:
                                self.exclude_zones.pop(0)
                            self.motion_ref = None
                            self.probed[axis] = False
                            return self._spiral_chunk(observation, ee, steps)
                        self.gain_vec[axis] = float(np.clip(gain, *GAIN_CLIP))
                        self.gain_sign[axis] = -1.0 if gain > 0 else 1.0
                        self.probed[axis] = True
                        self.probe_retries = 0
                        logger.info('scripted policy: closed-loop gain on axis %d: doffset/dee=%.2f '
                                    '-> sign %.0f factor %.3f', axis, gain, self.gain_sign[axis],
                                    self._servo_factor(axis))
                    elif abs(actual_move) < PROBE_MIN_MOVE:
                        if self.probe_retries < PROBE_RETRIES:
                            self.probe_retries += 1  # inconclusive (arm did not move): retry
                        else:
                            logger.info('scripted policy: probe on axis %d inconclusive after %d tries, '
                                        'proceeding with sign %.0f', axis, PROBE_RETRIES, self.gain_sign[axis])
                            self.probed[axis] = True
                    else:
                        logger.info('scripted policy: probe on axis %d saw no offset response '
                                    '(do/dee=%.3f); keeping sign %.0f', axis,
                                    float(shifted[axis]) / actual_move, self.gain_sign[axis])
                        self.probed[axis] = True
                self.motion_ref = None

        # One deliberate probe per axis before trusting the default signs. The probe
        # steps TOWARD the tracked object so it stays in the field of view (a fixed +
        # direction pushed edge-of-frame objects out and the probe never concluded).
        probe_axis = next((axis for axis in (0, 1) if not self.probed[axis]), None)
        if probe_axis is not None:
            step_vec = np.zeros(2)
            step_vec[probe_axis] = PROBE_STEP * (1.0 if offset[probe_axis] >= 0 else -1.0)
            target_xy = np.clip(ee[:2] + step_vec, self.origin_xy - 0.6, self.origin_xy + 0.6)
            self.motion_ref = {'offset0': offset.copy(), 'xy0': ee[:2].copy(), 'probe_axis': probe_axis}
            target = self._pose_target(np.array([*target_xy, self.origin_z]))
            return self._chunk(ee, self._clamped(ee, target, MAX_XY_STEP), 0.0, steps, None)

        if np.abs(offset).max() < CENTER_TOL:
            self.centered_run += 1
        else:
            self.centered_run = 0
        if self.centered_run >= CONVERGE_CHUNKS:
            self.grasp_xy = ee[:2].copy()
            self.state = 'descend'
            return self._descend(observation, ee, steps)
        # Move toward the centroid; the calibrated per-axis gain (sign AND magnitude)
        # absorbs the unknown camera-mount orientation and keeps the loop stable: a raw
        # SERVO_GAIN at |doffset/dEE| ~ 10 over-corrects (loop gain 2.5) and oscillates.
        self.motion_ref = {'offset0': offset.copy(), 'xy0': ee[:2].copy(), 'probe_axis': None}
        delta_xy = np.array([self._servo_factor(0) * self.gain_sign[0] * offset[0],
                             self._servo_factor(1) * self.gain_sign[1] * offset[1]])
        delta_xy = np.clip(delta_xy, -MAX_XY_STEP, MAX_XY_STEP)
        target_xy = ee[:2] + delta_xy
        # Keep inside a sane workspace around the episode start.
        target_xy = np.clip(target_xy, self.origin_xy - 0.6, self.origin_xy + 0.6)
        target = self._pose_target(np.array([*target_xy, self.origin_z]))
        return self._chunk(ee, self._clamped(ee, target, MAX_XY_STEP), 0.0, steps, None)

    def _servo_factor(self, axis: int) -> float:
        """Per-axis commanded metres per unit of image offset: SERVO_GAIN bounded so the
        measured gain makes the loop decay (|gain| * factor <= GAIN_SAFETY)."""
        measured = abs(float(self.gain_vec[axis]))
        if measured > 0:
            return min(SERVO_GAIN, GAIN_SAFETY / measured)
        return SERVO_GAIN

    def _descend(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """Lower straight down over the centred object until grasp height."""
        blob = self._detect(observation[WRIST_IMAGE])
        z_floor = self.origin_z - FLOOR_DROP
        near_floor = ee[2] <= z_floor + 1e-6
        if blob is not None and blob[2] >= AREA_GRASP_FRAC and not near_floor:
            self.grasp_xy = ee[:2].copy()
            self.grasp_z = float(ee[2])
            self.grasp_area = blob[2]
            self.hold_run = 0
            self.state = 'grasp'
            return self._grasp(observation, ee, steps)
        # Keep a light lateral correction while descending (blob may drift).
        target_xy = self.grasp_xy
        if blob is not None:
            correction = np.array([0.4 * self._servo_factor(0) * self.gain_sign[0] * blob[0],
                                   0.4 * self._servo_factor(1) * self.gain_sign[1] * blob[1]])
            target_xy = ee[:2] + np.clip(correction, -0.03, 0.03)
        z = max(z_floor, ee[2] - DESCEND_DZ)
        target = self._pose_target(np.array([*target_xy, z]))
        return self._chunk(ee, self._clamped(ee, target, MAX_Z_STEP), 0.0, steps, None)

    def _grasp(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """Hold still and close the gripper, then verify the object came along."""
        self.hold_run += 1
        if self.hold_run < GRASP_HOLD_CHUNKS:
            target = self._pose_target(np.array([*self.grasp_xy, self.grasp_z]))
            return self._chunk(ee, self._clamped(ee, target, MAX_Z_STEP), 1.0, steps, None)
        grip_obs = self._grip_of(observation)
        blob = self._detect(observation[WRIST_IMAGE])
        # Robotiq fingers stop at the object: closure clearly below 1.0 means something
        # is inside. The blob staying big and centred is the second, weaker signal.
        stopped = grip_obs < 0.92  # absolute: the pick path is validated against the
        # polygon's real contact and the wire mock; the measured-air baseline is only
        # sound in the drawer search, where the drag closes the gripper over free space
        following = blob is not None and blob[2] >= 0.5 * self.grasp_area and abs(blob[0]) < 0.5 and abs(blob[1]) < 0.5
        if stopped or following:
            logger.info('scripted policy: grasp verified (grip=%.2f stopped=%s following=%s)',
                        grip_obs, stopped, following)
            # the post-lift re-verification only makes sense when the fingers STOPPED on
            # the object at grasp time: a rig whose gripper closes fully while holding
            # (the idealized mock) reports grip ~1.0 during the lift too, and checking it
            # there would reject every good grasp
            self.grasp_grip_stopped = bool(stopped)
            self.state = 'lift'
            return self._lift(observation, ee, steps)
        if self.retries < GRASP_RETRIES:
            self.retries += 1
            self.grasp_z = max(self.origin_z - FLOOR_DROP, self.grasp_z - 0.04)
            self.hold_run = 0
            logger.info('scripted policy: grasp failed (grip=%.2f), retry %d lower', grip_obs, self.retries)
            self.state = 'descend'
            return self._descend(observation, ee, steps)
        logger.warning('scripted policy: grasp failed after %d retries, giving up', GRASP_RETRIES)
        return self._chunk(ee, ee, 0.0, steps, 'give_up')

    # -- drawer skill (open_drawer / close_drawer from the task grammar) ---

    def _drawer_approach(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """Cruise to the nominal handle point, then descend to handle height.

        The nominal point is blind (+HANDLE_FORWARD on x). If the wrist camera happens to
        centre a blob while cruising, that neighbourhood is adopted as the handle instead
        — vision refines, but never blocks (detect_blob_soft). Open mode additionally
        consults the colorless candidate detector when the colour scan sees nothing
        (FIX-3: cabinet fronts and handles are unsaturated — the colored-only adoption
        fired once in 60 drawer episodes of submission 3); close mode keeps its original
        detection path untouched.
        """
        blob = detect_blob_soft(observation.get(WRIST_IMAGE), None)
        if blob is None and self.drawer_mode in ('open', 'close'):
            candidates = detect_candidates_soft(observation.get(WRIST_IMAGE))
            if candidates:
                blob = candidates[0]
                logger.info('scripted policy: open mode using a colorless candidate for handle refinement')
        if self.drawer_phase == 'lateral' and blob is not None and blob[2] > 0.02 \
                and abs(blob[0]) < CENTER_TOL and abs(blob[1]) < CENTER_TOL:
            self.handle_xy = ee[:2].copy()
            self.drawer_phase = 'vertical'
            logger.info('scripted policy: adopting wrist blob neighbourhood as the handle')
        if self.drawer_phase == 'lateral':
            # Unreachable-nominal-point guard: when the cruise stops making progress the
            # arm has hit its reach limit short of the nominal point (gate round 2: the
            # approach froze 0.10 m short for 100+ chunks). Grasp at the reached point;
            # the attempt fails honestly and the open-mode search takes over.
            moved = float(np.linalg.norm(ee[:2] - self.prev_drawer_xy)) \
                if self.prev_drawer_xy is not None else 1.0
            self.prev_drawer_xy = ee[:2].copy()
            self.approach_stall_run = self.approach_stall_run + 1 if moved < 0.004 else 0
            if self.approach_stall_run >= STALL_CHUNKS:
                self.handle_xy = ee[:2].copy()
                self.drawer_phase = 'vertical'
                logger.warning('scripted policy: nominal drawer point unreachable (stalled at %s); '
                               'grasping here, then the open-mode search takes over',
                               np.round(ee[:2], 3))
        if self.drawer_phase == 'lateral':
            if float(np.linalg.norm(ee[:2] - self.handle_xy)) < 0.02:
                self.drawer_phase = 'vertical'
            else:
                target = self._pose_target(np.array([*self.handle_xy, self.origin_z]))
                return self._chunk(ee, self._clamped(ee, target, MAX_XY_STEP), 0.0, steps, None)
        # vertical phase
        if abs(ee[2] - self.handle_z) < 0.02:
            self.hold_run = 0
            self.state = 'drawer_grasp'
            return self._drawer_grasp(observation, ee, steps)
        target = self._pose_target(np.array([*self.handle_xy, self.handle_z]))
        return self._chunk(ee, self._clamped(ee, target, MAX_Z_STEP), 0.0, steps, None)

    def _drawer_grasp(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """Hold still, close the gripper on the handle, then arm the pull/push direction."""
        self.hold_run += 1
        grip_obs = self._grip_of(observation)
        settled = self.prev_grip_obs is not None and abs(grip_obs - self.prev_grip_obs) < 0.03
        self.prev_grip_obs = grip_obs
        # in the search mode the probe must let the grip SETTLE (the gripper's closure
        # rate is venv-dependent — a fixed 2-chunk hold 'verifies' everything on a slow
        # one and reads mid-closure as stopped on a fast one); the hold ends when the
        # grip stops changing or the extended budget is spent
        hold_needed = GRASP_HOLD_CHUNKS + (4 if self.drawer_search else 0)
        if self.hold_run < hold_needed and not (
                self.drawer_search and self.hold_run >= GRASP_HOLD_CHUNKS and settled):
            target = self._pose_target(np.array([*self.handle_xy, self.handle_z]))
            return self._chunk(ee, self._clamped(ee, target, MAX_Z_STEP), 1.0, steps, None)
        # the stopped test: air-relative in the search mode (the drag measured the true
        # free-air level over free space, so a venv-miscalibrated grip scale is handled);
        # the blind modes keep the validated absolute threshold
        air = self._air() if self.drawer_search else 1.0
        if grip_obs < air - GRIP_AIR_MARGIN_HELD:  # fingers stopped: the handle is inside
            if self.drawer_mode == 'open' and self.pull_dir_vision is not None:
                # Vision-localized handle: pull along the measured cabinet->handle
                # direction (the slide axis), not the blind approach line.
                self.pull_dir = self.pull_dir_vision
            else:
                # Drawer axis assumed along the start->handle line: pull toward the robot
                # to open, push away to close (the blind +X forward guess as CARRY_OFFSET).
                approach = self.handle_xy - self.origin_xy
                approach = approach / max(1e-6, float(np.linalg.norm(approach)))
                self.pull_dir = -approach if self.drawer_mode == 'open' else approach
            self.pull_start = ee[:2].copy()
            self.stall_run = 0
            self.drawer_steps = 0
            self.prev_drawer_xy = ee[:2].copy()
            self.state = 'drawer_act'
            logger.info('scripted policy: drawer handle grasped (grip=%.2f), mode=%s pull_dir=%s',
                        grip_obs, self.drawer_mode, self.pull_dir)
            return self._drawer_act(observation, ee, steps)
        if self.drawer_regrasps < DRAWER_REGRASP and not self.drawer_search:
            self.drawer_regrasps += 1
            self.hold_run = 0
            if self.drawer_mode == 'open':
                # FIX-3: the blind nominal handle point missed — retry a step lower
                # (close mode keeps its original behaviour untouched).
                self.handle_z = max(self.origin_z - FLOOR_DROP, self.handle_z - DRAWER_REGRASP_DZ)
                logger.info('scripted policy: drawer grasp failed (grip=%.2f), re-grasping lower z=%.3f',
                            grip_obs, self.handle_z)
            else:
                logger.info('scripted policy: drawer grasp failed (grip=%.2f), re-grasping', grip_obs)
            return self._chunk(ee, ee, 0.0, steps, None)  # open and try closing again
        if self.drawer_mode == 'open' and self.drawer_search:
            # probing the grid: this point held nothing (grip stayed at full closure)
            logger.info('scripted policy: open probe %d/%d held nothing (grip=%.2f), next probe',
                        self.drawer_search['i'], len(self.drawer_search['probes']), grip_obs)
            self.drawer_search['stage'] = 0
            self.state = 'drawer_probe'
            return self._drawer_probe(observation, ee, steps)
        if True:
            # BOTH modes fall back to the contact search before giving up: the blind
            # nominal point can be occupied by other scenery entirely (gate round 1)
            # or — close-after-open on the polygon (composite 2026-11-21) — sit 5-13 cm
            # beyond the blind nominal point. The act states branch pull/push by
            # drawer_mode, so the shared search serves both directions.
            logger.warning('scripted policy: drawer handle never grasped blind (grip=%.2f); '
                           'starting the contact search', grip_obs)
            self._start_drawer_search()
            return self._drawer_sweep(observation, ee, steps)
        logger.warning('scripted policy: drawer handle never grasped, giving up')
        return self._chunk(ee, ee, 0.0, steps, 'give_up')

    def _drawer_act(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """Pull/push along the drawer axis in DRAWER_STEP bites with stall detection.

        End-of-travel signal: |dq| collapse when the rig provides ``robot_state.dq``,
        commanded-vs-achieved position stall otherwise. Grip guard: fully-closed fingers
        mean the handle slipped out.
        """
        self.drawer_steps += 1
        grip_obs = self._grip_of(observation)
        lost_level = (self._air() - GRIP_AIR_MARGIN_LOST) if self.drawer_search else 0.95
        if grip_obs > lost_level and self.drawer_steps > 1:
            if self.drawer_regrasps < DRAWER_REGRASP:
                self.drawer_regrasps += 1
                self.hold_run = 0
                self.state = 'drawer_grasp'
                logger.info('scripted policy: lost the drawer handle (grip=%.2f), re-grasping', grip_obs)
                return self._drawer_grasp(observation, ee, steps)
            if self.drawer_mode == 'open' and self.drawer_search:
                # the pull tore this probe's grip (an edge grasp on scenery): the grid
                # already advanced — try the next probe point
                logger.info('scripted policy: open probe grip lost during the pull, next probe')
                self.drawer_search['stage'] = 0
                self.state = 'drawer_probe'
                return self._drawer_probe(observation, ee, steps)
            logger.warning('scripted policy: lost the drawer handle with no retries left, giving up')
            return self._chunk(ee, ee, 1.0, steps, 'give_up')
        target_xy = ee[:2] + self.pull_dir * DRAWER_STEP
        # FIX-3: hinged fronts (fridge/oven/dishwasher/cabinet doors) open up-and-back,
        # not purely along the horizontal axis; the rise is bounded per chunk by the same
        # DRAWER_STEP budget and capped at DOOR_LIFT.
        lift = 0.0
        if self.drawer_mode == 'open' and self.drawer_kind == 'hinge':
            lift = min(DOOR_LIFT, self.drawer_steps * DRAWER_STEP)
        target = self._pose_target(np.array([*target_xy, self.handle_z + lift]))
        chunk = self._chunk(ee, self._clamped(ee, target, DRAWER_STEP), 1.0, steps, None)
        stalled = False
        if self.drawer_steps > 2:  # let commanded motion start before judging it
            dq_value = observation.get(DQ)
            if dq_value is not None:
                dq_norm = float(np.linalg.norm(np.asarray(dq_value, dtype=np.float64).reshape(-1)))
                stalled = dq_norm < DQ_STALL_NORM
                self.stall_run = self.stall_run + 1 if stalled else 0
            else:
                stalled = self.prev_drawer_xy is not None \
                    and float(np.linalg.norm(ee[:2] - self.prev_drawer_xy)) < EE_STALL_MOVE
                self.stall_run = self.stall_run + 1 if stalled else 0
        self.prev_drawer_xy = ee[:2].copy()
        end_of_travel = self.stall_run >= STALL_CHUNKS
        budget_spent = self.drawer_steps >= DRAWER_ACT_BUDGET
        if end_of_travel or budget_spent:
            progress = float(np.dot(ee[:2] - self.pull_start, self.pull_dir))
            logger.info('scripted policy: drawer %s stopping (stall=%d/%d budget=%s progress=%.3f m)',
                        self.drawer_mode, self.stall_run, STALL_CHUNKS, budget_spent, progress)
            if self.drawer_mode == 'open' and progress < 0.02:
                if self.drawer_mode == 'open' and not self.pull_reversed:
                    # The slide axis sign was a guess: the drawer jammed, so pull the
                    # opposite way along the same line (bounded: one reversal).
                    self.pull_reversed = True
                    self.pull_dir = -self.pull_dir
                    self.pull_start = ee[:2].copy()
                    self.stall_run = 0
                    self.drawer_steps = 0
                    self.prev_drawer_xy = ee[:2].copy()
                    logger.warning('scripted policy: drawer never moved (progress=%.3f); '
                                   'reversing the pull to %s', progress, self.pull_dir)
                    return self._drawer_act(observation, ee, steps)
                if self.drawer_mode == 'open' and self.drawer_search:
                    # a false grasp (the pull moved nothing even reversed): this probe
                    # point held scenery, not the handle — advance the probe grid
                    logger.info('scripted policy: open probe point held nothing (pull jammed), '
                                'next probe')
                    search['stage'] = 0
                    self.state = 'drawer_probe'
                    return self._drawer_probe(observation, ee, steps)
                if self.drawer_mode == 'open':
                    logger.warning('scripted policy: drawer never moved (progress=%.3f); '
                                   'starting the open-mode contact search', progress)
                    self._start_drawer_search()
                    return self._drawer_sweep(observation, ee, steps)
                logger.warning('scripted policy: drawer never moved; giving up')
                return self._chunk(ee, ee, 1.0, steps, 'give_up')
            self.hold_run = 0
            self.state = 'hold_after_task'  # hold the final pose for the judge, then complete
            return self._hold_after_task(observation, ee, steps)
        return chunk

    # -- open-mode contact search (FIX-3; close mode never enters these states) ---

    def _start_drawer_search(self) -> None:
        """Arm the open-mode search: re-home, transit at the cruise height, drag the
        closed gripper along the lateral contact line, then walk the grasp-probe grid
        around the contact point. The blind attempt may have left the arm anywhere at a
        lowered grasp height — the search restores the planned geometry first."""
        self.drawer_search = {'waypoints': self._sweep_waypoints(), 'i': 0,
                              'phase': 'rehome', 'probes': None, 'prev_descend_z': None,
                              'descend_stall': 0, 'drag_z': None}
        self.handle_z = self.origin_z - HANDLE_HEIGHT_DROP
        self.state = 'drawer_sweep'

    def _sweep_waypoints(self) -> list[np.ndarray]:
        """Lateral drag lines behind the start pose: one full pass to each side,
        starting SWEEP_Y_START off the y=0 axis (the self-press zone). The pass is
        UNIDIRECTIONAL (all -y waypoints first, then all +y) — alternating sides makes
        the target flip across the arm every couple of chunks and the TCP ping-pongs
        around the start of the line without ever settling on a waypoint."""
        base = self.origin_xy + np.array([-SWEEP_BACK, 0.0])
        waypoints = []
        for side in (-1.0, 1.0):
            k = 0
            while SWEEP_Y_START + SWEEP_STEP * k <= SWEEP_HALF_SPAN + SWEEP_STEP:
                waypoints.append(base + np.array([0.0, side * (SWEEP_Y_START + SWEEP_STEP * k)]))
                k += 1
        lo, hi = self.origin_xy - 0.6, self.origin_xy + 0.6
        return [np.clip(w, lo, hi) for w in waypoints]

    def _drawer_sweep(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """Re-home, transit at the cruise height, descend to the drag height, then drag
        the closed gripper along the lateral line: the actual TCP riding above the
        commanded height (the raised cabinet body) or a grip stopping while dragging is
        the contact signal. Contact builds the grasp-probe grid around the contact point
        and hands over to the probe walk."""
        search = self.drawer_search
        waypoints = search['waypoints']
        contact_z = self.origin_z - SWEEP_CONTACT_DROP
        if search.get('phase') == 'rehome':
            # return to the start pose first: the failed blind attempt leaves the arm in
            # an extended posture from which the local IK cannot fold onto the sweep line
            # (it stalls ~4 cm short and the rotated gripper presses the forearm — a
            # spurious contact). From the re-homed posture the same waypoints are free.
            if float(np.linalg.norm(ee[:2] - self.origin_xy)) >= 0.03 \
                    or abs(float(ee[2]) - self.origin_z) >= 0.03:
                target = self._pose_target(np.array([*self.origin_xy, self.origin_z]))
                return self._chunk(ee, self._clamped(ee, target, MAX_XY_STEP), 1.0, steps, None)
            search['phase'] = 'transit'
        if search.get('phase') == 'transit':
            # transit at the cruise height first: descending into the contact band on
            # the way to the line could graze unrelated scenery
            target_xy = waypoints[0]
            if float(np.linalg.norm(ee[:2] - target_xy)) >= 0.02 \
                    or abs(float(ee[2]) - self.origin_z) >= 0.02:
                target = self._pose_target(np.array([*target_xy, self.origin_z]))
                return self._chunk(ee, self._clamped(ee, target, MAX_XY_STEP), 1.0, steps, None)
            search['phase'] = 'descend'
        if search.get('phase') == 'descend':
            # descend to the drag height BEFORE arming the contact criteria: at the
            # cruise height the z-excess test would fire instantly (0.06 above contact_z).
            # The arm can stall a little above the commanded height at this fold (wrist
            # torque limits), so the drag baseline is the REACHED height: the pad zone
            # still covers the raised cabinet there and the ride signal stays alive.
            prev_z = search.get('prev_descend_z') if search.get('prev_descend_z') is not None else 1.0
            search['prev_descend_z'] = float(ee[2])
            search['descend_stall'] = search.get('descend_stall', 0) + 1 \
                if abs(float(ee[2]) - prev_z) < 0.003 else 0
            if abs(float(ee[2]) - contact_z) < 0.01 or search['descend_stall'] >= 2:
                search['drag_z'] = float(ee[2])
                search['phase'] = 'drag'
                # the drag marches along the line's side: the fingers reach FINGERTIP_REACH
                # past the pads, so the contact fires that far early — the probe grid gets
                # built from the corrected contact point
                wp_y = float(search['waypoints'][min(search['i'], len(search['waypoints']) - 1)][1])
                base_y = float(search['waypoints'][0][1])
                search['drag_dir'] = np.array([0.0, -1.0 if wp_y <= base_y else 1.0])
                logger.info('scripted policy: open contact drag starts at z=%.3f', float(ee[2]))
            else:
                target = self._pose_target(np.array([*ee[:2], contact_z]))
                return self._chunk(ee, self._clamped(ee, target, MAX_Z_STEP), 1.0, steps, None)
        try:
            grip_obs = self._grip_of(observation)
        except Exception:  # noqa: BLE001 - keep sweeping without the grip signal
            grip_obs = 1.0
        z_excess = float(ee[2]) - float(search.get('drag_z', contact_z))
        if grip_obs < self._air() - GRIP_AIR_MARGIN_STOP or z_excess > SWEEP_RIDE_Z:
            raw_contact = ee[:2].copy()
            drag_dir = search.get('drag_dir', np.array([0.0, -1.0]))
            search['contact_xy'] = raw_contact + drag_dir * FINGERTIP_REACH
            search['probes'] = self._probe_waypoints(search['contact_xy'])
            search['i'] = 0
            search['stage'] = 0
            self.state = 'drawer_probe'
            logger.info('scripted policy: open contact sweep touched the cabinet at %s '
                        '(grip=%.2f, z_excess=%.3f); probing around the corrected point %s',
                        np.round(raw_contact, 3), grip_obs, z_excess,
                        np.round(search['contact_xy'], 3))
            return self._drawer_probe(observation, ee, steps)
        i = search['i']
        if i >= len(waypoints):
            logger.warning('scripted policy: open contact sweep found nothing along the line')
            return self._chunk(ee, ee, 1.0, steps, 'give_up')
        target_xy = waypoints[i]
        if float(np.linalg.norm(ee[:2] - target_xy)) < 0.02:
            search['i'] = i + 1  # waypoint reached: move on to the next one
        drag_z = float(search.get('drag_z', contact_z))
        target = self._pose_target(np.array([*target_xy, drag_z]))
        return self._chunk(ee, self._clamped(ee, target, MAX_Z_STEP), 1.0, steps, None)

    def _probe_waypoints(self, contact_xy: np.ndarray) -> list[np.ndarray]:
        """Grasp-probe grid around the corrected contact point: the handle protrudes
        toward the robot of the cabinet front (PROBE_DX) and sits BEYOND the first
        contacted wall along the drag direction (PROBE_DY, the cabinet half-width plus
        build uncertainty). Ordered most-likely first."""
        drag_dir = self.drawer_search.get('drag_dir', np.array([0.0, -1.0])) \
            if self.drawer_search else np.array([0.0, -1.0])
        dx_values = PROBE_DX_CLOSE if self.drawer_mode == 'close' else PROBE_DX_OPEN
        probes = []
        for dx in dx_values:
            for dy in PROBE_DY:
                probes.append(contact_xy + np.array([dx, 0.0]) + drag_dir * dy)
        lo, hi = self.origin_xy - 0.6, self.origin_xy + 0.6
        return [np.clip(p, lo, hi) for p in probes]

    def _drawer_probe(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """Walk the probe grid; each probe point is approached the way that works on the
        rig: fly high on the cabinet side of it, then descend diagonally ALONG the
        contact->probe direction (down the drawer front) to the grasp height. Direct
        flights at the cruise height into the folded configuration trap the local IK and
        the TCP never arrives (measured: the diagonal descent reaches the point and the
        grip stops on the front edge within 4 chunks)."""
        search = self.drawer_search
        i = search['i']
        if i >= len(search['probes']):
            logger.warning('scripted policy: open search exhausted the probe grid, giving up')
            return self._chunk(ee, ee, 0.0, steps, 'give_up')
        probe_xy = search['probes'][i]
        contact_xy = search.get('contact_xy', self.origin_xy)
        direction = np.asarray(contact_xy, dtype=float) - np.asarray(probe_xy, dtype=float)
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
        probe_z = self.handle_z - DRAWER_PROBE_Z_DROP
        stages = [
            (probe_xy + direction * 0.085, probe_z + 0.14),
            (probe_xy + direction * 0.03, probe_z + 0.03),
            (probe_xy, probe_z),
        ]
        stage_i = min(search.get('stage', 0), len(stages) - 1)
        stage_xy, stage_z = stages[stage_i]
        # the stage flight carries a chunk budget: a stage the arm cannot settle on
        # (an IK corner on some venvs) must not hold the episode for a hundred chunks
        search['stage_chunks'] = search.get('stage_chunks', 0) + 1
        arrived = float(np.linalg.norm(ee[:2] - stage_xy)) < 0.02 \
            and abs(float(ee[2]) - stage_z) < 0.02
        if not arrived and search['stage_chunks'] <= 8:
            target = self._pose_target(np.array([*stage_xy, stage_z]))
            return self._chunk(ee, self._clamped(ee, target, MAX_XY_STEP), 0.0, steps, None)
        search['stage'] = stage_i + 1
        search['stage_chunks'] = 0
        if stage_i + 1 < len(stages):
            return self._drawer_probe(observation, ee, steps)
        self.handle_xy = probe_xy.copy()
        search['i'] = i + 1
        search['hold_run'] = 0
        self.hold_run = 0
        # the drawer slides along its front normal — the component of (handle - contact)
        # PERPENDICULAR to the drag direction (the drag hits a side wall, whose normal is
        # the drag axis; the front normal is the slide axis). Pulling along it keeps the
        # grip loaded in its strong axis: a component along the closing axis slides the
        # rod out of the pad gap and drops the grasp.
        to_handle = np.asarray(probe_xy, dtype=float) - np.asarray(contact_xy, dtype=float)
        drag_dir = search.get('drag_dir', np.array([0.0, -1.0]))
        perp = to_handle - drag_dir * float(np.dot(to_handle, drag_dir))
        norm = float(np.linalg.norm(perp))
        unit = perp / norm if norm > 1e-6 else None
        if unit is not None and self.drawer_mode == 'close':
            unit = -unit
        self.pull_dir_vision = unit
        self.state = 'drawer_grasp'
        logger.info('scripted policy: open search probes a grasp at %s (%d/%d) pull_dir=%s',
                    np.round(probe_xy, 3), i + 1, len(search['probes']),
                    None if self.pull_dir_vision is None else np.round(self.pull_dir_vision, 2))
        return self._drawer_grasp(observation, ee, steps)


    def _container_target(self) -> bool:
        """True when the parsed place relation names a container (release from above)."""
        target = self.place_target or ''
        if target.startswith('into-'):
            return target[5:] in CONTAINERS
        if target.startswith('from-to:'):
            parts = target.split(':')
            return len(parts) > 2 and parts[2] in CONTAINERS
        return False

    def _lift(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        z_target = self.grasp_z + LIFT_HEIGHT
        if self.want_place and self._container_target():
            # into-container means releasing above the vessel mouth, not at table height.
            z_target = max(z_target, self.origin_z - BIN_RELEASE_DROP)
        if ee[2] < z_target - 0.01:
            # Post-grasp re-verification (gate round 1, pick-red): a grazing contact can
            # stop the fingers (grip 0.61 < 0.92) and still slip out during the lift —
            # the grip relaxing to ~1.0 while the EE has clearly left the grasp height
            # means nothing is held. Retry lower instead of completing empty-handed.
            if self.grasp_grip_stopped and ee[2] > self.grasp_z + 0.05 \
                    and self._grip_of(observation) >= 0.95:
                if self.retries < GRASP_RETRIES:
                    self.retries += 1
                    self.grasp_z = max(self.origin_z - FLOOR_DROP, self.grasp_z - 0.03)
                    self.hold_run = 0
                    logger.warning('scripted policy: grip relaxed during lift (grip=%.2f) — '
                                   'nothing held, retry %d at z=%.3f',
                                   self._grip_of(observation), self.retries, self.grasp_z)
                    self.state = 'descend'
                    return self._descend(observation, ee, steps)
                logger.warning('scripted policy: grip relaxed during lift with no retries '
                               'left (grip=%.2f), giving up', self._grip_of(observation))
                return self._chunk(ee, ee, 0.0, steps, 'give_up')
            target = self._pose_target(np.array([*self.grasp_xy, z_target]))
            return self._chunk(ee, self._clamped(ee, target, MAX_Z_STEP), 1.0, steps, None)
        if self.want_place:
            self.state = 'carry'
            return self._carry(observation, ee, steps)
        self.hold_run = 0
        self.state = 'hold_after_task'
        return self._hold_after_task(observation, ee, steps)

    def _hold_after_task(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """Hold the finished pose a while: the sim's judge may end the episode on success
        (the session then closes mid-hold); otherwise we end 'complete' ourselves."""
        self.hold_run += 1
        if self.hold_run < HOLD_AFTER_TASK_BUDGET:
            return self._chunk(ee, ee, self._grip_of(observation), steps, None)
        return self._chunk(ee, ee, self._grip_of(observation), steps, 'complete')

    def _place_plan(self) -> tuple[np.ndarray | None, float | None]:
        """Drop geometry from the parsed relation: (drop_xy, place_z), or (None, None) legacy.

        Blind priors, all honestly documented: containers are reached at the legacy carry
        offset, the table height is the descend floor, screen left/right maps to ±y (the
        world axis is unresolvable without camera extrinsics), and into-container targets
        release above the mouth instead of descending into the vessel.
        """
        target = self.place_target or ''
        carry_xy = self.grasp_xy + np.array(CARRY_OFFSET)
        table_z = self.origin_z - FLOOR_DROP
        if target in ('left', 'right') or target.startswith(('left-of:', 'right-of:')):
            sign = 1.0 if target.startswith(('left', 'left-of:')) else -1.0
            return self.grasp_xy + np.array([0.0, PLACE_LATERAL * sign]), table_z
        if target.startswith('from-to:'):
            parts = target.split(':')
            to_noun = parts[2] if len(parts) > 2 else ''
            if to_noun in CONTAINERS:
                return carry_xy, max(self.grasp_z, self.origin_z - BIN_RELEASE_DROP)
            if to_noun in SURFACES:
                return carry_xy, table_z
            return carry_xy, table_z
        if target.startswith('into-'):
            noun = target[5:]
            if noun in CONTAINERS:
                return carry_xy, max(self.grasp_z, self.origin_z - BIN_RELEASE_DROP)
            return carry_xy, table_z  # 'into' an unknown noun: table-height best effort
        if target.startswith('onto-'):
            noun = target[5:]
            z = table_z + (PLATE_HEIGHT if noun in ('plate', 'pot', 'pan', 'tray') else 0.0)
            return carry_xy, z
        if target.startswith('beside:'):
            return carry_xy, table_z
        return None, None

    def _carry(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        if self.drop_xy is None:
            drop_xy, place_z = self._place_plan()
            if drop_xy is None:  # no parsed relation: legacy blind offset
                drop_xy, place_z = self.grasp_xy + np.array(CARRY_OFFSET), None
            self.drop_xy = np.clip(drop_xy, self.origin_xy - 0.6, self.origin_xy + 0.6)
            self.place_z = place_z
            logger.info('scripted policy: place plan target=%r drop_xy=%s place_z=%s',
                        self.place_target, self.drop_xy, self.place_z)
        if float(np.linalg.norm(ee[:2] - self.drop_xy)) > 0.02:
            target = self._pose_target(np.array([*self.drop_xy, ee[2]]))
            return self._chunk(ee, self._clamped(ee, target, MAX_XY_STEP), 1.0, steps, None)
        self.hold_run = 0
        self.state = 'place'
        return self._place(observation, ee, steps)

    def _place(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        if self.place_z is not None:
            # Relation-planned height; release above it (into-container never descends).
            z = min(self.place_z, ee[2])
        else:
            z = max(self.origin_z - FLOOR_DROP, ee[2] - PLACE_DEPTH)  # legacy blind descend
        if ee[2] > z + 0.01:
            target = self._pose_target(np.array([*ee[:2], z]))
            return self._chunk(ee, self._clamped(ee, target, MAX_Z_STEP), 1.0, steps, None)
        self.hold_run = 0
        self.state = 'release'
        return self._release(observation, ee, steps)

    def _release(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        self.hold_run += 1
        if self.hold_run < RELEASE_HOLD_CHUNKS:
            return self._chunk(ee, ee, 0.0, steps, None)
        self.hold_run = 0
        self.state = 'retract'
        return self._retract(observation, ee, steps)

    def _retract(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        z = min(self.origin_z, ee[2] + RETRACT_HEIGHT)
        if ee[2] < z - 0.01:
            target = self._pose_target(np.array([*ee[:2], z]))
            return self._chunk(ee, self._clamped(ee, target, MAX_Z_STEP), 0.0, steps, None)
        self.hold_run = 0
        self.state = 'hold_after_task'
        return self._hold_after_task(observation, ee, steps)

    def _done(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        return self._chunk(ee, ee, self._grip_of(observation), steps, 'complete')

    # -- observation access -----------------------------------------------

    @staticmethod
    def _ee_of(observation: dict[str, Any]) -> np.ndarray:
        value = np.asarray(observation[EE_POSE], dtype=np.float64).reshape(-1)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f'robot_state.ee_pose must be finite float32(7), got {value.shape}')
        return value

    @staticmethod
    def _grip_of(observation: dict[str, Any]) -> float:
        return float(np.asarray(observation[GRIP], dtype=np.float64).reshape(-1)[0])

    def _air(self) -> float:
        """The measured free-air closure level (1.0 until first measured)."""
        return float(self.grip_air) if self.grip_air is not None else 1.0


def build_client_stack(*, fps: float, horizon_sec: float | None, compress_images: bool,
                       jpeg_quality: int, inference_lead: float) -> dict[str, Any]:
    """The client pipeline description sent in the handshake ``meta.local_stack``.

    The platform client (positronic.policy.spec.from_spec) instantiates these components
    by name; all of them are in its registry (positronic/policy/spec.py:60-90). Structure
    mirrors the GR00T vendor recipe (positronic/vendors/gr00t/serving/recipe.py:122-161)
    and the docs example (docs/connect-your-model.md:319-332):

    - ``end_on_field`` outermost ends the run when an element with ``end`` plays;
    - ``trajectory_player`` + ``inference_schedule`` own the cadence and chunk playback;
    - the parallel block selects the guaranteed observation keys on the way in, and on
      the way out decodes each native element into a CartesianPosition command + grip,
      and lifts the ``end`` decision into the reserved field;
    - ``encode_images`` JPEG-compresses the selected images just before upload (the
      platform forbids uncompressed images; for v3 the client does not compress on its
      own — positronic/policy/remote.py:246 — so this component is mandatory).
    """
    parts: list[dict[str, Any]] = [
        component('end_on_field'),
        component('trajectory_player', replan_on_error=True),
        component('inference_schedule', fps=fps,
                  request_timing={'reference': 'before_end'},
                  merge={'start': 'after_previous', 'horizon_sec': horizon_sec},
                  latency_estimate=inference_lead),
        parallel(
            component('select_observation_fields',
                      mapping={name: name for name in (EE_POSE, STATUS, GRIP, WRIST_IMAGE, TASK, DESCRIPTOR)}),
            component('absolute_position_action', tgt_ee_pose_key=EE_POSE, tgt_grip_key=GRIP,
                      rotation_rep='quat'),
            component('end_command', field='end'),
        ),
    ]
    if compress_images:
        parts.append(component('encode_images', quality=jpeg_quality))
    return sequence(*parts)


# ---------------------------------------------------------------------------
# Fingerprint-only wire observer (kit pattern: golem experiments/franka_golem_glass/
# wire_glass.py:16-103, kit #3 of 2026-11-21). Records what the client actually sent
# and what this session answered WITHOUT raw pixels and without any credential:
# images become {shape, dtype, sha256-of-bytes}, poses/commands become plain numbers.
# Rows carry schema golem.franka.v3_observer.v1, at most WIRE_TRACE_CAP calls per
# session, status successor_observed / no_successor_observed, and always
# execution_acknowledged=False / motor_authority=False — a receipt never certifies
# that the platform executed a command (ADR-256 law 1: receipts are observations).
# Deviation from the kit, stated: wire_glass.py disables the whole trace on the first
# non-JSON-safe number (ValueError path); here non-finite numbers are recorded as
# their marker strings ('nan'/'inf') so the receipt survives the exact anomalies a
# robolab diagnosis needs, while receipt_id stays verifiable (digest of canonical
# JSON, allow_nan=False).
# ---------------------------------------------------------------------------

WIRE_TRACE_SCHEMA = 'golem.franka.v3_observer.v1'
WIRE_TRACE_ENV_VARS = ('POLICY_WIRE_TRACE_DIR', 'FRANKA_GLASS_TRACE_DIR')
WIRE_TRACE_CAP = 256
_WIRE_TRACE_ERRORS = (OSError, ValueError, TypeError, KeyError)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _digest(value: Any) -> str:
    return 'sha256:' + hashlib.sha256(_canonical_json(value)).hexdigest()


def _jsonable(value: Any) -> Any:
    """JSON-safe copy that never hides a non-finite number (marker strings instead)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _fingerprint_observation(observation: Any) -> dict[str, Any]:
    """The observation as fingerprints only: shapes+hashes for frames, numbers for poses."""
    view: dict[str, Any] = {}
    if not isinstance(observation, dict):
        view['observation_type'] = type(observation).__name__
        return view
    for key in sorted(name for name in observation if name.startswith('image.')):
        frame = np.asarray(observation[key])
        view[key] = {'shape': list(frame.shape), 'dtype': str(frame.dtype),
                     'sha256': hashlib.sha256(frame.tobytes()).hexdigest()}
    for key in (EE_POSE, STATUS, GRIP):
        if key in observation:
            view[key] = _jsonable(np.asarray(observation[key]).tolist())
    for key in (TASK, DESCRIPTOR):
        value = observation.get(key)
        if isinstance(value, str):
            view[key] = value[:200]
        elif value is not None:
            view[key] = repr(value)[:200]
    return view


class WireObserver:
    """One served session's wire receipt; every method fails silent (never raises)."""

    def __init__(self, file: Any = None, session: str | None = None):
        self.file = file
        self.session = session
        self._current: dict[str, Any] | None = None
        self._pending: dict[str, Any] | None = None
        self._calls = 0

    @classmethod
    def optional(cls) -> 'WireObserver':
        directory = next((os.environ[name] for name in WIRE_TRACE_ENV_VARS
                          if os.environ.get(name)), None)
        if not directory:
            return cls()
        try:
            dest = Path(directory)
            dest.mkdir(parents=True, exist_ok=True)
            session = uuid.uuid4().hex
            return cls((dest / f'{session}.jsonl').open('x', encoding='utf-8'), session)
        except OSError as exc:
            logger.warning('wire trace unavailable (%s)', type(exc).__name__)
        return cls()

    def _disable(self, exc: Exception) -> None:
        logger.warning('wire trace stopped (%s)', type(exc).__name__)
        if self.file is not None:
            try:
                self.file.close()
            except OSError:
                pass
        self.file = None

    def _finish(self, after: dict[str, Any] | None) -> None:
        if self._pending is None:
            return
        row: dict[str, Any] = {
            'schema': WIRE_TRACE_SCHEMA, 'session': self.session, **self._pending,
            'after': after,
            'status': 'successor_observed' if after is not None else 'no_successor_observed',
            'execution_acknowledged': False, 'motor_authority': False,
        }
        row['receipt_id'] = _digest(row)
        self.file.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
        self.file.flush()
        self._pending = None

    def see(self, observation: Any) -> None:
        """Fingerprint the observation that just ARRIVED; close the previous call's row."""
        if self.file is None:
            return
        try:
            self._current = _fingerprint_observation(observation)
            self._finish(self._current)
            if self._calls >= WIRE_TRACE_CAP:
                self.close()
        except _WIRE_TRACE_ERRORS as exc:
            self._disable(exc)

    def emitted(self, chunk: Any) -> None:
        """Pair the chunk this session is about to RETURN with the observation it saw."""
        if self.file is None or self._current is None:
            return
        try:
            commands = [{'action': _jsonable(np.asarray(item['action']).tolist()),
                         'end': item.get('end')} for item in chunk]
            self._calls += 1
            self._pending = {'call': self._calls, 'before': self._current, 'commands': commands}
            self._current = None
        except _WIRE_TRACE_ERRORS as exc:
            self._disable(exc)

    def close(self) -> None:
        """Last command without a following observation gets no_successor_observed."""
        if self.file is None:
            return
        try:
            self._finish(None)
            self.file.close()
            self.file = None
        except _WIRE_TRACE_ERRORS as exc:
            self._disable(exc)


class PickSession(Session):
    """One episode: the state machine plus the resolved chunk geometry."""

    def __init__(self, *, fps: float, horizon_sec: float | None, compress_images: bool,
                 jpeg_quality: int, inference_lead: float):
        super().__init__(build_client_stack(
            fps=fps, horizon_sec=horizon_sec, compress_images=compress_images,
            jpeg_quality=jpeg_quality, inference_lead=inference_lead,
        ), metadata={'action_fps': fps, 'action_horizon_sec': horizon_sec,
                     # Per-session image quality the client should send with (overrides
                     # the model-level default when the session requested another one).
                     'compress_images': bool(compress_images), 'jpeg_quality': int(jpeg_quality)})
        horizon = DEFAULT_HORIZON_SEC if horizon_sec is None else horizon_sec
        steps = max(1, min(64, round(fps * horizon)))
        self._fps = float(fps)
        self._steps = int(steps)
        self._machine = PickStateMachine()
        # Fingerprint-only wire trace (wire_glass.py pattern). Deliberately NOT exposed
        # in session metadata: the wire surface the platform client sees stays byte-
        # identical to the submission-6 image; the trace session id lives only in the
        # trace file name and rows.
        self._observer = WireObserver.optional()
        self._frame_logged = False

    def _log_first_frame(self, observation: Any) -> None:
        # Vision calibration (Molmo Pick 0/204 diagnostics): the first frame of EVERY
        # camera (wrist + two exterior), downscaled and base64-encoded into the policy
        # log, so the detection can finally be tuned against the platform's actual
        # renders. The randomized exterior left/right order is visible in the dump.
        # OFF by default (sponsor: the log would carry the raw JPEGs even when the
        # fingerprint trace is disabled) — enable locally with POLICY_LOG_FRAMES=1.
        if not os.environ.get('POLICY_LOG_FRAMES'):
            return
        if self._frame_logged:
            return
        self._frame_logged = True
        if not isinstance(observation, dict):
            return
        try:
            import base64
            import cv2
            for key in (WRIST_IMAGE, EXTERIOR_IMAGE, EXTERIOR2_IMAGE):
                img = observation.get(key)
                if img is None or not hasattr(img, 'shape') or len(getattr(img, 'shape', ()) or ()) != 3 or img.shape[0] < 8 or img.shape[1] < 8:
                    logger.warning('policy: %s frame unusable for calibration (shape %s)', key, getattr(img, 'shape', None))
                    continue
                h, w = img.shape[:2]
                scale = min(1.0, 320.0 / max(h, 1))
                small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1.0 else img
                ok, buf = cv2.imencode('.jpg', small[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    logger.warning('policy: FRAME_B64 %s %s', key, base64.b64encode(buf.tobytes()).decode('ascii'))
        except Exception as exc:
            logger.warning('policy: frame logging skipped (%s: %.60s)', type(exc).__name__, exc)

    def infer(self, observation: Any) -> Any:
        # The observer fingerprints the observation EXACTLY as it arrived, before any
        # tolerance substitution, so a malformed client message is receipted verbatim.
        self._observer.see(observation)
        # FIX-1: an inference error response is something the client "could not use"
        # (the platform's policy_inference_crash class) — never raise; missing keys fall
        # through to the machine's sanitiser and safe-hold path, which serve valid chunks.
        if not isinstance(observation, dict):
            logger.warning('policy: observation is %s instead of a dict, serving a safe hold',
                           type(observation).__name__)
            observation = {}
        missing = [key for key in (EE_POSE, GRIP, WRIST_IMAGE, TASK) if key not in observation]
        if missing:
            logger.warning('policy: observation is missing %s; serving on the tolerant path', missing)
        self._log_first_frame(observation)
        chunk = self._machine.step(observation, self._fps, self._steps)
        self._observer.emitted(chunk)
        return chunk

    def close(self) -> None:
        self._machine = None
        self._observer.close()


class PickAndPlaceModel(Model):
    """The scripted policy factory; parameters are the session names the client may set."""

    def __init__(self, *, fps: float = DEFAULT_FPS, horizon_sec: float | None = DEFAULT_HORIZON_SEC,
                 compress_images: bool = True, jpeg_quality: int = DEFAULT_JPEG_QUALITY):
        super().__init__(
            parameters={
                PARAM_FPS: float(fps),
                PARAM_HORIZON_SEC: None if horizon_sec is None else float(horizon_sec),
                PARAM_COMPRESS_IMAGES: bool(compress_images),
                PARAM_JPEG_QUALITY: int(jpeg_quality),
                PARAM_INFERENCE_LEAD: 0.0,
                PARAM_RTC: None,
            },
            metadata={
                'type': 'scripted_pick',
                'checkpoint_id': 'scripted-pick-v1',
                # The v3 client reads this top-level meta key to record the JPEG quality
                # it should send with (positronic/policy/remote.py:231, 239-240).
                'compress_images': bool(compress_images),
                'jpeg_quality': int(jpeg_quality),
                # RoboLab harness reads ee_frames from the session metadata (v3):
                # poses are stated at the rig's gripper site (roboarm models.py:25
                # GRIPPER_SITE = 'gripper'), zero offset.
                'ee_frames': {
                    'robot_state.ee_pose': {'site': 'gripper', 'offset': [0.0, 0.0, 0.0]},
                    'robot_command.pose': {'site': 'gripper', 'offset': [0.0, 0.0, 0.0]},
                },
            },
        )

    def meta(self) -> dict[str, Any]:
        # Declare the frames of every pose signal this policy reads or commands.
        # RoboLab's harness compares the policy's ee_frames with the rig's at session
        # open, BEFORE the first observation (roboarm/README.md:24-29): a policy that
        # states none "speaks joints only, and it observes no pose signal", so its
        # cartesian answers are unusable to the client — the exact signature of the
        # robolab policy_inference_crash of submissions 6482ea4f/b4d91d1a/72c9c793/42e513aa.
        # Both signals sit at the rig's end_effector site, zero offset.
        return {
            'ee_frames': {
                'robot_state.ee_pose': {'site': 'gripper', 'offset': [0.0, 0.0, 0.0]},
                'robot_command.pose': {'site': 'gripper', 'offset': [0.0, 0.0, 0.0]},
            },
        }

    def prepare_session(self, params: dict[str, Any]) -> PickSession:
        """Build a session from ANY requested parameter set — never a refusal (FIX-1).

        The RoboLab client's query parameters are not observable from this image (its
        code is not in the reference clone), so every session must be served: unknown
        names are ignored upstream (policy_server patches the wire resolution), and any
        value that is missing, of the wrong type or out of range falls back to the
        declared default with a log line instead of raising.
        """
        requested = dict(params or {})
        ignored = requested.keys() - {PARAM_FPS, PARAM_HORIZON_SEC, PARAM_COMPRESS_IMAGES,
                                      PARAM_JPEG_QUALITY, PARAM_INFERENCE_LEAD, PARAM_RTC}
        if ignored:
            logger.info('session parameters %s are not declared; ignoring', sorted(ignored))

        def number(name: str, default: float, *, positive: bool = False) -> float:
            value = requested.get(name, default)
            # bool is an int subclass; a JSON boolean is not a number here
            if type(value) is bool or type(value) not in (int, float) \
                    or not math.isfinite(value) or (positive and value <= 0):
                logger.info('session parameter %s=%r is not a usable %s; using default %s',
                            name, value, 'positive number' if positive else 'number', default)
                return float(default)
            return float(value)

        fps = number(PARAM_FPS, DEFAULT_FPS, positive=True)
        horizon = requested.get(PARAM_HORIZON_SEC)
        if horizon is not None:
            horizon = number(PARAM_HORIZON_SEC, DEFAULT_HORIZON_SEC)
        lead = number(PARAM_INFERENCE_LEAD, 0.0)
        if requested.get(PARAM_RTC) is not None:
            # This policy has no RTC component in its client stack; the declared fixed
            # cadence is what the client will get — proceed instead of refusing.
            logger.info('session parameter rtc=%r requested; this policy has no RTC component, '
                        'proceeding on the declared fixed cadence', requested[PARAM_RTC])
        compress = requested.get(PARAM_COMPRESS_IMAGES)
        if type(compress) is not bool:
            logger.info('session parameter compress_images=%r is not a boolean; using default True', compress)
            compress = True
        quality = requested.get(PARAM_JPEG_QUALITY, DEFAULT_JPEG_QUALITY)
        if type(quality) is not int or not 0 <= quality <= 100:
            logger.info('session parameter jpeg_quality=%r is not an int in [0, 100]; using default %s',
                        quality, DEFAULT_JPEG_QUALITY)
            quality = int(DEFAULT_JPEG_QUALITY)
        return PickSession(fps=fps, horizon_sec=horizon, compress_images=compress,
                           jpeg_quality=quality, inference_lead=lead)


def warmup_observation() -> dict[str, Any]:
    """A synthetic observation with the exact keys the client stack selects."""
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    image[28:44, 40:56] = (200, 30, 30)  # a red square for the detector to find
    return {
        EE_POSE: np.array([0.4, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        STATUS: 0,
        GRIP: 0.0,
        WRIST_IMAGE: image,
        TASK: 'pick up the red block',
        DESCRIPTOR: 'mujoco.franka',
    }


def warmup(model: PickAndPlaceModel) -> None:
    """Run one throwaway inference so the first real call pays no cold-path cost."""
    session = model.prepare_session(model.parameters)
    try:
        session.infer(warmup_observation())
    finally:
        session.close()


def build_model() -> PickAndPlaceModel:
    model = PickAndPlaceModel()
    warmup(model)
    return model
