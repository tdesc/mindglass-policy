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

import logging
import math
from typing import Any

import cv2
import numpy as np
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, check_int_range
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
TASK = 'task'  # str
DESCRIPTOR = 'descriptor'  # str, e.g. 'mujoco.franka'

# Session parameter names (same names the GR00T vendor server declares — the platform
# may send them as query params; unknown params are refused by the wire, so we accept
# the union of plausible ones and ignore what we do not use).
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
EPISODE_BUDGET = 100  # chunks total, safety net against a livelock

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


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    """Normalize a wxyz quaternion; fall back to identity when it is degenerate."""
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


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
        # Servo bookkeeping.
        self.gain_sign = np.array([1.0, 1.0])  # per-axis sign of image-offset -> metres
        self.flips = np.zeros(2, dtype=int)
        self.prev_offset = np.zeros(2)
        self.prev_xy: np.ndarray | None = None  # EE xy at the previous servo chunk
        self.centered_run = 0
        self.search_count = 0
        self.spiral_i = 0
        # Grasp/place bookkeeping.
        self.grasp_xy = np.zeros(2)
        self.grasp_z = 0.0
        self.grasp_area = 0.0
        self.hold_run = 0
        self.retries = 0
        self.error_run = 0
        self.chunk_count = 0
        self.want_place = False
        self.color: str | None = None
        # Task-grammar intent (task_parser.parse_task_text): skill/target for place and
        # the drawer bookkeeping below.
        self.place_target: str | None = None
        self.drop_xy: np.ndarray | None = None
        self.place_z: float | None = None
        self.drawer_mode: str | None = None  # 'open' | 'close'
        self.drawer_phase = ''  # 'lateral' | 'vertical' inside drawer_approach
        self.handle_xy = np.zeros(2)
        self.handle_z = 0.0
        self.pull_dir = np.zeros(2)
        self.pull_start = np.zeros(2)
        self.drawer_regrasps = 0
        self.drawer_steps = 0
        self.safety_fallbacks = 0
        self.last_good_ee = None
        self.stall_run = 0
        self.prev_drawer_xy: np.ndarray | None = None

    # -- helpers ----------------------------------------------------------

    def _clamped(self, current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        """Move ``target`` onto the per-chunk step budget around ``current``."""
        delta = target - current
        norm = float(np.linalg.norm(delta))
        if norm <= max_step:
            return target
        return current + delta * (max_step / norm)

    def _pose_target(self, xyz: np.ndarray) -> np.ndarray:
        return np.concatenate([xyz, self.rotation]).astype(np.float64)

    def _chunk(self, ee: np.ndarray, target: np.ndarray, grip: float, steps: int, end: str | None) -> list[dict[str, Any]]:
        """Interpolate EE translation from ``ee`` to ``target`` across the chunk; rotation is constant."""
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
            return self._chunk(ee, ee, self._grip_of(observation), steps, end)

    def _safe_ee(self, observation: dict[str, Any]) -> np.ndarray:
        # last good pose, else the deterministic home pose
        if getattr(self, 'last_good_ee', None) is not None:
            return self.last_good_ee
        return np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])

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
            self.want_place = False
            self.place_target = intent['target']
            self.handle_xy = self.origin_xy + np.array([HANDLE_FORWARD, 0.0])
            self.handle_z = self.origin_z - HANDLE_HEIGHT_DROP
            self.drawer_phase = 'lateral'
            self.state = 'drawer_approach'
            logger.info('scripted policy: drawer skill mode=%s handle=%s z=%.3f',
                        self.drawer_mode, self.handle_xy, self.handle_z)
            return self._drawer_approach(observation, ee, steps)
        self.want_place = skill in (SKILL_PLACE, SKILL_MOVE_DIR)
        self.place_target = intent['target']
        self.state = 'servo'
        return self._servo(observation, ee, steps)

    def _servo(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """Hover at the cruise height and centre the object in the wrist image."""
        blob = detect_blob(observation[WRIST_IMAGE], self.color)
        if blob is None:
            self.centered_run = 0
            self.search_count += 1
            if self.search_count > SEARCH_BUDGET:
                logger.warning('scripted policy: no object detected in %d chunks, giving up', SEARCH_BUDGET)
                return self._chunk(ee, ee, self._grip_of(observation), steps, 'give_up')
            # Growing spiral around the start position; keep the cruise height.
            radius = min(SPIRAL_MAX_RADIUS, SPIRAL_STEP * self.spiral_i)
            angle = 2.39996 * self.spiral_i  # golden-angle spacing
            self.spiral_i += 1
            offset_xy = self.origin_xy + radius * np.array([math.cos(angle), math.sin(angle)])
            xy = self._clamped(ee[:2], offset_xy, MAX_XY_STEP)
            target = self._pose_target(np.array([*xy, self.origin_z]))
            return self._chunk(ee, target, 0.0, steps, None)

        u, v, area = blob
        offset = np.array([u, v])
        # Sign auto-calibration: after a real horizontal move, an axis whose error grew
        # had the wrong sign. Bounded flips, then servo blindly with what we have.
        if self.prev_xy is not None and float(np.linalg.norm(ee[:2] - self.prev_xy)) > 0.03:
            grew = np.abs(offset) > np.abs(self.prev_offset) + 0.02
            for axis in (0, 1):
                if grew[axis] and self.flips[axis] < 2:
                    self.gain_sign[axis] *= -1.0
                    self.flips[axis] += 1
                    logger.info('scripted policy: flipped servo gain sign on axis %d (flip %d)', axis, self.flips[axis])
        self.prev_offset = offset
        self.prev_xy = ee[:2].copy()
        if np.abs(offset).max() < CENTER_TOL:
            self.centered_run += 1
        else:
            self.centered_run = 0
        if self.centered_run >= CONVERGE_CHUNKS:
            self.grasp_xy = ee[:2].copy()
            self.state = 'descend'
            return self._descend(observation, ee, steps)
        # Move toward the centroid; image v grows downward, the gain sign absorbs the
        # unknown camera-mount orientation.
        delta_xy = SERVO_GAIN * self.gain_sign * offset
        delta_xy = np.clip(delta_xy, -MAX_XY_STEP, MAX_XY_STEP)
        target_xy = ee[:2] + delta_xy
        # Keep inside a sane workspace around the episode start.
        target_xy = np.clip(target_xy, self.origin_xy - 0.6, self.origin_xy + 0.6)
        target = self._pose_target(np.array([*target_xy, self.origin_z]))
        return self._chunk(ee, self._clamped(ee, target, MAX_XY_STEP), 0.0, steps, None)

    def _descend(self, observation: dict[str, Any], ee: np.ndarray, steps: int) -> list[dict[str, Any]]:
        """Lower straight down over the centred object until grasp height."""
        blob = detect_blob(observation[WRIST_IMAGE], self.color)
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
            target_xy = ee[:2] + np.clip(0.4 * SERVO_GAIN * self.gain_sign * np.array([blob[0], blob[1]]),
                                         -0.03, 0.03)
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
        blob = detect_blob(observation[WRIST_IMAGE], self.color)
        # Robotiq fingers stop at the object: closure clearly below 1.0 means something
        # is inside. The blob staying big and centred is the second, weaker signal.
        stopped = grip_obs < 0.92
        following = blob is not None and blob[2] >= 0.5 * self.grasp_area and abs(blob[0]) < 0.5 and abs(blob[1]) < 0.5
        if stopped or following:
            logger.info('scripted policy: grasp verified (grip=%.2f stopped=%s following=%s)',
                        grip_obs, stopped, following)
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
        — vision refines, but never blocks (detect_blob_soft).
        """
        blob = detect_blob_soft(observation.get(WRIST_IMAGE), None)
        if self.drawer_phase == 'lateral' and blob is not None and blob[2] > 0.02 \
                and abs(blob[0]) < CENTER_TOL and abs(blob[1]) < CENTER_TOL:
            self.handle_xy = ee[:2].copy()
            self.drawer_phase = 'vertical'
            logger.info('scripted policy: adopting wrist blob neighbourhood as the handle')
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
        if self.hold_run < GRASP_HOLD_CHUNKS:
            target = self._pose_target(np.array([*self.handle_xy, self.handle_z]))
            return self._chunk(ee, self._clamped(ee, target, MAX_Z_STEP), 1.0, steps, None)
        grip_obs = self._grip_of(observation)
        if grip_obs < 0.92:  # fingers stopped: the handle is inside
            # Drawer axis assumed along the start->handle line: pull toward the robot to
            # open, push away to close (the same blind +X forward guess as CARRY_OFFSET).
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
        if self.drawer_regrasps < DRAWER_REGRASP:
            self.drawer_regrasps += 1
            self.hold_run = 0
            logger.info('scripted policy: drawer grasp failed (grip=%.2f), re-grasping', grip_obs)
            return self._chunk(ee, ee, 0.0, steps, None)  # open and try closing again
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
        if grip_obs > 0.95 and self.drawer_steps > 1:
            if self.drawer_regrasps < DRAWER_REGRASP:
                self.drawer_regrasps += 1
                self.hold_run = 0
                self.state = 'drawer_grasp'
                logger.info('scripted policy: lost the drawer handle (grip=%.2f), re-grasping', grip_obs)
                return self._drawer_grasp(observation, ee, steps)
            logger.warning('scripted policy: lost the drawer handle with no retries left, giving up')
            return self._chunk(ee, ee, 1.0, steps, 'give_up')
        target_xy = ee[:2] + self.pull_dir * DRAWER_STEP
        target = self._pose_target(np.array([*target_xy, self.handle_z]))
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
                logger.warning('scripted policy: drawer never moved; open failed, giving up')
                return self._chunk(ee, ee, 1.0, steps, 'give_up')
            self.hold_run = 0
            self.state = 'hold_after_task'  # hold the final pose for the judge, then complete
            return self._hold_after_task(observation, ee, steps)
        return chunk

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


class PickSession(Session):
    """One episode: the state machine plus the resolved chunk geometry."""

    def __init__(self, *, fps: float, horizon_sec: float | None, compress_images: bool,
                 jpeg_quality: int, inference_lead: float):
        super().__init__(build_client_stack(
            fps=fps, horizon_sec=horizon_sec, compress_images=compress_images,
            jpeg_quality=jpeg_quality, inference_lead=inference_lead,
        ), metadata={'action_fps': fps, 'action_horizon_sec': horizon_sec})
        horizon = DEFAULT_HORIZON_SEC if horizon_sec is None else horizon_sec
        steps = max(1, min(64, round(fps * horizon)))
        self._fps = float(fps)
        self._steps = int(steps)
        self._machine = PickStateMachine()

    def infer(self, observation: Any) -> Any:
        if not isinstance(observation, dict):
            raise ValueError(f'Expected an observation dict, got {type(observation).__name__}')
        for key in (EE_POSE, GRIP, WRIST_IMAGE, TASK):
            if key not in observation:
                raise ValueError(f'Observation is missing the required key {key!r}')
        return self._machine.step(observation, self._fps, self._steps)

    def close(self) -> None:
        self._machine = None


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
            },
        )

    def prepare_session(self, params: dict[str, Any]) -> PickSession:
        fps, horizon = params[PARAM_FPS], params[PARAM_HORIZON_SEC]
        lead = params[PARAM_INFERENCE_LEAD]
        for name, value in ((PARAM_FPS, fps), *(tuple() if horizon is None else ((PARAM_HORIZON_SEC, horizon),)),
                            (PARAM_INFERENCE_LEAD, lead)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be a finite non-negative number')
        if fps <= 0:
            raise ValueError('fps must be positive')
        if params[PARAM_RTC] is not None:
            raise ValueError('This policy does not support RTC; leave rtc unset')
        compress = params[PARAM_COMPRESS_IMAGES]
        if type(compress) is not bool:
            raise ValueError('compress_images must be a boolean')
        check_int_range(params[PARAM_JPEG_QUALITY], 0, 100, PARAM_JPEG_QUALITY)
        return PickSession(fps=float(fps), horizon_sec=horizon, compress_images=compress,
                           jpeg_quality=int(params[PARAM_JPEG_QUALITY]),
                           inference_lead=float(lead))


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
