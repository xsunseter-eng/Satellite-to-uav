"""Origin-centered randomized UAV trajectory generator with 6-DoF telemetry."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import GenerationConfig, VideoSampleParams, sample_video_params
from .scene import SceneBundle


@dataclass
class FramePose:
    """Per-frame 6-DoF UAV camera telemetry."""

    frame_idx: int
    time_sec: float
    x_m: float
    y_m: float
    z_msl_m: float
    z_agl_m: float
    ground_z_m: float
    lat: float
    lon: float
    yaw_deg: float
    pitch_deg: float
    speed_mps: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _frustum_corners_inside(
    x: float,
    y: float,
    altitude_agl_m: float,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    aspect: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> bool:
    """Verify that the camera ground footprint (near & far corners) stays inside scene bounds."""
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return False

    hfov_rad = math.radians(fov_deg)
    vfov_rad = 2.0 * math.atan(math.tan(0.5 * hfov_rad) / max(aspect, 0.5))
    pitch_rad = math.radians(pitch_deg)  # negative = downward

    # Top-of-frame ray elevation angle (closest to horizon)
    top_angle = pitch_rad + 0.5 * vfov_rad
    bot_angle = pitch_rad - 0.5 * vfov_rad

    # If top ray is looking at/above horizon, require large forward ground buffer (2.5 * alt)
    if top_angle >= -math.radians(8.0):
        far_dist = min(2200.0, altitude_agl_m * 4.5)
    else:
        far_dist = min(2200.0, altitude_agl_m / math.tan(-top_angle))

    if bot_angle >= -math.radians(8.0):
        near_dist = altitude_agl_m * 0.5
    else:
        near_dist = altitude_agl_m / math.tan(-bot_angle)

    yaw_rad = math.radians(yaw_deg)
    fx, fy = math.sin(yaw_rad), math.cos(yaw_rad)
    rx, ry = math.cos(yaw_rad), -math.sin(yaw_rad)

    # Extra 15% margin for radial lens distortion at corners
    tan_half_h = math.tan(0.5 * hfov_rad) * 1.15
    far_half_w = math.hypot(altitude_agl_m, far_dist) * tan_half_h
    near_half_w = math.hypot(altitude_agl_m, near_dist) * tan_half_h

    for dist, half_w in ((near_dist, near_half_w), (far_dist, far_half_w)):
        for side in (-1.0, 0.0, 1.0):
            px = x + dist * fx + side * half_w * rx
            py = y + dist * fy + side * half_w * ry
            if not (xmin <= px <= xmax and ymin <= py <= ymax):
                return False
    return True


def sample_valid_flight(
    scene: SceneBundle,
    cfg: GenerationConfig,
    rng: np.random.Generator,
    video_index: int,
    max_attempts: int = 400,
) -> tuple[VideoSampleParams, list[FramePose]]:
    """Sample a randomized flight starting within X km of origin whose path and
    full 3D oblique camera frustum stay safely inside the loaded scene boundaries.
    """
    scene_w = scene.x_max_m - scene.x_min_m
    scene_h = scene.y_max_m - scene.y_min_m
    edge_pad = min(25.0, 0.04 * min(scene_w, scene_h))
    b_xmin = scene.x_min_m + edge_pad
    b_xmax = scene.x_max_m - edge_pad
    b_ymin = scene.y_min_m + edge_pad
    b_ymax = scene.y_max_m - edge_pad

    max_scene_r = 0.45 * min(b_xmax - b_xmin, b_ymax - b_ymin)
    max_offset_m = min(cfg.flight.max_start_offset_km * 1000.0, max(20.0, max_scene_r))

    best_choice: tuple[float, float, float, float, float, float, float, float, tuple[int, int]] | None = None

    for _ in range(max_attempts):
        # Uniform disk sampling around scene origin (sqrt(u) for area uniformity)
        r = math.sqrt(float(rng.random())) * max_offset_m
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        sx = scene.origin_x_m + r * math.cos(theta)
        sy = scene.origin_y_m + r * math.sin(theta)

        yaw_deg = float(rng.uniform(*cfg.flight.yaw_deg_range)) % 360.0
        speed_mps = float(rng.uniform(*cfg.flight.speed_mps_range))
        altitude_agl_m = float(rng.uniform(*cfg.flight.altitude_agl_m_range))
        fov_deg = float(rng.uniform(*cfg.camera.fov_deg_range))
        pitch_deg = float(rng.uniform(*cfg.camera.pitch_deg_range))
        res = cfg.camera.resolutions[int(rng.integers(0, len(cfg.camera.resolutions)))]
        aspect = res[0] / float(res[1])

        yaw_rad = math.radians(yaw_deg)
        ex = sx + speed_mps * math.sin(yaw_rad) * cfg.duration_sec
        ey = sy + speed_mps * math.cos(yaw_rad) * cfg.duration_sec

        if _frustum_corners_inside(
            sx, sy, altitude_agl_m, yaw_deg, pitch_deg, fov_deg, aspect,
            b_xmin, b_xmax, b_ymin, b_ymax,
        ) and _frustum_corners_inside(
            ex, ey, altitude_agl_m, yaw_deg, pitch_deg, fov_deg, aspect,
            b_xmin, b_xmax, b_ymin, b_ymax,
        ):
            best_choice = (sx, sy, r, yaw_deg, speed_mps, altitude_agl_m, fov_deg, pitch_deg, res)
            break

    if best_choice is None:
        # Fallback for compact scenes: center near origin with steeper pitch to stay strictly in-bounds
        sx, sy = scene.origin_x_m, scene.origin_y_m
        r = 0.0
        yaw_deg = float(rng.uniform(*cfg.flight.yaw_deg_range)) % 360.0
        res = cfg.camera.resolutions[int(rng.integers(0, len(cfg.camera.resolutions)))]
        fov_deg = min(float(rng.uniform(*cfg.camera.fov_deg_range)), 68.0)
        pitch_deg = min(float(rng.uniform(*cfg.camera.pitch_deg_range)), -62.0)
        max_span = min(scene_w, scene_h)
        altitude_agl_m = min(float(rng.uniform(*cfg.flight.altitude_agl_m_range)), max_span * 0.22)
        speed_mps = min(
            float(rng.uniform(*cfg.flight.speed_mps_range)),
            max(2.0, (max_span * 0.18) / max(cfg.duration_sec, 0.5)),
        )
        best_choice = (sx, sy, r, yaw_deg, speed_mps, altitude_agl_m, fov_deg, pitch_deg, res)

    sx, sy, offset_m, yaw_deg, speed_mps, altitude_agl_m, fov_deg, pitch_deg, res = best_choice
    params = sample_video_params(
        cfg=cfg,
        rng=rng,
        video_index=video_index,
        start_x_m=sx,
        start_y_m=sy,
        start_offset_from_origin_m=offset_m,
        yaw_deg=yaw_deg,
        speed_mps=speed_mps,
        altitude_agl_m=altitude_agl_m,
    )
    # Apply the exact frustum-verified camera geometry
    params.fov_deg = float(fov_deg)
    params.pitch_deg = float(pitch_deg)
    params.width, params.height = int(res[0]), int(res[1])

    num_frames = max(1, int(round(params.duration_sec * params.fps)))
    yaw_rad = math.radians(params.yaw_deg)
    vx = params.speed_mps * math.sin(yaw_rad)
    vy = params.speed_mps * math.cos(yaw_rad)

    # Compute constant MSL cruise altitude from start ground elevation + AGL
    start_ground_z = scene.sample_elevation(sx, sy)
    cruise_z_msl = start_ground_z + params.altitude_agl_m

    poses: list[FramePose] = []
    for i in range(num_frames):
        t = i / float(params.fps)
        x = sx + vx * t
        y = sy + vy * t
        gz = scene.sample_elevation(x, y)
        # Ensure minimum 25 m terrain clearance over steep ridges/canyons
        z_msl = max(cruise_z_msl, gz + 25.0)
        z_agl = z_msl - gz
        lat, lon = scene.to_latlon(x, y)
        poses.append(
            FramePose(
                frame_idx=i,
                time_sec=round(t, 4),
                x_m=round(x, 3),
                y_m=round(y, 3),
                z_msl_m=round(z_msl, 3),
                z_agl_m=round(z_agl, 3),
                ground_z_m=round(gz, 3),
                lat=round(lat, 7),
                lon=round(lon, 7),
                yaw_deg=round(params.yaw_deg, 3),
                pitch_deg=round(params.pitch_deg, 3),
                speed_mps=round(params.speed_mps, 3),
            )
        )

    return params, poses
