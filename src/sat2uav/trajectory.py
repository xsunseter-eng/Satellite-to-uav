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


def sample_valid_flight(
    scene: SceneBundle,
    cfg: GenerationConfig,
    rng: np.random.Generator,
    video_index: int,
    max_attempts: int = 250,
) -> tuple[VideoSampleParams, list[FramePose]]:
    """Sample a randomized flight starting within X km of origin whose path and
    oblique camera look-ahead stay safely inside the loaded scene boundaries.
    """
    margin = min(
        cfg.flight.boundary_margin_m,
        0.18 * min(scene.x_max_m - scene.x_min_m, scene.y_max_m - scene.y_min_m),
    )
    safe_xmin = scene.x_min_m + margin
    safe_xmax = scene.x_max_m - margin
    safe_ymin = scene.y_min_m + margin
    safe_ymax = scene.y_max_m - margin

    max_scene_r = 0.45 * min(safe_xmax - safe_xmin, safe_ymax - safe_ymin)
    max_offset_m = min(cfg.flight.max_start_offset_km * 1000.0, max(50.0, max_scene_r))

    best_choice: tuple[float, float, float, float, float, float] | None = None

    for _ in range(max_attempts):
        # Uniform disk sampling around scene origin (sqrt(u) for area uniformity)
        r = math.sqrt(float(rng.random())) * max_offset_m
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        sx = scene.origin_x_m + r * math.cos(theta)
        sy = scene.origin_y_m + r * math.sin(theta)

        yaw_deg = float(rng.uniform(*cfg.flight.yaw_deg_range)) % 360.0
        speed_mps = float(rng.uniform(*cfg.flight.speed_mps_range))
        altitude_agl_m = float(rng.uniform(*cfg.flight.altitude_agl_m_range))

        yaw_rad = math.radians(yaw_deg)
        vx = speed_mps * math.sin(yaw_rad)  # +East
        vy = speed_mps * math.cos(yaw_rad)  # +North

        ex = sx + vx * cfg.duration_sec
        ey = sy + vy * cfg.duration_sec

        # Also check modest forward look-ahead so oblique pitch stays on texture
        look_ahead = altitude_agl_m * 0.65
        lx = ex + look_ahead * math.sin(yaw_rad)
        ly = ey + look_ahead * math.cos(yaw_rad)

        if (
            safe_xmin <= sx <= safe_xmax
            and safe_ymin <= sy <= safe_ymax
            and safe_xmin <= ex <= safe_xmax
            and safe_ymin <= ey <= safe_ymax
            and safe_xmin <= lx <= safe_xmax
            and safe_ymin <= ly <= safe_ymax
        ):
            best_choice = (sx, sy, r, yaw_deg, speed_mps, altitude_agl_m)
            break

    if best_choice is None:
        # Fallback: start near origin and fly inward along the safest direction
        sx, sy = scene.origin_x_m, scene.origin_y_m
        r = 0.0
        yaw_deg = float(rng.uniform(*cfg.flight.yaw_deg_range)) % 360.0
        max_dist = 0.4 * min(safe_xmax - safe_xmin, safe_ymax - safe_ymin)
        speed_mps = min(
            float(rng.uniform(*cfg.flight.speed_mps_range)),
            max(2.0, max_dist / max(cfg.duration_sec, 0.5)),
        )
        altitude_agl_m = float(rng.uniform(*cfg.flight.altitude_agl_m_range))
        best_choice = (sx, sy, r, yaw_deg, speed_mps, altitude_agl_m)

    sx, sy, offset_m, yaw_deg, speed_mps, altitude_agl_m = best_choice
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
