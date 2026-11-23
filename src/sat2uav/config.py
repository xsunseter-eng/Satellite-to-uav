"""Configuration schemas and loaders for Satellite-to-UAV video generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class FlightConfig:
    max_start_offset_km: float = 1.0
    yaw_deg_range: tuple[float, float] = (0.0, 360.0)
    speed_mps_range: tuple[float, float] = (12.0, 32.0)
    altitude_agl_m_range: tuple[float, float] = (90.0, 300.0)
    boundary_margin_m: float = 120.0


@dataclass
class LensDistortionConfig:
    enabled: bool = True
    k1_range: tuple[float, float] = (-0.22, 0.12)
    k2_range: tuple[float, float] = (-0.05, 0.05)
    p1_range: tuple[float, float] = (-0.01, 0.01)
    p2_range: tuple[float, float] = (-0.01, 0.01)


@dataclass
class CameraConfig:
    resolutions: list[tuple[int, int]] = field(
        default_factory=lambda: [(854, 480), (1280, 720), (1920, 1080)]
    )
    fov_deg_range: tuple[float, float] = (50.0, 90.0)
    pitch_deg_range: tuple[float, float] = (-85.0, -28.0)
    spectral_modes: list[str] = field(
        default_factory=lambda: [
            "rgb",
            "panchromatic",
            "nir",
            "false_color_cir",
            "thermal_lwir",
            "night_vision",
        ]
    )
    lens_distortion: LensDistortionConfig = field(default_factory=LensDistortionConfig)


@dataclass
class LightingConfig:
    sun_azimuth_deg_range: tuple[float, float] = (0.0, 360.0)
    sun_elevation_deg_range: tuple[float, float] = (15.0, 75.0)
    night_probability: float = 0.20
    shadow_strength_range: tuple[float, float] = (0.25, 0.70)
    color_temp_k_range: tuple[float, float] = (3600.0, 8200.0)
    wb_tint_range: tuple[float, float] = (-0.07, 0.07)


@dataclass
class AtmosphereConfig:
    haze_enabled: bool = True
    visibility_km_range: tuple[float, float] = (1.8, 22.0)
    sun_glare_enabled: bool = True
    glare_intensity_range: tuple[float, float] = (0.15, 0.70)
    specular_glint_range: tuple[float, float] = (0.10, 0.50)


@dataclass
class SensorArtifactsConfig:
    motion_blur_enabled: bool = True
    shutter_sec_range: tuple[float, float] = (0.004, 0.025)
    rolling_shutter_strength_range: tuple[float, float] = (0.0, 0.60)
    auto_exposure_enabled: bool = True
    ae_target_luma_range: tuple[float, float] = (0.44, 0.58)
    ae_lag_alpha_range: tuple[float, float] = (0.10, 0.35)
    dr_clip_strength_range: tuple[float, float] = (0.0, 0.30)
    sensor_noise_enabled: bool = True
    iso_noise_range: tuple[float, float] = (0.005, 0.040)
    thermal_fpn_range: tuple[float, float] = (0.008, 0.035)
    lens_crack_probability: float = 0.25
    watermark_droplet_probability: float = 0.30


@dataclass
class GenerationConfig:
    num_videos: int = 4
    duration_sec: float = 5.0
    fps: int = 24
    seed: int | None = 42
    flight: FlightConfig = field(default_factory=FlightConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    sensor_and_artifacts: SensorArtifactsConfig = field(
        default_factory=SensorArtifactsConfig
    )


@dataclass
class VideoSampleParams:
    """Concrete sampled parameters for a single generated UAV video."""

    video_index: int
    seed: int
    width: int
    height: int
    fps: int
    duration_sec: float
    # Flight
    start_x_m: float
    start_y_m: float
    start_offset_from_origin_m: float
    yaw_deg: float
    speed_mps: float
    altitude_agl_m: float
    # Camera
    fov_deg: float
    pitch_deg: float
    spectral_mode: str
    k1: float
    k2: float
    p1: float
    p2: float
    # Lighting
    sun_azimuth_deg: float
    sun_elevation_deg: float
    is_night: bool
    shadow_strength: float
    color_temp_k: float
    wb_tint: float
    # Atmosphere
    haze_enabled: bool
    visibility_km: float
    sun_glare_enabled: bool
    glare_intensity: float
    specular_glint: float
    # Sensor & Artifacts
    motion_blur_enabled: bool
    shutter_sec: float
    rolling_shutter_strength: float
    auto_exposure_enabled: bool
    ae_target_luma: float
    ae_lag_alpha: float
    dr_clip_strength: float
    sensor_noise_enabled: bool
    iso_noise: float
    thermal_fpn: float
    has_lens_crack: bool
    has_watermark_droplets: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pair(val: Any) -> tuple[float, float]:
    return (float(val[0]), float(val[1]))


def load_generation_config(path: str | Path | None = None) -> GenerationConfig:
    """Load GenerationConfig from a YAML file (or return defaults if path is None)."""
    if path is None:
        return GenerationConfig()

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    f_raw = raw.get("flight", {})
    c_raw = raw.get("camera", {})
    ld_raw = c_raw.get("lens_distortion", {})
    l_raw = raw.get("lighting", {})
    a_raw = raw.get("atmosphere", {})
    s_raw = raw.get("sensor_and_artifacts", {})

    flight = FlightConfig(
        max_start_offset_km=float(f_raw.get("max_start_offset_km", 1.0)),
        yaw_deg_range=_pair(f_raw.get("yaw_deg_range", (0.0, 360.0))),
        speed_mps_range=_pair(f_raw.get("speed_mps_range", (12.0, 32.0))),
        altitude_agl_m_range=_pair(f_raw.get("altitude_agl_m_range", (90.0, 300.0))),
        boundary_margin_m=float(f_raw.get("boundary_margin_m", 120.0)),
    )

    lens_dist = LensDistortionConfig(
        enabled=bool(ld_raw.get("enabled", True)),
        k1_range=_pair(ld_raw.get("k1_range", (-0.22, 0.12))),
        k2_range=_pair(ld_raw.get("k2_range", (-0.05, 0.05))),
        p1_range=_pair(ld_raw.get("p1_range", (-0.01, 0.01))),
        p2_range=_pair(ld_raw.get("p2_range", (-0.01, 0.01))),
    )

    res_list = [
        (int(r[0]), int(r[1]))
        for r in c_raw.get("resolutions", [(854, 480), (1280, 720), (1920, 1080)])
    ]
    camera = CameraConfig(
        resolutions=res_list,
        fov_deg_range=_pair(c_raw.get("fov_deg_range", (50.0, 90.0))),
        pitch_deg_range=_pair(c_raw.get("pitch_deg_range", (-85.0, -28.0))),
        spectral_modes=list(
            c_raw.get(
                "spectral_modes",
                ["rgb", "panchromatic", "nir", "false_color_cir", "thermal_lwir", "night_vision"],
            )
        ),
        lens_distortion=lens_dist,
    )

    lighting = LightingConfig(
        sun_azimuth_deg_range=_pair(l_raw.get("sun_azimuth_deg_range", (0.0, 360.0))),
        sun_elevation_deg_range=_pair(l_raw.get("sun_elevation_deg_range", (15.0, 75.0))),
        night_probability=float(l_raw.get("night_probability", 0.20)),
        shadow_strength_range=_pair(l_raw.get("shadow_strength_range", (0.25, 0.70))),
        color_temp_k_range=_pair(l_raw.get("color_temp_k_range", (3600.0, 8200.0))),
        wb_tint_range=_pair(l_raw.get("wb_tint_range", (-0.07, 0.07))),
    )

    atmosphere = AtmosphereConfig(
        haze_enabled=bool(a_raw.get("haze_enabled", True)),
        visibility_km_range=_pair(a_raw.get("visibility_km_range", (1.8, 22.0))),
        sun_glare_enabled=bool(a_raw.get("sun_glare_enabled", True)),
        glare_intensity_range=_pair(a_raw.get("glare_intensity_range", (0.15, 0.70))),
        specular_glint_range=_pair(a_raw.get("specular_glint_range", (0.10, 0.50))),
    )

    sensor = SensorArtifactsConfig(
        motion_blur_enabled=bool(s_raw.get("motion_blur_enabled", True)),
        shutter_sec_range=_pair(s_raw.get("shutter_sec_range", (0.004, 0.025))),
        rolling_shutter_strength_range=_pair(
            s_raw.get("rolling_shutter_strength_range", (0.0, 0.60))
        ),
        auto_exposure_enabled=bool(s_raw.get("auto_exposure_enabled", True)),
        ae_target_luma_range=_pair(s_raw.get("ae_target_luma_range", (0.44, 0.58))),
        ae_lag_alpha_range=_pair(s_raw.get("ae_lag_alpha_range", (0.10, 0.35))),
        dr_clip_strength_range=_pair(s_raw.get("dr_clip_strength_range", (0.0, 0.30))),
        sensor_noise_enabled=bool(s_raw.get("sensor_noise_enabled", True)),
        iso_noise_range=_pair(s_raw.get("iso_noise_range", (0.005, 0.040))),
        thermal_fpn_range=_pair(s_raw.get("thermal_fpn_range", (0.008, 0.035))),
        lens_crack_probability=float(s_raw.get("lens_crack_probability", 0.25)),
        watermark_droplet_probability=float(
            s_raw.get("watermark_droplet_probability", 0.30)
        ),
    )

    return GenerationConfig(
        num_videos=int(raw.get("num_videos", 4)),
        duration_sec=float(raw.get("duration_sec", 5.0)),
        fps=int(raw.get("fps", 24)),
        seed=raw.get("seed", 42),
        flight=flight,
        camera=camera,
        lighting=lighting,
        atmosphere=atmosphere,
        sensor_and_artifacts=sensor,
    )


def sample_video_params(
    cfg: GenerationConfig,
    rng: np.random.Generator,
    video_index: int,
    start_x_m: float,
    start_y_m: float,
    start_offset_from_origin_m: float,
    yaw_deg: float,
    speed_mps: float,
    altitude_agl_m: float,
) -> VideoSampleParams:
    """Sample all camera, lighting, atmosphere, and sensor parameters for one clip."""
    res_idx = int(rng.integers(0, len(cfg.camera.resolutions)))
    width, height = cfg.camera.resolutions[res_idx]

    spectral_mode = str(
        cfg.camera.spectral_modes[int(rng.integers(0, len(cfg.camera.spectral_modes)))]
    )
    is_night = bool(
        spectral_mode == "night_vision"
        or rng.random() < cfg.lighting.night_probability
    )

    if cfg.camera.lens_distortion.enabled:
        k1 = float(rng.uniform(*cfg.camera.lens_distortion.k1_range))
        k2 = float(rng.uniform(*cfg.camera.lens_distortion.k2_range))
        p1 = float(rng.uniform(*cfg.camera.lens_distortion.p1_range))
        p2 = float(rng.uniform(*cfg.camera.lens_distortion.p2_range))
    else:
        k1 = k2 = p1 = p2 = 0.0

    return VideoSampleParams(
        video_index=video_index,
        seed=int(rng.integers(0, 2**31 - 1)),
        width=width,
        height=height,
        fps=cfg.fps,
        duration_sec=cfg.duration_sec,
        start_x_m=float(start_x_m),
        start_y_m=float(start_y_m),
        start_offset_from_origin_m=float(start_offset_from_origin_m),
        yaw_deg=float(yaw_deg),
        speed_mps=float(speed_mps),
        altitude_agl_m=float(altitude_agl_m),
        fov_deg=float(rng.uniform(*cfg.camera.fov_deg_range)),
        pitch_deg=float(rng.uniform(*cfg.camera.pitch_deg_range)),
        spectral_mode=spectral_mode,
        k1=k1,
        k2=k2,
        p1=p1,
        p2=p2,
        sun_azimuth_deg=float(rng.uniform(*cfg.lighting.sun_azimuth_deg_range)),
        sun_elevation_deg=float(rng.uniform(*cfg.lighting.sun_elevation_deg_range)),
        is_night=is_night,
        shadow_strength=float(rng.uniform(*cfg.lighting.shadow_strength_range)),
        color_temp_k=float(rng.uniform(*cfg.lighting.color_temp_k_range)),
        wb_tint=float(rng.uniform(*cfg.lighting.wb_tint_range)),
        haze_enabled=cfg.atmosphere.haze_enabled,
        visibility_km=float(rng.uniform(*cfg.atmosphere.visibility_km_range)),
        sun_glare_enabled=cfg.atmosphere.sun_glare_enabled and not is_night,
        glare_intensity=float(rng.uniform(*cfg.atmosphere.glare_intensity_range)),
        specular_glint=float(rng.uniform(*cfg.atmosphere.specular_glint_range)),
        motion_blur_enabled=cfg.sensor_and_artifacts.motion_blur_enabled,
        shutter_sec=float(rng.uniform(*cfg.sensor_and_artifacts.shutter_sec_range)),
        rolling_shutter_strength=float(
            rng.uniform(*cfg.sensor_and_artifacts.rolling_shutter_strength_range)
        ),
        auto_exposure_enabled=cfg.sensor_and_artifacts.auto_exposure_enabled,
        ae_target_luma=float(
            rng.uniform(*cfg.sensor_and_artifacts.ae_target_luma_range)
        ),
        ae_lag_alpha=float(rng.uniform(*cfg.sensor_and_artifacts.ae_lag_alpha_range)),
        dr_clip_strength=float(
            rng.uniform(*cfg.sensor_and_artifacts.dr_clip_strength_range)
        ),
        sensor_noise_enabled=cfg.sensor_and_artifacts.sensor_noise_enabled,
        iso_noise=float(rng.uniform(*cfg.sensor_and_artifacts.iso_noise_range)),
        thermal_fpn=float(rng.uniform(*cfg.sensor_and_artifacts.thermal_fpn_range)),
        has_lens_crack=bool(
            rng.random() < cfg.sensor_and_artifacts.lens_crack_probability
        ),
        has_watermark_droplets=bool(
            rng.random() < cfg.sensor_and_artifacts.watermark_droplet_probability
        ),
    )


def load_regions_registry(registry_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the confirmed region definitions from REGIONS_AND_PARAMS.json."""
    if registry_path is None:
        registry_path = Path(__file__).resolve().parents[2] / "REGIONS_AND_PARAMS.json"
    data = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    return {r["id"]: r for r in data.get("regions", [])}
