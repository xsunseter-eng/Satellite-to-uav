"""Unit and integration tests for the Satellite-to-UAV generation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sat2uav.cli import generate_videos_from_scene
from sat2uav.config import load_generation_config, load_regions_registry
from sat2uav.effects import EffectsPipeline
from sat2uav.geo_fetcher import create_synthetic_test_scene
from sat2uav.renderer import DSMCameraRenderer
from sat2uav.scene import load_scene
from sat2uav.trajectory import sample_valid_flight


def test_regions_registry_confirmed_and_geofence_valid():
    regions = load_regions_registry()
    assert len(regions) == 6
    for rid, reg in regions.items():
        assert reg["status"] == "confirmed", f"{rid} should be confirmed"
        r = reg["geofence_radius_km"]
        assert 100 <= r <= 300, f"{rid} radius {r} outside [100, 300]"


def test_end_to_end_synthetic_video_generation(tmp_path: Path):
    scene_dir = create_synthetic_test_scene(tmp_path / "scene", width=320, height=320, gsd_m=2.0)
    cfg = load_generation_config(Path(__file__).resolve().parents[1] / "configs" / "default_generation.yaml")
    cfg.num_videos = 3
    cfg.duration_sec = 0.6
    cfg.fps = 10
    cfg.camera.resolutions = [(426, 240), (640, 360)]
    cfg.flight.max_start_offset_km = 0.15
    cfg.flight.boundary_margin_m = 40.0
    cfg.sensor_and_artifacts.lens_crack_probability = 1.0
    cfg.sensor_and_artifacts.watermark_droplet_probability = 1.0

    out_dir = tmp_path / "outputs"
    results = generate_videos_from_scene(scene_dir, out_dir, cfg)

    assert len(results) == 3
    assert (out_dir / "batch_manifest.json").exists()

    for res in results:
        mp4 = Path(res["video_path"])
        meta = Path(res["metadata_path"])
        assert mp4.exists() and mp4.stat().st_size > 1000
        assert meta.exists()
        payload = json.loads(meta.read_text(encoding="utf-8"))
        assert len(payload["telemetry_frames"]) == 6
        params = payload["parameters"]
        assert params["has_lens_crack"] is True
        assert params["has_watermark_droplets"] is True
        assert params["start_offset_from_origin_m"] <= 155.0


def test_all_spectral_modes_render_cleanly(tmp_path: Path):
    scene_dir = create_synthetic_test_scene(tmp_path / "scene", width=256, height=256, gsd_m=2.0)
    scene = load_scene(scene_dir)
    assert scene.has_real_nir
    assert scene.has_real_dsm

    cfg = load_generation_config()
    cfg.duration_sec = 0.2
    cfg.fps = 5
    cfg.camera.resolutions = [(320, 180)]
    rng = np.random.default_rng(7)

    for mode in ["rgb", "panchromatic", "nir", "false_color_cir", "thermal_lwir", "night_vision"]:
        cfg.camera.spectral_modes = [mode]
        params, poses = sample_valid_flight(scene, cfg, rng, video_index=0)
        renderer = DSMCameraRenderer(scene, params, coarse_steps=20, refine_steps=4)
        effects = EffectsPipeline(params)
        render_out = renderer.render_frame(poses[0])
        frame = effects.apply(render_out, poses[0], renderer.focal_px)

        assert frame.shape == (180, 320, 3)
        assert frame.dtype == np.uint8
        assert np.isfinite(render_out.depth_m).all()
        assert frame.std() > 1.0, f"Mode {mode} rendered a flat image"
