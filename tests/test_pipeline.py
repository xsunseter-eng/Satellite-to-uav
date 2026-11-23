"""Unit and integration tests for the Satellite-to-UAV generation pipeline."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from sat2uav.cli import generate_videos_from_scene
from sat2uav.config import load_generation_config, load_regions_registry
from sat2uav.distributions import (
    STANDARD_RESOLUTIONS,
    sample_airspeed_kmh,
    sample_altitude_agl_m,
    sample_visibility_km,
)
from sat2uav.effects import EffectsPipeline
from sat2uav.geo_fetcher import create_synthetic_test_scene, sample_point_in_geofence
from sat2uav.gpu_renderer import create_renderer
from sat2uav.scene import load_scene
from sat2uav.trajectory import sample_valid_flight


def test_regions_registry_confirmed_and_geofence_valid():
    regions = load_regions_registry()
    assert len(regions) == 6
    for rid, reg in regions.items():
        assert reg["status"] == "confirmed", f"{rid} should be confirmed"
        r = reg["geofence_radius_km"]
        assert 100 <= r <= 300, f"{rid} radius {r} outside [100, 300]"


def test_aviation_distributions_match_spec():
    rng = np.random.default_rng(12345)
    speeds = np.array([sample_airspeed_kmh(rng) for _ in range(15000)])
    assert speeds.min() >= 100.0
    assert speeds.max() <= 2205.2
    # Mean must be centered on 204 km/h
    assert 199.0 <= float(np.mean(speeds)) <= 209.0
    # Median anchored near Bayraktar TB2 operational airspeed envelope (~170 km/h)
    assert 160.0 <= float(np.median(speeds)) <= 185.0

    alts = np.array([sample_altitude_agl_m(rng) for _ in range(10000)])
    assert alts.min() >= 120.0
    assert alts.max() <= 15240.0
    assert 1400.0 <= float(np.median(alts)) <= 1900.0

    # Desert plateau should be clearer on average than temperate France basin
    vis_moab = np.mean([sample_visibility_km("R4_MOAB_CANYONLANDS_UT", 1500.0, rng)[0] for _ in range(500)])
    vis_fr = np.mean([sample_visibility_km("R6_FRANCE_PARIS_LOIRE_BASIN", 1500.0, rng)[0] for _ in range(500)])
    assert vis_moab > vis_fr


def test_end_to_end_gpu_and_xml_keyframe_outputs(tmp_path: Path):
    scene_dir = create_synthetic_test_scene(tmp_path / "scene", width=320, height=320, gsd_m=2.0)
    cfg = load_generation_config(Path(__file__).resolve().parents[1] / "configs" / "default_generation.yaml")
    cfg.num_videos = 4
    cfg.duration_sec = 0.7
    cfg.fps = 30  # 21 frames -> keyframes at 0, 10, 20 (3 PNGs)
    cfg.keyframe_interval = 10
    cfg.flight.max_start_offset_km = 0.15
    cfg.sensor_and_artifacts.lens_crack_probability = 1.0
    cfg.sensor_and_artifacts.watermark_droplet_probability = 1.0

    out_dir = tmp_path / "R_TEST"
    results = generate_videos_from_scene(scene_dir, out_dir, cfg, engine="gpu_opengl_raymarch")

    assert len(results) == 4
    # Verify equal distribution across all 4 standard non-4K resolutions
    res_seen = [r["resolution"] for r in results]
    expected_res = [f"{w}x{h}" for w, h in STANDARD_RESOLUTIONS]
    assert res_seen == expected_res

    for res in results:
        mp4 = Path(res["video_path"])
        xml_p = Path(res["xml_path"])
        assert mp4.exists() and mp4.stat().st_size > 1000
        assert xml_p.exists()
        assert res["keyframe_count"] == 3  # frames 0, 10, 20

        root = ET.parse(xml_p).getroot()
        assert root.tag == "UAVFootageMetadata"
        frames = root.findall("./TelemetryFrames/Frame")
        assert len(frames) == 21
        assert frames[0].attrib.get("keyframePng") == "frames/frame_000000.png"
        assert frames[10].attrib.get("keyframePng") == "frames/frame_000010.png"
        assert (Path(res["footage_dir"]) / "frames" / "frame_000010.png").exists()


def test_all_three_engines_and_spectral_modes(tmp_path: Path):
    scene_dir = create_synthetic_test_scene(tmp_path / "scene", width=256, height=256, gsd_m=2.0)
    scene = load_scene(scene_dir)
    cfg = load_generation_config()
    cfg.duration_sec = 0.2
    cfg.fps = 10
    cfg.camera.resolutions = [(640, 360)]
    rng = np.random.default_rng(7)

    for engine in ("gpu_opengl_raymarch", "gpu_opengl_mesh", "cpu_raymarch"):
        for mode in ("rgb", "panchromatic", "nir", "false_color_cir", "thermal_lwir", "night_vision"):
            cfg.camera.spectral_modes = [mode]
            params, poses = sample_valid_flight(scene, cfg, rng, video_index=0)
            renderer = create_renderer(engine, scene, params)
            try:
                effects = EffectsPipeline(params)
                render_out = renderer.render_frame(poses[0])
                frame = effects.apply(render_out, poses[0], renderer.focal_px)
            finally:
                if hasattr(renderer, "close"):
                    renderer.close()

            assert frame.shape == (360, 640, 3)
            assert frame.dtype == np.uint8
            assert np.isfinite(render_out.depth_m).all()
            assert frame.std() > 1.0, f"{engine}/{mode} rendered a flat image"


def test_geofence_sampling_respects_radius():
    rng = np.random.default_rng(99)
    for _ in range(50):
        lat, lon, dist_km = sample_point_in_geofence(38.5733, -109.5498, 300.0, rng)
        assert 0.0 <= dist_km <= 300.01
        assert 35.0 <= lat <= 42.0
        assert -114.0 <= lon <= -105.0
