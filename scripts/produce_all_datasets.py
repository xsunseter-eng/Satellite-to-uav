"""Resumable full-scale production batch runner for all 8 confirmed regions.

Generates 110 videos per region (100 clean + 10 lens-crack / water-droplet defect
videos), 10.0 s @ 30 FPS (300 frames), saving inside `dataset/<REGION_ID>/`:
  - <VIDEO_NAME>.mp4
  - <VIDEO_NAME>.xml  (all parameters, visibility diagnostics & 300-frame 6-DoF telemetry)
  - <VIDEO_NAME>/frame_000000.png ... frame_000290.png  (every 10 frames = 30 PNGs)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sat2uav.config import GenerationConfig, load_generation_config, load_regions_registry
from sat2uav.effects import EffectsPipeline
from sat2uav.fast_gpu_renderer import FastSceneGPURenderer
from sat2uav.geo_fetcher import fetch_scene_corridor, sample_point_in_geofence
from sat2uav.scene import SceneBundle, load_scene
from sat2uav.trajectory import sample_valid_flight
from sat2uav.video_writer import FFmpegVideoWriter
from sat2uav.xml_export import save_footage_xml

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset"
REVIEW_SCENES_CACHE = ROOT / "outputs" / "quality_review_30" / "scenes"

SPECTRAL_MODES = ["rgb", "panchromatic", "nir", "false_color_cir", "thermal_lwir", "night_vision"]


def _is_video_complete(region_dir: Path, video_name: str, expected_keyframes: int) -> bool:
    mp4 = region_dir / f"{video_name}.mp4"
    xml = region_dir / f"{video_name}.xml"
    shot_dir = region_dir / video_name
    if not (mp4.exists() and mp4.stat().st_size > 2000 and xml.exists() and shot_dir.is_dir()):
        return False
    return len(list(shot_dir.glob("frame_*.png"))) >= expected_keyframes


def _prepare_region_corridors(
    region_id: str,
    reg: dict[str, Any],
    corridor_root: Path,
    num_corridors: int = 5,
    size_km: float = 3.0,
    gsd_m: float = 1.2,
    seed: int = 42,
) -> list[Path]:
    """Fetch (or reuse cached) diverse corridors across the region's 100-300 km geofence."""
    corridor_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + abs(hash(region_id)) % 100000)

    corridors: list[Path] = []
    origin_lat = float(reg["origin"]["lat"])
    origin_lon = float(reg["origin"]["lon"])
    radius_km = float(reg["geofence_radius_km"])
    country = str(reg["country"])

    # Corridor 0: reuse quality_review_30 cache if available, otherwise fetch at exact origin
    c0 = corridor_root / "corridor_00"
    if not (c0 / "imagery.tif").exists():
        cached = REVIEW_SCENES_CACHE / region_id
        if (cached / "imagery.tif").exists() and (cached / "dsm.tif").exists():
            shutil.copytree(cached, c0, dirs_exist_ok=True)
            print(f"  [corridor 0] Reused cached scene for {region_id}")
        else:
            print(f"  [corridor 0] Fetching origin ({origin_lat:.4f}, {origin_lon:.4f}) for {region_id}...")
            fetch_scene_corridor(
                lat=origin_lat,
                lon=origin_lon,
                output_dir=c0,
                size_km=size_km,
                gsd_m=gsd_m,
                region_id=region_id,
                country=country,
            )
    corridors.append(c0)

    # Corridors 1..num_corridors-1: randomly sampled inside the full 100-300 km geofence
    for c_idx in range(1, num_corridors):
        cdir = corridor_root / f"corridor_{c_idx:02d}"
        if (cdir / "imagery.tif").exists() and (cdir / "dsm.tif").exists():
            corridors.append(cdir)
            continue

        fetched = False
        for attempt in range(5):
            # Stay within 85% of geofence radius so edge tiles remain well inside land bounds
            lat, lon, dist_km = sample_point_in_geofence(
                origin_lat, origin_lon, radius_km * 0.85, rng
            )
            print(
                f"  [corridor {c_idx}] Fetching {region_id} @ ({lat:.4f}, {lon:.4f}, "
                f"{dist_km:.1f} km from origin, try {attempt + 1})..."
            )
            try:
                fetch_scene_corridor(
                    lat=lat,
                    lon=lon,
                    output_dir=cdir,
                    size_km=size_km,
                    gsd_m=gsd_m,
                    region_id=region_id,
                    country=country,
                )
                # Verify downloaded imagery isn't blank/corrupt
                sc_check = load_scene(cdir)
                if float(sc_check.imagery[:, :, :3].std()) > 0.01:
                    corridors.append(cdir)
                    fetched = True
                    break
                shutil.rmtree(cdir, ignore_errors=True)
            except Exception as exc:
                print(f"    [warn] Fetch failed ({exc}); retrying another point inside geofence...")
                shutil.rmtree(cdir, ignore_errors=True)
        if not fetched:
            print(f"  [warn] Corridor {c_idx} fallback -> reusing corridor 0 for {region_id}")
            corridors.append(c0)

    return corridors


def produce_region(
    region_id: str,
    reg: dict[str, Any],
    cfg: GenerationConfig,
    dataset_root: Path,
    clean_count: int = 100,
    defect_count: int = 10,
    num_corridors: int = 5,
) -> dict[str, Any]:
    """Produce 110 videos (100 clean + 10 crack/droplet defect) for one region."""
    total_videos = clean_count + defect_count
    region_short = region_id.split("_")[0]  # e.g. R1, R2, ..., R8
    region_dir = dataset_root / region_id
    corridor_root = region_dir / "_corridors"
    region_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n======================================================================")
    print(f"[REGION] {region_id} ({reg['name']}) | Target: {total_videos} videos ({clean_count} clean + {defect_count} defect)")
    print(f"======================================================================")

    corridor_paths = _prepare_region_corridors(
        region_id=region_id,
        reg=reg,
        corridor_root=corridor_root,
        num_corridors=num_corridors,
        seed=cfg.seed or 42,
    )

    # Group video indices by corridor so each corridor's textures are uploaded to GPU VRAM only ONCE!
    videos_per_corridor: dict[Path, list[int]] = {}
    for v_idx in range(total_videos):
        c_path = corridor_paths[v_idx % len(corridor_paths)]
        videos_per_corridor.setdefault(c_path, []).append(v_idx)

    expected_frames = int(round(cfg.duration_sec * cfg.fps))
    kf_step = max(1, int(cfg.keyframe_interval))
    expected_kf = len(range(0, expected_frames, kf_step))

    region_records: list[dict[str, Any]] = []
    t_region0 = time.perf_counter()

    for c_idx, (c_path, vid_indices) in enumerate(videos_per_corridor.items()):
        scene: SceneBundle = load_scene(c_path)
        print(
            f"  [GPU Load] Corridor {c_idx + 1}/{len(videos_per_corridor)}: {c_path.name} "
            f"({scene.width_px}x{scene.height_px} px, {len(vid_indices)} videos)"
        )

        with FastSceneGPURenderer(scene) as gpu:
            for v_idx in vid_indices:
                # Deterministic per-video RNG so resuming produces identical parameters
                v_rng = np.random.default_rng(
                    (cfg.seed or 42) * 10000 + abs(hash(region_id)) % 997 * 200 + v_idx
                )

                # Stratify spectral mode evenly across the 110 videos
                forced_mode = SPECTRAL_MODES[v_idx % len(SPECTRAL_MODES)]
                cfg_v = GenerationConfig(
                    num_videos=1,
                    duration_sec=cfg.duration_sec,
                    fps=cfg.fps,
                    keyframe_interval=cfg.keyframe_interval,
                    keyframe_format=cfg.keyframe_format,
                    render_engine=gpu.engine_name,
                    seed=int(v_rng.integers(0, 2**31 - 1)),
                    flight=cfg.flight,
                    camera=cfg.camera,
                    lighting=cfg.lighting,
                    atmosphere=cfg.atmosphere,
                    sensor_and_artifacts=cfg.sensor_and_artifacts,
                )

                params, poses = sample_valid_flight(scene, cfg_v, v_rng, video_index=v_idx)
                params.spectral_mode = forced_mode
                if forced_mode == "night_vision":
                    params.is_night = True

                # Enforce exact 100 clean + 10 defect split
                if v_idx < clean_count:
                    params.has_lens_crack = False
                    params.has_watermark_droplets = False
                    defect_tag = "clean"
                else:
                    d_i = v_idx - clean_count
                    if d_i < 4:
                        params.has_lens_crack = True
                        params.has_watermark_droplets = False
                        defect_tag = "defect_crack"
                    elif d_i < 8:
                        params.has_lens_crack = False
                        params.has_watermark_droplets = True
                        defect_tag = "defect_droplet"
                    else:
                        params.has_lens_crack = True
                        params.has_watermark_droplets = True
                        defect_tag = "defect_crack_droplet"

                time_tag = "night" if params.is_night else "day"
                video_name = (
                    f"{region_short}_v{v_idx:03d}_{defect_tag}_"
                    f"{params.spectral_mode}_{time_tag}_{params.width}x{params.height}"
                )
                video_path = region_dir / f"{video_name}.mp4"
                xml_path = region_dir / f"{video_name}.xml"
                shot_dir = region_dir / video_name

                if _is_video_complete(region_dir, video_name, expected_kf):
                    region_records.append(
                        {
                            "video_index": v_idx,
                            "video_name": video_name,
                            "status": "skipped_existing",
                            "resolution": f"{params.width}x{params.height}",
                            "spectral_mode": params.spectral_mode,
                            "defect_tag": defect_tag,
                        }
                    )
                    continue

                t0 = time.perf_counter()
                shot_dir.mkdir(parents=True, exist_ok=True)
                gpu.configure_video(params)
                effects = EffectsPipeline(params)
                keyframe_files: dict[int, str] = {}

                with FFmpegVideoWriter(
                    video_path, width=params.width, height=params.height, fps=params.fps
                ) as writer:
                    for pose in poses:
                        img_f = gpu.render_frame_rgb_float(pose)
                        frame_rgb = effects.apply_post_only(img_f, pose, gpu.focal_px)
                        writer.write_frame(frame_rgb)

                        if pose.frame_idx % kf_step == 0:
                            kf_name = f"frame_{pose.frame_idx:06d}.png"
                            cv2.imwrite(
                                str(shot_dir / kf_name),
                                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                            )
                            keyframe_files[pose.frame_idx] = f"{video_name}/{kf_name}"

                save_footage_xml(
                    output_xml=xml_path,
                    video_filename=video_path.name,
                    engine_name=gpu.engine_name,
                    scene=scene,
                    params=params,
                    poses=poses,
                    keyframe_interval=kf_step,
                    keyframe_files=keyframe_files,
                    visibility_diagnostics=params.visibility_diagnostics,
                )

                elapsed = time.perf_counter() - t0
                rec = {
                    "video_index": v_idx,
                    "video_name": video_name,
                    "mp4": video_path.name,
                    "xml": xml_path.name,
                    "screenshots_dir": video_name,
                    "keyframe_count": len(keyframe_files),
                    "corridor": c_path.name,
                    "defect_tag": defect_tag,
                    "has_lens_crack": params.has_lens_crack,
                    "has_watermark_droplets": params.has_watermark_droplets,
                    "spectral_mode": params.spectral_mode,
                    "is_night": params.is_night,
                    "resolution": f"{params.width}x{params.height}",
                    "speed_kmh": round(params.speed_mps * 3.6, 1),
                    "altitude_agl_m": round(params.altitude_agl_m, 1),
                    "visibility_km": round(params.visibility_km, 2),
                    "render_sec": round(elapsed, 2),
                }
                region_records.append(rec)
                print(
                    f"    [{len(region_records):3d}/{total_videos}] {video_name} "
                    f"| alt={params.altitude_agl_m:.0f}m spd={params.speed_mps*3.6:.0f}km/h "
                    f"| {len(keyframe_files)} PNGs | {elapsed:.1f}s",
                    flush=True,
                )

    region_records.sort(key=lambda r: r["video_index"])
    manifest = {
        "region_id": region_id,
        "region_name": reg["name"],
        "country": reg["country"],
        "origin": reg["origin"],
        "geofence_radius_km": reg["geofence_radius_km"],
        "total_videos": len(region_records),
        "clean_videos": clean_count,
        "defect_videos": defect_count,
        "elapsed_sec": round(time.perf_counter() - t_region0, 1),
        "videos": region_records,
    }
    (region_dir / "region_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "region_id": region_id,
        "total_videos": len(region_records),
        "elapsed_sec": manifest["elapsed_sec"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce 110 UAV videos per region across all 8 regions.")
    parser.add_argument("--regions", nargs="*", default=None, help="Optional subset of region IDs")
    parser.add_argument("--clean", type=int, default=100, help="Number of clean videos per region")
    parser.add_argument("--defect", type=int, default=10, help="Number of crack/droplet videos per region")
    parser.add_argument("--corridors", type=int, default=5, help="Distinct corridors fetched per region geofence")
    parser.add_argument("--out", default=str(DATASET_ROOT), help="Root dataset output directory")
    args = parser.parse_args()

    cfg = load_generation_config(ROOT / "configs" / "default_generation.yaml")
    regions = load_regions_registry()
    target_rids = args.regions or list(regions.keys())

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    progress: list[dict[str, Any]] = []

    for rid in target_rids:
        if rid not in regions:
            raise KeyError(f"Unknown region {rid}")
        res = produce_region(
            region_id=rid,
            reg=regions[rid],
            cfg=cfg,
            dataset_root=out_root,
            clean_count=args.clean,
            defect_count=args.defect,
            num_corridors=args.corridors,
        )
        progress.append(res)
        (out_root / "production_status.json").write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )

    print("\n[ALL DONE] Completed production across regions:", [p["region_id"] for p in progress])


if __name__ == "__main__":
    main()
