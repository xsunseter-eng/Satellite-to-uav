"""Command-line interface and orchestration pipeline for Satellite-to-UAV."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    GenerationConfig,
    load_generation_config,
    load_regions_registry,
)
from .effects import EffectsPipeline
from .geo_fetcher import fetch_region_sample
from .renderer import DSMCameraRenderer
from .scene import load_scene
from .trajectory import sample_valid_flight
from .video_writer import FFmpegVideoWriter, save_video_metadata


def generate_videos_from_scene(
    scene_path: str | Path,
    output_dir: str | Path,
    cfg: GenerationConfig,
    dsm_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Generate N randomized synthetic UAV videos + telemetry JSONs from a scene."""
    scene = load_scene(scene_path, dsm_path=dsm_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    results: list[dict[str, Any]] = []

    print(
        f"[sat2uav] Loaded scene: {scene.width_px}x{scene.height_px} px "
        f"(GSD={scene.gsd_m:.2f}m, CRS={scene.crs}, "
        f"DSM={scene.z_min_m:.1f}..{scene.z_max_m:.1f}m, real_nir={scene.has_real_nir})"
    )

    for idx in range(cfg.num_videos):
        t0 = time.perf_counter()
        params, poses = sample_valid_flight(scene, cfg, rng, video_index=idx)
        renderer = DSMCameraRenderer(scene, params)
        effects = EffectsPipeline(params)

        stem = f"uav_{idx:03d}_{params.spectral_mode}_{params.width}x{params.height}"
        video_path = out_dir / f"{stem}.mp4"
        meta_path = out_dir / f"{stem}_metadata.json"

        with FFmpegVideoWriter(
            video_path, width=params.width, height=params.height, fps=params.fps
        ) as writer:
            for pose in poses:
                render_out = renderer.render_frame(pose)
                frame_rgb = effects.apply(render_out, pose, renderer.focal_px)
                writer.write_frame(frame_rgb)

        payload = {
            "video_file": video_path.name,
            "scene_metadata": scene.metadata,
            "scene_summary": {
                "crs": scene.crs,
                "gsd_m": scene.gsd_m,
                "has_real_nir": scene.has_real_nir,
                "has_real_dsm": scene.has_real_dsm,
                "z_min_m": scene.z_min_m,
                "z_max_m": scene.z_max_m,
            },
            "parameters": params.to_dict(),
            "telemetry_frames": [p.to_dict() for p in poses],
        }
        save_video_metadata(meta_path, payload)

        elapsed = time.perf_counter() - t0
        summary = {
            "video_path": str(video_path),
            "metadata_path": str(meta_path),
            "resolution": f"{params.width}x{params.height}",
            "spectral_mode": params.spectral_mode,
            "is_night": params.is_night,
            "yaw_deg": round(params.yaw_deg, 1),
            "pitch_deg": round(params.pitch_deg, 1),
            "fov_deg": round(params.fov_deg, 1),
            "speed_mps": round(params.speed_mps, 1),
            "altitude_agl_m": round(params.altitude_agl_m, 1),
            "render_sec": round(elapsed, 2),
        }
        results.append(summary)
        print(
            f"  [{idx + 1}/{cfg.num_videos}] {video_path.name} "
            f"| mode={params.spectral_mode}{'(night)' if params.is_night else ''} "
            f"| alt={params.altitude_agl_m:.0f}m spd={params.speed_mps:.1f}m/s "
            f"| yaw={params.yaw_deg:.0f}° pitch={params.pitch_deg:.0f}° fov={params.fov_deg:.0f}° "
            f"| {elapsed:.1f}s"
        )

    manifest_path = out_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sat2uav",
        description="Generate randomized synthetic UAV flight videos from satellite & DSM data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list-regions
    sub.add_parser("list-regions", help="List all confirmed target regions and geofences.")

    # fetch
    p_fetch = sub.add_parser(
        "fetch", help="Fetch an on-demand standardized GeoTIFF+DSM corridor for a region."
    )
    p_fetch.add_argument("--region", required=True, help="Region ID (e.g. R4_MOAB_CANYONLANDS_UT)")
    p_fetch.add_argument("--out", required=True, help="Output scene directory")
    p_fetch.add_argument("--size-km", type=float, default=2.0, help="Square corridor size in km")
    p_fetch.add_argument("--gsd", type=float, default=1.0, help="Target resolution in meters/pixel")
    p_fetch.add_argument("--lat", type=float, default=None, help="Optional custom center latitude")
    p_fetch.add_argument("--lon", type=float, default=None, help="Optional custom center longitude")
    p_fetch.add_argument(
        "--random-in-geofence",
        action="store_true",
        help="Randomly pick a corridor inside the region's 100-300 km geofence",
    )
    p_fetch.add_argument("--seed", type=int, default=42)

    # generate
    p_gen = sub.add_parser(
        "generate", help="Generate N UAV videos from a standardized local scene directory or GeoTIFF."
    )
    p_gen.add_argument("--scene", required=True, help="Path to scene directory or GeoTIFF file")
    p_gen.add_argument("--dsm", default=None, help="Optional explicit path to DSM GeoTIFF")
    p_gen.add_argument("--out", required=True, help="Output directory for generated MP4s + JSONs")
    p_gen.add_argument("--config", default=None, help="Path to YAML generation config")
    p_gen.add_argument("--num-videos", type=int, default=None, help="Override number of videos N")
    p_gen.add_argument("--duration", type=float, default=None, help="Override video duration (sec)")
    p_gen.add_argument("--fps", type=int, default=None, help="Override video FPS")
    p_gen.add_argument("--seed", type=int, default=None, help="Override RNG seed")

    # run-region (all-in-one fetch + generate)
    p_run = sub.add_parser(
        "run-region", help="End-to-end: fetch a corridor from a region and generate N UAV videos."
    )
    p_run.add_argument("--region", required=True, help="Region ID (e.g. R4_MOAB_CANYONLANDS_UT)")
    p_run.add_argument("--out", required=True, help="Output run directory")
    p_run.add_argument("--config", default=None, help="Path to YAML generation config")
    p_run.add_argument("--size-km", type=float, default=3.5, help="Corridor box size in km")
    p_run.add_argument("--gsd", type=float, default=1.0, help="Corridor GSD in meters/pixel")
    p_run.add_argument("--num-videos", type=int, default=None)
    p_run.add_argument("--duration", type=float, default=None)
    p_run.add_argument("--fps", type=int, default=None)
    p_run.add_argument("--random-in-geofence", action="store_true")
    p_run.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-regions":
        regions = load_regions_registry()
        for rid, reg in regions.items():
            o = reg["origin"]
            print(
                f"{rid:28s} | {reg['country']:6s} | "
                f"origin=({o['lat']:8.4f}, {o['lon']:9.4f}) | "
                f"r={reg['geofence_radius_km']:3d} km | {reg['name']}"
            )
        return 0

    if args.command == "fetch":
        scene_dir = fetch_region_sample(
            region_id=args.region,
            output_dir=args.out,
            size_km=args.size_km,
            gsd_m=args.gsd,
            random_within_geofence=args.random_in_geofence,
            seed=args.seed,
            lat=args.lat,
            lon=args.lon,
        )
        print(f"[sat2uav] Fetched standardized scene bundle -> {scene_dir}")
        return 0

    if args.command == "generate":
        cfg = load_generation_config(args.config)
        if args.num_videos is not None:
            cfg.num_videos = args.num_videos
        if args.duration is not None:
            cfg.duration_sec = args.duration
        if args.fps is not None:
            cfg.fps = args.fps
        if args.seed is not None:
            cfg.seed = args.seed
        generate_videos_from_scene(
            scene_path=args.scene, output_dir=args.out, cfg=cfg, dsm_path=args.dsm
        )
        return 0

    if args.command == "run-region":
        out_root = Path(args.out)
        scene_dir = fetch_region_sample(
            region_id=args.region,
            output_dir=out_root / "scene",
            size_km=args.size_km,
            gsd_m=args.gsd,
            random_within_geofence=args.random_in_geofence,
            seed=args.seed,
        )
        cfg = load_generation_config(args.config)
        if args.num_videos is not None:
            cfg.num_videos = args.num_videos
        if args.duration is not None:
            cfg.duration_sec = args.duration
        if args.fps is not None:
            cfg.fps = args.fps
        cfg.seed = args.seed
        generate_videos_from_scene(
            scene_path=scene_dir, output_dir=out_root / "videos", cfg=cfg
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
