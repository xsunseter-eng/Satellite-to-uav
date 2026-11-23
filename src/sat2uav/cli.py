"""Command-line interface and orchestration pipeline for Satellite-to-UAV."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    GenerationConfig,
    load_generation_config,
    load_regions_registry,
)
from .effects import EffectsPipeline
from .geo_fetcher import fetch_region_sample
from .gpu_renderer import EngineType, create_renderer
from .scene import load_scene
from .trajectory import sample_valid_flight
from .video_writer import FFmpegVideoWriter
from .xml_export import save_footage_xml


def generate_videos_from_scene(
    scene_path: str | Path,
    output_dir: str | Path,
    cfg: GenerationConfig,
    dsm_path: str | Path | None = None,
    engine: EngineType | None = None,
) -> list[dict[str, Any]]:
    """Generate N randomized synthetic UAV footages (MP4 + every-10-frame PNG + XML)."""
    scene = load_scene(scene_path, dsm_path=dsm_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_engine: EngineType = engine or cfg.render_engine  # type: ignore[assignment]
    rng = np.random.default_rng(cfg.seed)
    results: list[dict[str, Any]] = []

    print(
        f"[sat2uav] Loaded scene: {scene.width_px}x{scene.height_px} px "
        f"(GSD={scene.gsd_m:.2f}m, CRS={scene.crs}, engine={selected_engine}, "
        f"DSM={scene.z_min_m:.1f}..{scene.z_max_m:.1f}m, real_nir={scene.has_real_nir})"
    )

    for idx in range(cfg.num_videos):
        t0 = time.perf_counter()
        params, poses = sample_valid_flight(scene, cfg, rng, video_index=idx)

        try:
            renderer = create_renderer(selected_engine, scene, params)
            active_engine = selected_engine
        except Exception as exc:
            print(f"  [warn] Engine '{selected_engine}' unavailable ({exc}); falling back to cpu_raymarch")
            renderer = create_renderer("cpu_raymarch", scene, params)
            active_engine = "cpu_raymarch"

        effects = EffectsPipeline(params)

        stem = f"footage_{idx:03d}"
        footage_dir = out_dir / stem
        frames_dir = footage_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        video_path = footage_dir / f"{stem}.mp4"
        xml_path = footage_dir / f"{stem}.xml"
        keyframe_files: dict[int, str] = {}
        kf_step = max(1, int(cfg.keyframe_interval))

        try:
            with FFmpegVideoWriter(
                video_path, width=params.width, height=params.height, fps=params.fps
            ) as writer:
                for pose in poses:
                    render_out = renderer.render_frame(pose)
                    frame_rgb = effects.apply(render_out, pose, renderer.focal_px)
                    writer.write_frame(frame_rgb)

                    if pose.frame_idx % kf_step == 0:
                        kf_name = f"frame_{pose.frame_idx:06d}.png"
                        kf_path = frames_dir / kf_name
                        cv2.imwrite(str(kf_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                        keyframe_files[pose.frame_idx] = f"frames/{kf_name}"
        finally:
            if hasattr(renderer, "close"):
                renderer.close()

        save_footage_xml(
            output_xml=xml_path,
            video_filename=video_path.name,
            engine_name=active_engine,
            scene=scene,
            params=params,
            poses=poses,
            keyframe_interval=kf_step,
            keyframe_files=keyframe_files,
            visibility_diagnostics=params.visibility_diagnostics,
        )

        elapsed = time.perf_counter() - t0
        speed_kmh = params.speed_mps * 3.6
        summary = {
            "footage_dir": str(footage_dir),
            "video_path": str(video_path),
            "xml_path": str(xml_path),
            "keyframe_count": len(keyframe_files),
            "engine": active_engine,
            "resolution": f"{params.width}x{params.height}",
            "spectral_mode": params.spectral_mode,
            "is_night": params.is_night,
            "yaw_deg": round(params.yaw_deg, 1),
            "pitch_deg": round(params.pitch_deg, 1),
            "fov_deg": round(params.fov_deg, 1),
            "speed_kmh": round(speed_kmh, 1),
            "altitude_agl_m": round(params.altitude_agl_m, 1),
            "visibility_km": round(params.visibility_km, 2),
            "render_sec": round(elapsed, 2),
        }
        results.append(summary)
        print(
            f"  [{idx + 1}/{cfg.num_videos}] {stem} ({active_engine}) "
            f"| {params.width}x{params.height} {params.spectral_mode}{'(night)' if params.is_night else ''} "
            f"| alt={params.altitude_agl_m:.0f}m spd={speed_kmh:.0f}km/h vis={params.visibility_km:.1f}km "
            f"| kf={len(keyframe_files)} PNGs | {elapsed:.1f}s"
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

    sub.add_parser("list-regions", help="List all confirmed target regions and geofences.")

    p_fetch = sub.add_parser(
        "fetch", help="Fetch an on-demand standardized GeoTIFF+DSM corridor for a region."
    )
    p_fetch.add_argument("--region", required=True, help="Region ID (e.g. R4_MOAB_CANYONLANDS_UT)")
    p_fetch.add_argument("--out", required=True, help="Output scene directory")
    p_fetch.add_argument("--size-km", type=float, default=3.5, help="Square corridor size in km")
    p_fetch.add_argument("--gsd", type=float, default=1.0, help="Target resolution in meters/pixel")
    p_fetch.add_argument("--lat", type=float, default=None, help="Optional custom center latitude")
    p_fetch.add_argument("--lon", type=float, default=None, help="Optional custom center longitude")
    p_fetch.add_argument(
        "--random-in-geofence",
        action="store_true",
        help="Randomly pick a corridor inside the region's 100-300 km geofence",
    )
    p_fetch.add_argument("--seed", type=int, default=42)

    p_gen = sub.add_parser(
        "generate", help="Generate N UAV footages (MP4 + 10-frame PNG + XML) from a local scene."
    )
    p_gen.add_argument("--scene", required=True, help="Path to scene directory or GeoTIFF file")
    p_gen.add_argument("--dsm", default=None, help="Optional explicit path to DSM GeoTIFF")
    p_gen.add_argument("--out", required=True, help="Output directory (region folder)")
    p_gen.add_argument("--config", default=None, help="Path to YAML generation config")
    p_gen.add_argument(
        "--engine",
        default=None,
        choices=["gpu_opengl_raymarch", "gpu_opengl_mesh", "cpu_raymarch"],
        help="Render engine override",
    )
    p_gen.add_argument("--num-videos", type=int, default=None, help="Override number of videos N")
    p_gen.add_argument("--duration", type=float, default=None, help="Override video duration (sec)")
    p_gen.add_argument("--fps", type=int, default=None, help="Override video FPS")
    p_gen.add_argument("--seed", type=int, default=None, help="Override RNG seed")

    p_run = sub.add_parser(
        "run-region", help="End-to-end: fetch a corridor from a region and generate N UAV footages."
    )
    p_run.add_argument("--region", required=True, help="Region ID (e.g. R4_MOAB_CANYONLANDS_UT)")
    p_run.add_argument("--out", required=True, help="Dataset root directory (creates <out>/<REGION_ID>/)")
    p_run.add_argument("--config", default=None, help="Path to YAML generation config")
    p_run.add_argument(
        "--engine",
        default=None,
        choices=["gpu_opengl_raymarch", "gpu_opengl_mesh", "cpu_raymarch"],
    )
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
            scene_path=args.scene,
            output_dir=args.out,
            cfg=cfg,
            dsm_path=args.dsm,
            engine=args.engine,
        )
        return 0

    if args.command == "run-region":
        region_dir = Path(args.out) / args.region
        scene_dir = fetch_region_sample(
            region_id=args.region,
            output_dir=region_dir / "_scene_cache",
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
            scene_path=scene_dir,
            output_dir=region_dir,
            cfg=cfg,
            engine=args.engine,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
