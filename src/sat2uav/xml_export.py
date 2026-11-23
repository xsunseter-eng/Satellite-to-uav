"""Structured XML metadata and per-frame 6-DoF telemetry serializer."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from .config import VideoSampleParams
from .scene import SceneBundle
from .trajectory import FramePose


def _add_kv(parent: ET.Element, mapping: dict[str, Any]) -> None:
    for k, v in mapping.items():
        child = ET.SubElement(parent, str(k))
        if isinstance(v, (list, tuple)):
            child.text = ", ".join(str(x) for x in v)
        elif isinstance(v, dict):
            _add_kv(child, v)
        else:
            child.text = "" if v is None else str(v)


def save_footage_xml(
    output_xml: str | Path,
    video_filename: str,
    engine_name: str,
    scene: SceneBundle,
    params: VideoSampleParams,
    poses: list[FramePose],
    keyframe_interval: int = 10,
    keyframe_files: dict[int, str] | None = None,
    visibility_diagnostics: dict[str, Any] | None = None,
) -> Path:
    """Serialize all footage parameters, scene provenance, keyframes, and telemetry to XML."""
    root = ET.Element(
        "UAVFootageMetadata",
        attrib={"schemaVersion": "1.1", "generator": "sat2uav"},
    )

    summary = ET.SubElement(root, "Summary")
    _add_kv(
        summary,
        {
            "VideoFile": video_filename,
            "RenderEngine": engine_name,
            "ResolutionWidth": params.width,
            "ResolutionHeight": params.height,
            "FPS": params.fps,
            "DurationSec": params.duration_sec,
            "TotalFrames": len(poses),
            "KeyframeInterval": keyframe_interval,
            "KeyframeFormat": "PNG",
        },
    )

    scene_el = ET.SubElement(root, "SceneProvenance")
    _add_kv(
        scene_el,
        {
            "RegionId": scene.metadata.get("region_id", "CUSTOM"),
            "Country": scene.metadata.get("country", "UNKNOWN"),
            "CRS": scene.crs,
            "GSDMeters": round(scene.gsd_m, 4),
            "OriginLat": scene.metadata.get("origin_lat", ""),
            "OriginLon": scene.metadata.get("origin_lon", ""),
            "ImageryCollection": scene.metadata.get("imagery_collection", "local_geotiff"),
            "DSMCollection": scene.metadata.get("dsm_collection", "local_dsm"),
            "ElevationMinMSL": round(scene.z_min_m, 3),
            "ElevationMaxMSL": round(scene.z_max_m, 3),
            "HasRealNIR": scene.has_real_nir,
            "HasRealDSM": scene.has_real_dsm,
        },
    )

    speed_kmh = params.speed_mps * 3.6
    speed_kt = speed_kmh / 1.852
    mach = speed_kmh / 1225.06

    flight_el = ET.SubElement(root, "FlightParameters")
    _add_kv(
        flight_el,
        {
            "StartEastingM": round(params.start_x_m, 3),
            "StartNorthingM": round(params.start_y_m, 3),
            "StartOffsetFromOriginM": round(params.start_offset_from_origin_m, 3),
            "YawDeg": round(params.yaw_deg, 3),
            "SpeedMPS": round(params.speed_mps, 3),
            "SpeedKMH": round(speed_kmh, 3),
            "SpeedKTAS": round(speed_kt, 3),
            "MachNumberISA": round(mach, 4),
            "AltitudeAGLMeters": round(params.altitude_agl_m, 2),
            "AltitudeAGLFeet": round(params.altitude_agl_m * 3.28084, 1),
        },
    )

    cam_el = ET.SubElement(root, "CameraAndOptics")
    _add_kv(
        cam_el,
        {
            "SpectralMode": params.spectral_mode,
            "HorizontalFOVDeg": round(params.fov_deg, 3),
            "PitchDepressionDeg": round(params.pitch_deg, 3),
            "DistortionK1": round(params.k1, 6),
            "DistortionK2": round(params.k2, 6),
            "DistortionP1": round(params.p1, 6),
            "DistortionP2": round(params.p2, 6),
        },
    )

    env_el = ET.SubElement(root, "LightingAndAtmosphere")
    _add_kv(
        env_el,
        {
            "IsNight": params.is_night,
            "SunAzimuthDeg": round(params.sun_azimuth_deg, 3),
            "SunElevationDeg": round(params.sun_elevation_deg, 3),
            "ShadowStrength": round(params.shadow_strength, 4),
            "ColorTempKelvin": round(params.color_temp_k, 1),
            "WhiteBalanceTint": round(params.wb_tint, 4),
            "HazeEnabled": params.haze_enabled,
            "VisibilityKM": round(params.visibility_km, 3),
            "SunGlareEnabled": params.sun_glare_enabled,
            "GlareIntensity": round(params.glare_intensity, 4),
            "SpecularGlint": round(params.specular_glint, 4),
        },
    )
    if visibility_diagnostics:
        vis_el = ET.SubElement(env_el, "VisibilityModelDiagnostics")
        _add_kv(vis_el, visibility_diagnostics)

    art_el = ET.SubElement(root, "SensorAndArtifacts")
    _add_kv(
        art_el,
        {
            "MotionBlurEnabled": params.motion_blur_enabled,
            "ShutterSec": round(params.shutter_sec, 5),
            "RollingShutterStrength": round(params.rolling_shutter_strength, 4),
            "AutoExposureEnabled": params.auto_exposure_enabled,
            "AETargetLuma": round(params.ae_target_luma, 4),
            "AELagAlpha": round(params.ae_lag_alpha, 4),
            "DynamicRangeClipStrength": round(params.dr_clip_strength, 4),
            "SensorNoiseEnabled": params.sensor_noise_enabled,
            "ISONoise": round(params.iso_noise, 5),
            "ThermalFPN": round(params.thermal_fpn, 5),
            "HasLensCrack": params.has_lens_crack,
            "HasWaterDroplets": params.has_watermark_droplets,
        },
    )

    frames_el = ET.SubElement(root, "TelemetryFrames", attrib={"count": str(len(poses))})
    kf_map = keyframe_files or {}
    for pose in poses:
        attrs = {
            "index": str(pose.frame_idx),
            "timeSec": f"{pose.time_sec:.4f}",
        }
        if pose.frame_idx in kf_map:
            attrs["keyframePng"] = kf_map[pose.frame_idx]
        f_el = ET.SubElement(frames_el, "Frame", attrib=attrs)
        _add_kv(
            f_el,
            {
                "Lat": pose.lat,
                "Lon": pose.lon,
                "EastingM": pose.x_m,
                "NorthingM": pose.y_m,
                "AltitudeMSLM": pose.z_msl_m,
                "AltitudeAGLM": pose.z_agl_m,
                "GroundElevationMSLM": pose.ground_z_m,
                "YawDeg": pose.yaw_deg,
                "PitchDeg": pose.pitch_deg,
                "SpeedMPS": pose.speed_mps,
            },
        )

    ET.indent(root, space="  ")
    out_path = Path(output_xml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path
