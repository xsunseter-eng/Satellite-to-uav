# `sat2uav` — Exhaustive Customization & Developer Guide

This handbook explains how to configure, extend, tune, and deploy the **Satellite-to-UAV Synthetic Flight Video Generator** (`sat2uav`) for custom geographic regions, sensor suites, atmospheric regimes, and machine-learning training pipelines.

---

## Table of Contents

1. [Configuration Reference (`configs/default_generation.yaml`)](#1-configuration-reference-configsdefault_generationyaml)
2. [Adding a New Geographic Region](#2-adding-a-new-geographic-region)
3. [Using Your Own Local GeoTIFF Imagery & DSM](#3-using-your-own-local-geotiff-imagery--dsm)
4. [Customizing Rendering Engines (GPU vs. CPU)](#4-customizing-rendering-engines-gpu-vs-cpu)
5. [Modifying Spectral Models & GLSL Shaders](#5-modifying-spectral-models--glsl-shaders)
6. [Tuning Atmospheric Scattering, Lighting & Sensor Defects](#6-tuning-atmospheric-scattering-lighting--sensor-defects)
7. [Customizing Batch Split (Clean vs. Defect Ratios)](#7-customizing-batch-split-clean-vs-defect-ratios)
8. [Parsing XML Telemetry for PyTorch / OpenCV Benchmarks](#8-parsing-xml-telemetry-for-pytorch--opencv-benchmarks)
9. [Headless Linux, Docker & Multi-GPU Scaling](#9-headless-linux-docker--multi-gpu-scaling)
10. [Troubleshooting & Common Pitfalls](#10-troubleshooting--common-pitfalls)

---

## 1. Configuration Reference (`configs/default_generation.yaml`)

All high-level parameters live in [`configs/default_generation.yaml`](../configs/default_generation.yaml) and are parsed by [`src/sat2uav/config.py`](../src/sat2uav/config.py). You can copy this file to create experiment-specific configs (e.g., `configs/low_altitude_suas.yaml`) and pass `--config configs/low_altitude_suas.yaml` to `sat2uav`.

### Top-Level Video Settings
```yaml
num_videos: 100
duration_sec: 10.0                  # Video duration in seconds
fps: 30                             # Frames per second (300 frames for 10.0 s)
keyframe_interval: 10               # Save a lossless PNG every N frames (30 PNGs/video)
keyframe_format: png                # png | jpg
render_engine: gpu_opengl_raymarch  # gpu_opengl_raymarch | gpu_opengl_mesh | cpu_raymarch
seed: 42                            # Deterministic base RNG seed
```

### Flight Envelope (`flight:`)
```yaml
flight:
  max_start_offset_km: null         # null = full region geofence radius (110–300 km)
  yaw_deg_range: [0.0, 360.0]       # True heading (0=North, 90=East, 180=South, 270=West)
  use_aviation_distributions: true  # true = Truncated Log-Normal (TB2/F-22/sUAS); false = uniform
  speed_mps_range: [27.78, 612.53]  # 100 km/h (27.78 m/s) to Mach 1.8 ISA MSL (612.53 m/s)
  altitude_agl_m_range: [120.0, 15240.0] # 400 ft sUAS ceiling to FL500
  boundary_margin_m: 120.0          # Minimum ground buffer from corridor edges
```

> **Tip — Simulating only low-altitude quadcopters (DJI/Skydio):**
> Set `use_aviation_distributions: false`, `speed_mps_range: [5.0, 22.0]` (18–80 km/h), and `altitude_agl_m_range: [40.0, 180.0]`.

### Camera, Optics & Spectral Mix (`camera:`)
```yaml
camera:
  equal_resolution_distribution: true  # 25% deterministic stratification across the list below
  resolutions:
    - [640, 360]
    - [854, 480]
    - [1280, 720]
    - [1920, 1080]
  fov_deg_range: [24.0, 90.0]          # Horizontal FoV (Truncated Normal mu=58, sigma=16)
  pitch_deg_range: [-90.0, -20.0]      # -90 = straight nadir, -20 = shallow oblique
  spectral_modes:
    - rgb
    - panchromatic
    - nir
    - false_color_cir
    - thermal_lwir
    - night_vision
  lens_distortion:                     # Brown-Conrady radial (k1, k2) & tangential (p1, p2)
    enabled: true
    k1_range: [-0.22, 0.12]
    k2_range: [-0.05, 0.05]
    p1_range: [-0.01, 0.01]
    p2_range: [-0.01, 0.01]
```

> **Tip — RGB-only dataset:** Simply remove the other modes under `spectral_modes:` and leave only `- rgb`. For `scripts/produce_all_datasets.py`, also edit `SPECTRAL_MODES` at line 35 of `scripts/produce_all_datasets.py`.

### Lighting, Atmosphere & Sensor Artifacts
```yaml
lighting:
  sun_azimuth_deg_range: [0.0, 360.0]
  sun_elevation_deg_range: [8.0, 78.0]
  night_probability: 0.20              # Fraction of non-NVG sorties rendered at night
  shadow_strength_range: [0.25, 0.70]  # Ray-marched 3D terrain cast shadow opacity
  color_temp_k_range: [3600.0, 8200.0] # Kelvin white-balance shift (#16)
  wb_tint_range: [-0.07, 0.07]

atmosphere:
  haze_enabled: true
  use_mathematical_visibility_model: true # Regional climatology + ABL + stochastic weather
  visibility_km_range: [2.0, 40.0]        # MVFR operational floor to crisp desert air
  sun_glare_enabled: true
  glare_intensity_range: [0.15, 0.70]
  specular_glint_range: [0.10, 0.50]

sensor_and_artifacts:
  motion_blur_enabled: true               # Velocity-coupled vertical motion blur (#6)
  shutter_sec_range: [0.002, 0.018]
  rolling_shutter_strength_range: [0.0, 0.60]
  auto_exposure_enabled: true             # Temporal AE lag & DR highlight clip (#8)
  sensor_noise_enabled: true              # Optical ISO shot noise + LWIR column/row FPN (#9)
  lens_crack_probability: 0.20            # Used by CLI `sat2uav generate`
  watermark_droplet_probability: 0.25
```

---

## 2. Adding a New Geographic Region

Adding a new region anywhere on Earth takes **2 file edits** (and optionally a custom imagery provider if you prefer a national orthophoto WMS over ESRI + Sentinel-2).

### Step 1: Register the Region in `REGIONS_AND_PARAMS.json`
Add a new entry under `"regions"` in [`REGIONS_AND_PARAMS.json`](../REGIONS_AND_PARAMS.json):

```json
"R9_ANATOLIA_PLATEAU_TR": {
  "name": "Central Anatolia Steppe & Valleys (Turkey)",
  "country": "Turkey",
  "status": "confirmed",
  "origin": {
    "lat": 39.9334,
    "lon": 32.8597
  },
  "geofence_radius_km": 180,
  "terrain_difficulty": "Semi-arid highland plateau, urban-to-steppe transition, rolling hills",
  "datasets": {
    "imagery": "ESRI World Imagery + Sentinel-2 B08 NIR",
    "dsm": "Copernicus DEM GLO-30"
  }
}
```

### Step 2: Add Regional Climatology in `src/sat2uav/distributions.py`
Add a `RegionClimatology` entry inside `REGION_CLIMATOLOGY` in [`src/sat2uav/distributions.py`](../src/sat2uav/distributions.py):

```python
"R9_ANATOLIA_PLATEAU_TR": RegionClimatology(
    base_visibility_km=21.0,      # Clear-sky surface VFR baseline V0 (km)
    volatility_factor=0.90,       # Weather log-normal volatility
    adverse_prob=0.08,            # Probability of mist/dust event
    adverse_range_km=(2.2, 4.8),  # MVFR visibility during adverse event
    biome_label="Semi-arid continental Anatolian highland",
),
```

### Step 3: Fetch & Generate Immediately!
For any country outside USA/Spain/France, [`src/sat2uav/geo_fetcher.py`](../src/sat2uav/geo_fetcher.py) automatically falls back to **ESRI World Imagery (sub-meter RGB) + Planetary Computer Sentinel-2 L2A Band 08 (real 10 m NIR) + Copernicus DEM GLO-30 (30 m 3D elevation)** — zero extra code needed!

```bash
python scripts/produce_all_datasets.py --regions R9_ANATOLIA_PLATEAU_TR --clean 100 --defect 10 --corridors 3
```

### Optional: Adding a National High-Res WMS / STAC Provider
Inside `fetch_scene_corridor()` in [`src/sat2uav/geo_fetcher.py`](../src/sat2uav/geo_fetcher.py), route your country to a custom fetcher (just like `"Spain"` → IGN PNOA WMS and `"France"` → IGN Géoplateforme BD ORTHO + IRC WMS):

```python
elif country_lower == "germany":
    # Example: DOP20 WMS in EPSG:3857 (cap tile size <= 2048px to respect WMS limits)
    rgb = _fetch_wms_tile(wms_url, layer="nw_dop_rgb", bbox_3857=..., width=2048, height=2048)
```

---

## 3. Using Your Own Local GeoTIFF Imagery & DSM

You do not need internet access if you already have aerial/satellite GeoTIFFs (e.g., drone orthomosaics, WorldView, Pleiades, or LiDAR DSMs).

### Standardized Scene Bundle Folder Layout
Place your files in a directory:
```text
my_custom_scene/
├── imagery.tif    # 3-band (R,G,B) or 4-band (R,G,B,NIR) GeoTIFF in a projected metric CRS (e.g. UTM)
├── dsm.tif        # Optional 1-band float32 elevation GeoTIFF (meters MSL); auto-reprojected to imagery grid
└── metadata.json  # Optional provenance JSON (region_id, country, origin_lat, origin_lon)
```

> **Note:** If `imagery.tif` is 3-band RGB only, `scene.py` automatically synthesizes a vegetation-sensitive NIR proxy so `nir`, `false_color_cir`, `thermal_lwir`, and `night_vision` continue to work seamlessly. If `dsm.tif` is omitted, a flat elevation plane is used.

Run generation directly:
```bash
sat2uav generate --scene my_custom_scene --out outputs/my_videos --config configs/default_generation.yaml --num-videos 25
```

---

## 4. Customizing Rendering Engines (GPU vs. CPU)

`sat2uav` provides **3 rendering engines** plus a persistent-VRAM batch fast path:

| Engine Name | File | Architecture | Speed | Best For |
| :--- | :--- | :--- | :---: | :--- |
| **`gpu_opengl_raymarch_fast`** *(default in batch runner)* | [`fast_gpu_renderer.py`](../src/sat2uav/fast_gpu_renderer.py) | All-in-shader GLSL 3.30 EGL (3D ray-march + spectral + haze + glare + WB on GPU; RGB8 readback) | **Fastest** (`~7.6 ms/f` GPU) | Large-scale production (`produce_all_datasets.py`) |
| **`gpu_opengl_raymarch`** | [`gpu_renderer.py`](../src/sat2uav/gpu_renderer.py) | ModernGL EGL MRT (returns full float32 depth, normals, rays to CPU) | Fast | Single-clip CLI & exact float32 depth-map export |
| **`gpu_opengl_mesh`** | [`gpu_renderer.py`](../src/sat2uav/gpu_renderer.py) | Hardware 3D triangle mesh rasterization over DSM heightfield | Fast | Steep mountain relief comparison |
| **`cpu_raymarch`** | [`renderer.py`](../src/sat2uav/renderer.py) | Vectorized NumPy coarse-to-fine ray-marcher | Slowest (no GPU) | Headless CPU-only CI runners / non-GPU machines |

### Switching Engines
- In `configs/default_generation.yaml`: set `render_engine: cpu_raymarch` (or `gpu_opengl_mesh`).
- On the CLI: pass `--engine cpu_raymarch` to `sat2uav run-region` or `sat2uav generate`.
- Ray-march quality vs. speed knobs: in `FastSceneGPURenderer(scene, coarse_steps=44, refine_steps=7)`, increase `coarse_steps=64, refine_steps=8` for ultra-steep canyon cliffs or reduce to `coarse_steps=32` for flat plains.

---

## 5. Modifying Spectral Models & GLSL Shaders

Spectral radiance is implemented identically in two places so GPU and CPU outputs stay synchronized:
- **GPU GLSL Shader:** `_ALL_IN_ONE_FRAG` in [`src/sat2uav/fast_gpu_renderer.py`](../src/sat2uav/fast_gpu_renderer.py) (lines 177–215)
- **CPU NumPy Reference:** `render_spectral_radiance()` in [`src/sat2uav/spectral.py`](../src/sat2uav/spectral.py)

### Example: Adding a "Black-Hot" Thermal Mode or Custom False-Color Palette
1. Add `"thermal_blackhot": 6` to `_SPECTRAL_ID` in `fast_gpu_renderer.py`.
2. Add an `else if (u_spectral_id == 6)` branch in `_ALL_IN_ONE_FRAG` (invert `1.0 - clamp(temp, 0.04, 0.96)`).
3. Mirror the branch in `spectral.py`.
4. Add `"thermal_blackhot"` to `camera.spectral_modes` in `configs/default_generation.yaml`.

### Tuning Thermal LWIR Contrast (AGC)
In both `fast_gpu_renderer.py` and `spectral.py`, daytime and nighttime thermal bands apply an Automatic Gain Control (AGC) contrast expansion:
```glsl
// Nighttime diurnal inversion (water & urban warm; dry soil/ridges cool)
temp = (temp - 0.34) * 1.75 + 0.46;
// Daytime solar heating (sunlit dark asphalt hot; vegetation & water cool)
temp = (temp - 0.38) * 1.55 + 0.48;
```
Increase the gain multiplier (`1.75` → `2.2`) for higher-contrast urban/rural thermal separation, or lower it (`1.3`) for raw unstretched radiometric look.

---

## 6. Tuning Atmospheric Scattering, Lighting & Sensor Defects

### Exponential ABL Haze / Fog Physics
In [`src/sat2uav/fast_gpu_renderer.py`](../src/sat2uav/fast_gpu_renderer.py) and [`src/sat2uav/effects.py`](../src/sat2uav/effects.py):
- **Surface extinction:** $\beta_0 = 3.912 / V_{\text{m}}$ (Koschmieder's law; scaled by `0.18` for LWIR and `0.65` for NIR).
- **Scale height:** $H_{\text{layer}} = 220 + 780 \cdot \mathrm{clip}\!\left(\frac{V_{\text{km}} - 1.5}{6.5}, 0, 1\right)$ (shallow ~250 m ground fog vs. 1000 m clear boundary-layer aerosol).
- **Effective optical path along slant ray $d$ across vertical drop $\Delta z = z_{\text{cam}} - z_{\text{hit}}$:**
  $$d_{\text{eff}} = d \cdot \frac{H_{\text{layer}}}{\Delta z}\left(1 - e^{-\Delta z / H_{\text{layer}}}\right), \qquad \tau = \min(\beta_0 \, d_{\text{eff}},\; 1.05)$$
- **Want denser fog?** Increase the optical depth clamp `1.05` → `1.6` (min transmittance drops from `35%` to `20%`).
- **Want zero atmospheric haze?** Set `atmosphere.haze_enabled: false` in `configs/default_generation.yaml`.

### Customizing Optical Defects (`src/sat2uav/effects.py`)
- **Lens Cracks (`_generate_lens_crack_overlay`):** Adjust `num_rays = rng.integers(4, 9)` (fracture spokes), `base_thick`, and fork probability (`0.28`).
- **Water Droplets (`_generate_water_droplets_overlay`):** Adjust `num_drops = rng.integers(8, 22)`, droplet radius bounds (`0.012–0.048 * scale`), and Gaussian defocus blur radius inside `_apply_water_droplets`.
- **Motion Blur & Rolling Shutter (`_apply_motion_blur_and_rolling_shutter`):** Motion blur kernel length is physically coupled to `pose.speed_mps`, `pose.z_agl_m`, `pose.pitch_deg`, `focal_px`, and `shutter_sec` (capped at `15 px`).

---

## 7. Customizing Batch Split (Clean vs. Defect Ratios)

[`scripts/produce_all_datasets.py`](../scripts/produce_all_datasets.py) defaults to **110 videos per region** (`100 clean + 10 defect`: 4 crack-only, 4 droplet-only, 2 crack+droplet):

```bash
# Example: 200 clean + 40 defect videos across 5 corridors per region
python scripts/produce_all_datasets.py \
  --clean 200 \
  --defect 40 \
  --corridors 5 \
  --out dataset_large
```

To adjust the defect breakdown (currently `40% crack / 40% droplet / 20% both`), edit lines 208–218 in `scripts/produce_all_datasets.py`.

---

## 8. Parsing XML Telemetry for PyTorch / OpenCV Benchmarks

Every video has a sibling `<VIDEO_NAME>.xml` containing exact per-frame 6-DoF camera poses synchronized with both the MP4 frame index and every 10th-frame lossless PNG.

```python
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2

def load_sortie_samples(xml_path: str | Path):
    """Parse a sat2uav XML file and return paired keyframe PNG paths + 6-DoF poses."""
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    region_dir = xml_path.parent

    meta = {
        "region_id": root.findtext("SceneProvenance/RegionId"),
        "crs": root.findtext("SceneProvenance/CRS"),
        "spectral_mode": root.findtext("CameraAndOptics/SpectralMode"),
        "fov_deg": float(root.findtext("CameraAndOptics/HorizontalFOVDeg")),
        "visibility_km": float(root.findtext("LightingAndAtmosphere/VisibilityKM")),
        "has_crack": root.findtext("SensorAndArtifacts/HasLensCrack") == "True",
        "has_droplets": root.findtext("SensorAndArtifacts/HasWaterDroplets") == "True",
    }

    keyframes = []
    for frame_el in root.findall("TelemetryFrames/Frame"):
        kf_rel = frame_el.attrib.get("keyframePng")
        if not kf_rel:
            continue
        keyframes.append({
            "frame_idx": int(frame_el.attrib["index"]),
            "time_sec": float(frame_el.attrib["timeSec"]),
            "image_path": region_dir / kf_rel,
            "lat": float(frame_el.findtext("Lat")),
            "lon": float(frame_el.findtext("Lon")),
            "easting_m": float(frame_el.findtext("EastingM")),
            "northing_m": float(frame_el.findtext("NorthingM")),
            "alt_msl_m": float(frame_el.findtext("AltitudeMSLM")),
            "alt_agl_m": float(frame_el.findtext("AltitudeAGLM")),
            "yaw_deg": float(frame_el.findtext("YawDeg")),
            "pitch_deg": float(frame_el.findtext("PitchDeg")),
        })
    return meta, keyframes

# Example usage:
meta, kfs = load_sortie_samples("dataset/R1_SE_US_LAND_CORRIDOR/R1_v018_clean_rgb_day_1280x720.xml")
print(meta["spectral_mode"], len(kfs), kfs[0]["lat"], kfs[0]["lon"], kfs[0]["yaw_deg"])
img_bgr = cv2.imread(str(kfs[0]["image_path"]))
```

---

## 9. Headless Linux, Docker & Multi-GPU Scaling

### Headless Linux Servers (No X11 Display Needed)
`FastSceneGPURenderer` creates a pure headless **EGL** context (`moderngl.create_standalone_context(backend="egl")`) — no `DISPLAY`, Xvfb, or desktop session is required.

On Ubuntu/Debian servers, ensure NVIDIA EGL libraries are installed:
```bash
sudo apt-get install -y libegl1 libgl1 ffmpeg
```

### Multi-GPU / Multi-Worker Parallel Production
Each process binds cleanly to a single GPU via `EGL_DEVICE_ID` or `CUDA_VISIBLE_DEVICES`:
```bash
# Example on a 2-GPU workstation (or 2 parallel workers on a single 12GB+ GPU):
CUDA_VISIBLE_DEVICES=0 python scripts/produce_all_datasets.py --regions R1_SE_US_LAND_CORRIDOR R2_LAS_VEGAS_NV R3_MOJAVE_DESERT_CA R4_MOAB_CANYONLANDS_UT &
CUDA_VISIBLE_DEVICES=1 python scripts/produce_all_datasets.py --regions R5_CENTRAL_SPAIN_PNOA R6_FRANCE_PARIS_LOIRE_BASIN R7_CHINA_UAV_VISLOC R8_KENYA_KABR_MPALA &
wait
```

### Reliable Satellite Fetching on Slow / Flaky Networks
Always export GDAL HTTP retry timeouts before long unattended runs:
```bash
export GDAL_HTTP_TIMEOUT=30
export GDAL_HTTP_MAX_RETRY=3
export GDAL_HTTP_RETRY_DELAY=2
```

---

## 10. Troubleshooting & Common Pitfalls

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `moderngl.Error: cannot create EGL context` | Missing NVIDIA/Mesa EGL driver or running in a container without `--gpus all` | Run with `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all`, or fall back to `--engine cpu_raymarch` |
| `HTTP 400` from national WMS (Spain/France) | Requesting a WMS tile larger than `2048×2048` or unsupported CRS | `geo_fetcher.py` caps WMS requests at `2048×2048` in `EPSG:3857` (WMS 1.3.0) and resamples to UTM |
| `ffmpeg: command not found` | FFmpeg binary missing from `PATH` | `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS) |
| Interrupted batch run leaves 1 partial video | Process killed mid-encode | Simply re-run the exact same command — `_is_video_complete()` detects any missing XML or `< 30` PNGs and cleanly re-renders only unfinished sorties |
| High-altitude flight clamped below requested AGL | Requested altitude's 3D frustum exceeds the `3 km × 3 km` corridor footprint | Increase `--size-km` (e.g., `5.0` or `8.0`) in `sat2uav run-region` or `size_km` in `produce_all_datasets.py` |
