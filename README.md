# Satellite-to-UAV Synthetic Flight Video Generator

A configurable pipeline for generating synthetic Unmanned Aerial Vehicle (UAV) flight videos from standardized public satellite and high-resolution overhead geospatial datasets (RGB + Near-Infrared / multispectral imagery paired with 3D Digital Surface Models).

---

## Project Purpose

Generate `N` randomized, ground-truth-annotated UAV flight videos around a target geographic origin point across diverse visual and geometric difficulty regimes (deserts, steep canyons, dense urban skylines, industrial river corridors, and arid plateaus) to support aerial perception, cross-view geo-localization, and visual navigation research.

### Core Generation Capabilities
1. **Per-Region Standardized Output Layout:** Each confirmed region (`R1–R6`) outputs into its own directory (`dataset/<REGION_ID>/footage_XXX/`) for `N = 100` sorties (`10.0 s` @ `30 FPS`), generating:
   - `footage_XXX.mp4` — H.264 video stream across 4 equally stratified resolutions (`360p`, `480p`, `720p`, `1080p`; 25% each, 4K excluded).
   - `frames/frame_000000.png ...` — Lossless PNG keyframes sampled every 10 frames (`30` PNGs per sortie).
   - `footage_XXX.xml` — Structured `UAVFootageMetadata` XML document with scene provenance, sensor/atmosphere diagnostics, and per-frame 6-DoF pose telemetry.
2. **Aeronautical Parameter Distributions (Standard Aviation Justification):**
   - **Departure Radius (`X`):** Full confirmed region geofence radius (`110–300 km`), fetched on demand via corridor clipping.
   - **True Airspeed (`100 km/h – Mach 1.8` ISA MSL):** Truncated Log-Normal ($\mu_{\ln}=4.8243,\; \sigma_{\ln}=0.65$) yielding **mean = `204.0 km/h` (110.2 KTAS)** and **median = `170.0 km/h` (91.8 KTAS)** — anchored on the **Bayraktar TB2** MALE UAS operational envelope (`70–120 KTAS`), spanning continuously from small/amateur sUAS (`100 km/h`) up to **F-22 Raptor** supercruise (`Mach 1.8`).
   - **Altitude AGL (`120 m – 15,240 m`):** Truncated Log-Normal ($\mu_{\ln}=7.4046,\; \sigma_{\ln}=0.95$; median `1,631 m` / `~5,350 ft`, P90 `5,316 m` / `~17,440 ft`), spanning FAA/EASA sUAS ceiling (`400 ft AGL`) through **Bayraktar TB2** operational altitudes (`~18,000 ft`) up to HALE/fighter flight levels (`FL500`).
   - **Horizontal FoV (`24°–90°`, $\mu=58^\circ, \sigma=16^\circ$):** Calibrated against airborne EO/IR turrets (**ASELSAN CATS**, **WESCAM MX-15**, **Sniper ATP** wide modes `24°–45°`) and tactical sUAS EO optics (**DJI Zenmuse / Skydio** `60°–90°`).
   - **Camera Depression / Pitch (`-90° to -20°`, $\mu=-55^\circ, \sigma=16^\circ$):** Centered on operational ISR oblique depression while retaining nadir mapping coverage.
   - **Solar Elevation (`8°–78°`, $\mu=42^\circ, \sigma=18^\circ$)** & uniform azimuth `[0°, 360°)` + nocturnal sortie probability.
   - **Mathematical Visibility & Haze Model (`#12`):** Combines Köppen-Geiger regional climatological visibility $V_0(r)$, Atmospheric Boundary Layer exponential altitude clearing ($H_{\text{scale}} = 1,800\text{ m}$), and stochastic log-normal + adverse fog/dust chance events, integrated via Koschmieder's law along per-pixel 3D slant depth.
3. **Multi-Engine 3D DSM Rendering & Sensor Effects:**
   - Primary GPU engine: **`gpu_opengl_raymarch`** (Headless NVIDIA EGL OpenGL 3.30 3D DSM ray-marcher with 16× anisotropic mipmapping and native Brown-Conrady lens distortion `#7`), plus **`gpu_opengl_mesh`** (3D triangle mesh rasterizer) and **`cpu_raymarch`** fallback.
   - 6 spectral camera models (`rgb`, `panchromatic`, `nir`, `false_color_cir`, `thermal_lwir`, `night_vision`), plus motion blur & rolling shutter (`#6`), auto-exposure & DR clipping (`#8`), optical ISO & thermal FPN noise (`#9`), sun glare & specular glint (`#15`), Kelvin color temperature & WB (`#16`), lens cracks, and water droplets.

*(Full aviation derivations and equations are documented in [`update_logs/2026-11-23_log_004_aviation_parameters_and_engines.md`](update_logs/2026-11-23_log_004_aviation_parameters_and_engines.md).)*

---

## Target Regions & Geofences

All regions enforce a **100–300 km land-focused radius** with public sub-meter RGB+NIR imagery and high-resolution LiDAR DSMs:

| ID | Region | Status | Origin (Lat, Lon) | Radius | Terrain / Difficulty | Open Data Stack |
| :-: | :--- | :---: | :--- | :-: | :--- | :--- |
| **R1** | South Georgia / North Florida *(Jacksonville Land Corridor, USA)* | Confirmed | `31.1500°N, 82.8500°W` | 120 km | Canopy, wetlands, bridges, metro edge | NAIP + USGS 3DEP + US3D |
| **R2** | Las Vegas, NV (USA) | Confirmed | `36.1699°N, 115.1398°W` | 120 km | Urban-to-desert edge, high-rise, night lights | NAIP + USGS 3DEP + SpaceNet 2 + SDGSAT-1 |
| **R3** | West/Central Mojave Desert, CA (USA) | Confirmed | `35.0200°N, 117.1800°W` | 110 km | Feature-sparse sand/scrub desert | NAIP + USGS 3DEP + ECOSTRESS |
| **R4** | Moab / Canyonlands, UT (USA) | Confirmed | `38.5733°N, 109.5498°W` | 300 km | Steep canyons, mesas, high relief | NAIP + USGS 3DEP + ECOSTRESS |
| **R5** | Central Spain: Madrid–Toledo–La Mancha (Spain) | Confirmed | `39.9500°N, 3.6500°W` | 200 km | Arid plateau, low-contrast urban + metro | **IGN PNOA** (RGB+NIR + LiDAR) + SDGSAT-1 |
| **R6** | Paris–Loire Basin: Orléans–Paris (France) | Confirmed | `47.9000°N, 2.0000°E` | 180 km | Dense EU urban, Seine industrial ports, rural | **IGN BD ORTHO** + **LiDAR HD** + SDGSAT-1 |
| **R7** | Shiyan / Zhushan Valleys, Hubei (China — *UAV-VisLoc*) | Confirmed (Real-UAV Test) | `32.3602°N, 109.6466°E` | 200 km | Subtropical mountain valleys, terraced farms | ESRI World Imagery + Sentinel-2 NIR + Copernicus GLO-30 |
| **R8** | Mpala Research Centre, Laikipia (Kenya — *KABR*) | Confirmed (Real-UAV Test) | `0.3839°N, 36.8806°E` | 180 km | Equatorial highland acacia savanna | ESRI World Imagery + Sentinel-2 NIR + Copernicus GLO-30 |

---

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
*(Requires `ffmpeg` installed on your system `PATH`.)*

### 2. List Confirmed Regions
```bash
sat2uav list-regions
```

### 3. Fetch Real Public Satellite / Overhead + 3D DSM Data & Generate UAV Videos
Fetch a corridor on demand from any confirmed region (`R1`–`R6`) and generate `N` randomized UAV videos:
```bash
sat2uav run-region \
  --region R4_MOAB_CANYONLANDS_UT \
  --out outputs/moab_demo \
  --config configs/default_generation.yaml \
  --num-videos 4 \
  --duration 5.0 \
  --size-km 2.5 \
  --gsd 1.0
```

Or generate directly from any standardized local GeoTIFF + DSM bundle (`imagery.tif`, `dsm.tif`, `metadata.json`):
```bash
sat2uav generate \
  --scene data/my_scene \
  --out outputs/my_run \
  --config configs/default_generation.yaml \
  --num-videos 5
```

---

## Repository Contents

- [`src/sat2uav/`](src/sat2uav/) — Core Python package (on-demand STAC/WMS fetcher, 3D DSM ray-marching camera renderer, 6 spectral camera models, all optical/atmospheric/sensor effects, FFmpeg video writer, and CLI).
- [`configs/default_generation.yaml`](configs/default_generation.yaml) — Configurable parameter ranges and toggles for batch UAV generation.
- [`tests/test_pipeline.py`](tests/test_pipeline.py) — Automated test suite.
- [`PROJECT_PURPOSE.txt`](PROJECT_PURPOSE.txt) — Full project mission, generation requirements, and variation specification.
- [`SATELLITE_DATASETS.md`](SATELLITE_DATASETS.md) — Feasibility analysis of public satellite/overhead datasets (depth, multispectral, day/night, high-res) and ingestion strategy.
- [`REGIONS_AND_PARAMS.json`](REGIONS_AND_PARAMS.json) — Machine-readable registry of geofenced regions and active variation parameters.
- [`update_logs/`](update_logs/) — Chronological project progress and decision logs.
