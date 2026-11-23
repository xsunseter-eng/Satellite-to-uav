# Satellite-to-UAV Synthetic Flight Video Generator

A configurable pipeline for generating synthetic Unmanned Aerial Vehicle (UAV) flight videos from standardized public satellite and high-resolution overhead geospatial datasets (RGB + Near-Infrared / multispectral imagery paired with 3D Digital Surface Models).

---

## Project Purpose

Generate `N` randomized, ground-truth-annotated UAV flight videos around a target geographic origin point across diverse visual and geometric difficulty regimes (deserts, steep canyons, dense urban skylines, industrial river corridors, and arid plateaus) to support aerial perception, cross-view geo-localization, and visual navigation research.

### Core Generation Capabilities
1. **Standardized Origin-Centered Input:** Consumes georeferenced overhead imagery (GeoTIFF/COG, RGB+NIR/multispectral) + co-registered 3D elevation (DSM/DTM) within land-focused geofences (`100 km <= radius <= 300 km`), fetched/clipped on demand along flight corridors.
2. **Randomized Flight Initialization:**
   - Random starting position (up to `X` km from origin inside the geofence)
   - Random yaw heading (`0°–360°`)
   - Random flight speed from a bounded range
   - Random flight altitude from a bounded range
3. **Configurable & Limitable Variations:**
   - **Baseline:** Sun angle, spectral camera models (RGB / Grayscale / NIR / Multispectral / simulated Thermal), FoV angle, camera pitch direction, and footage defects (lens cracks, water marks).
   - **Extended Sensor & Environment (v1):**
     - Motion blur & CMOS rolling shutter
     - Radial/tangential lens distortion (`k1, k2, p1, p2`)
     - Auto-exposure & dynamic range clipping
     - Sensor noise (optical ISO/shot noise & thermal FPN)
     - Atmospheric haze, fog & visibility reduction
     - Sun glare, lens flare & specular glint
     - Color temperature & white balance shift
     - Variable video output resolution (e.g., `480p`, `720p`, `1080p`, `4K`)

---

## Target Regions & Geofences

All regions enforce a **100–300 km land-focused radius** with public sub-meter RGB+NIR imagery and high-resolution LiDAR DSMs:

| ID | Region | Status | Origin (Lat, Lon) | Radius | Terrain / Difficulty | Open Data Stack |
| :-: | :--- | :---: | :--- | :-: | :--- | :--- |
| **R1** | South Georgia / North Florida *(Jacksonville Land Corridor, USA)* | Confirmed | `31.1500°N, 82.8500°W` | 120 km | Canopy, wetlands, bridges, metro edge | NAIP + USGS 3DEP + US3D |
| **R2** | Las Vegas, NV (USA) | Confirmed | `36.1699°N, 115.1398°W` | 120 km | Urban-to-desert edge, high-rise, night lights | NAIP + USGS 3DEP + SpaceNet 2 + SDGSAT-1 |
| **R3** | West/Central Mojave Desert, CA (USA) | Confirmed | `35.0200°N, 117.1800°W` | 110 km | Feature-sparse sand/scrub desert | NAIP + USGS 3DEP + ECOSTRESS |
| **R4** | Moab / Canyonlands, UT (USA) | Confirmed | `38.5733°N, 109.5498°W` | 300 km | Steep canyons, mesas, high relief | NAIP + USGS 3DEP + ECOSTRESS |
| **R5** | Central Spain: Madrid–Toledo–La Mancha (Spain) | Proposed (Non-US) | `39.9500°N, 3.6500°W` | 200 km | Arid plateau, low-contrast urban + metro | **IGN PNOA** (RGB+NIR + LiDAR) + SDGSAT-1 |
| **R6** | Paris–Loire Basin: Orléans–Paris (France) | Proposed (Non-US) | `47.9000°N, 2.0000°E` | 180 km | Dense EU urban, Seine industrial ports, rural | **IGN BD ORTHO** + **LiDAR HD** + SDGSAT-1 |

---

## Repository Contents

- [`PROJECT_PURPOSE.txt`](PROJECT_PURPOSE.txt) — Full project mission, generation requirements, and variation specification.
- [`SATELLITE_DATASETS.md`](SATELLITE_DATASETS.md) — Feasibility analysis of public satellite/overhead datasets (depth, multispectral, day/night, high-res) and ingestion strategy.
- [`REGIONS_AND_PARAMS.json`](REGIONS_AND_PARAMS.json) — Machine-readable registry of geofenced regions and active variation parameters.
- [`update_logs/`](update_logs/) — Chronological project progress and decision logs.
