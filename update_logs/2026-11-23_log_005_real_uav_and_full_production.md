# Log 005: Real UAV Test Footage Ingestion (UAV-VisLoc China & KABR Kenya), Regions R7–R8 & 880-Video Production Pipeline
- **Date:** 2026-11-23
- **Status:** In Progress (Downloads & Full 8-Region × 110-Video Production Launched)

## 1. Real UAV Reference Footage Acquired (`real_footages/`)
Downloaded real-world UAV datasets with synchronized GPS + orientation/heading telemetry for cross-domain validation against synthetic satellite-to-UAV footage:
- **`real_footages/uav_visloc_china/` (UAV-VisLoc Folder 06 — Shiyan/Zhushan, Hubei, China):**
  - 344 real UAV frames (`06/drone/*.JPG`) paired with `06/06.csv` (`lat`, `lon`, `height`, `Omega`, `Kappa`, `Phi1`, `Phi2` heading angles).
  - Extracted GPS centroid: **`32.3602°N, 109.6466°E`** (mean altitude ~840 m).
- **`real_footages/kabr_kenya/` (KABR — Mpala Research Centre, Laikipia Plateau, Kenya):**
  - Real 4K/5.4K DJI video (`DJI_0049_trimmed.MP4`) paired with 30 FPS sidecar `DJI_0049.SRT` (per-frame GPS, shutter, ISO, color temperature) and full `kabr_telemetry.csv` (30 FPS `latitude_y`, `longitude_y`, `altitude (m)`, `compass_heading`, `pitch`, `roll`, `gimbal_heading`, `gimbal_pitch`, `gimbal_roll`).
  - Extracted GPS centroid: **`0.3839°N, 36.8806°E`**.

## 2. Two New Real-UAV-Paired Geographic Regions Added (`R7` & `R8`)
Expanded the confirmed region registry (`REGIONS_AND_PARAMS.json`) from 6 to **8 land-focused regions**:
- **`R7_CHINA_UAV_VISLOC`:** `32.3602°N, 109.6466°E`, **`r = 200 km`** (100% land — humid subtropical Qinling-Daba mountain valleys, Hubei, China). Imagery stack: ESRI World Imagery (~1 m RGB) fused with Sentinel-2 L2A `B08` (10 m real NIR) + Copernicus DEM GLO-30.
- **`R8_KENYA_KABR_MPALA`:** `0.3839°N, 36.8806°E`, **`r = 180 km`** (100% land — equatorial highland semi-arid acacia savanna, Laikipia Plateau, Kenya). Imagery stack: ESRI World Imagery (~1 m RGB) fused with Sentinel-2 L2A `B08` (10 m real NIR) + Copernicus DEM GLO-30.

## 3. Full Production Specification (8 Regions × 110 Videos = 880 Videos)
- **Per-Region Quota (`110` videos × `10.0 s` @ `30 FPS` = `300 frames/video`):**
  - **100 Clean Videos (`v000`–`v099`):** `has_lens_crack = False`, `has_watermark_droplets = False`.
  - **10 Defect Videos (`v100`–`v109`):** 4 lens-crack-only, 4 water-droplet-only, 2 combined lens-crack + water-droplet.
- **Per-Video Naming & Synchronized Directory Layout (`dataset/<REGION_ID>/`):**
  - `<VIDEO_NAME>.mp4` — H.264 video (4 equally stratified resolutions: `640×360`, `854×480`, `1280×720`, `1920×1080`, no 4K).
  - `<VIDEO_NAME>.xml` — Complete `UAVFootageMetadata` XML mapping all flight, camera, optical, spectral, lighting, meteorological visibility, sensor/artifact parameters, and 300-frame 6-DoF telemetry linked to screenshots.
  - `<VIDEO_NAME>/frame_000000.png ... frame_000290.png` — Exact same-named directory storing 30 lossless PNG screenshots captured every 10 frames.

## 4. Performance Engineering (`FastSceneGPURenderer`)
- Moved spectral radiance (all 6 models), day/night relighting, DSM Lambertian hillshading, Koschmieder 3D haze (`#12`), Blinn-Phong specular glint + lens flare (`#15`), and Kelvin white balance (`#16`) directly into a single GLSL 3.30 fragment pass (`src/sat2uav/fast_gpu_renderer.py`), reducing GPU readback to RGB8 (**~7.6 ms/frame** on NVIDIA GTX 1050).
- Precomputed rolling-shutter remap grids and a cyclic Gaussian noise bank in `src/sat2uav/effects.py`, plus VRAM texture reuse across 22 videos per corridor via `scripts/produce_all_datasets.py` (fully resumable).
