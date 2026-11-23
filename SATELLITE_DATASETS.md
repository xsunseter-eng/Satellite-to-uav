# Public Satellite Imagery & Elevation Datasets

This document catalogs the publicly available satellite and nationwide high-resolution overhead geospatial datasets evaluated against the project's four requirements:

1. **Depth / 3D Elevation (DSM / LiDAR / Stereo)**
2. **Multispectral Vision (NIR, SWIR, Thermal IR)**
3. **Day / Night Variation**
4. **High Spatial Resolution (<1 m/px preferred)**

---

## 1. Feasibility Matrix & Hard Limitations

**No single public satellite dataset satisfies all 4 requirements simultaneously.**

| Requirement | Publicly Available? | High-Res (<1 m) at 100–300 km Radius? | Key Constraint |
| :--- | :--- | :--- | :--- |
| **1. Depth (DSM / 3D)** | Yes | Yes (via national open LiDAR: US 3DEP, Spain PNOA, France IGN LiDAR HD) | Global satellite DSMs (Copernicus GLO-30) are 30 m; pure satellite stereo benchmarks (US3D, MVS3DM) are only 4–15 km radius. |
| **2. Multispectral / IR** | Yes | **NIR:** Yes (via national 4-band RGB+NIR orthophotos) <br> **Thermal IR:** No | Global free NIR/SWIR (Sentinel-2) is 10–20 m; public Thermal IR is 30–100 m (SDGSAT-1, ECOSTRESS, ASTER). |
| **3. Day / Night Variation** | Partial | **No** (for optical & IR) | High-res optical satellites operate in daytime sun-synchronous orbits (~10:30 AM local time). Public night optical is 10–40 m (SDGSAT-1). |
| **4. High Resolution (<1 m)** | Yes | Yes (national open programs) / No (global continuous satellite) | Sub-meter open satellite archives (SpaceNet, US3D, Maxar Open Data) cover small city/event footprints (<15 km radius). |

### Explicitly Unavailable in Public Satellite Data
- **Sub-meter Nighttime Optical / NIR:** Passive optical sensors require sunlight; commercial nighttime systems (e.g., Jilin-1 ~1 m) are not publicly open at scale. Best open nighttime optical is **SDGSAT-1 GLI (10 m pan / 40 m RGB)**.
- **Sub-meter Thermal Infrared (LWIR/MWIR):** Sharpest open thermal satellite is **SDGSAT-1 TIS (30 m)**; commercial thermal (SatVu HotSat ~3.5 m) is restricted.

---

## 2. Nationwide Open High-Res + LiDAR Sources (Supports 100–300 km Radius)

Because all true sub-meter satellite benchmarks (US3D ~50 km², SpaceNet 2 Las Vegas ~216 km², SpaceNet 6 Rotterdam ~120 km², SpaceNet 2 Khartoum ~765 km²) have effective radii of only **4–15 km**, satisfying our **`100 km <= radius <= 300 km`** geofence policy at sub-meter resolution requires nationwide open 4-band (RGB+NIR) + LiDAR programs paired with satellite priors:

### 1. United States: USDA NAIP + USGS 3DEP (1 m LiDAR / S1M)
- **Imagery (NAIP):** **0.3–0.6 m**, **4-band (R, G, B, NIR)** across the entire contiguous US.
  - Access: AWS Open Data (`naip-analytic`, `naip-source`, `naip-visualization`) & Microsoft Planetary Computer STAC.
- **Depth (USGS 3DEP):** **<1 m airborne LiDAR** point clouds (`.LAZ` / Entwine Point Tiles) and Seamless 1-Meter DEMs (`S1M`).
  - Access: AWS (`s3://usgs-lidar`), USGS LidarExplorer, OpenTopography, Microsoft Planetary Computer.

### 2. Spain: IGN PNOA Orthophoto + PNOA-LiDAR *(Non-US Open Program)*
- **License:** Open Data, **CC BY 4.0** compatible (*Orden FOM/2807/2015*).
- **Imagery (PNOA):** **0.25–0.50 m** nationwide **4-band (R, G, B, NIR / IRC)** orthophotos (up to 10–15 cm in urban updates).
- **Depth (PNOA-LiDAR):** Nationwide classified `.LAZ` point clouds (0.5–5+ pts/m² across 1st–3rd coverages) + derived 0.5–2 m DTM/DSM grids.
- **Access:** CNIG Download Centre (`centrodedescargas.cnig.es`) & OGC WMS/WMTS services.

### 3. France: IGN BD ORTHO® + LiDAR HD *(Non-US Open Program)*
- **License:** Open Data, *Licence Ouverte / Etalab 2.0*.
- **Imagery (BD ORTHO®):** **0.20 m** nationwide **4-band (RGB + IRC/NIR)** orthoimagery on a 3-year rotational cycle.
- **Depth (LiDAR HD):** High-density nationwide LiDAR (**>=10 pts/m²**, >90% communal coverage complete) + **0.5 m DTM (MNT) / DSM (MNS) / Height Model (MNH)**.
- **Access:** IGN Géoservices (`geoservices.ign.fr`), `cartes.gouv.fr`, and `data.gouv.fr`.

---

## 3. Auxiliary & Core Satellite Datasets (Validation Cores, Thermal & Day/Night Priors)

### A. Sub-Meter True Satellite Validation Cores (Small Footprints, <15 km Radius)
Useful for validating domain transfer from aerial orthophotos to true orbital imagery inside overlapping regions:
- **DFC2019 / US3D:** WorldView-3 (0.35 m pan, 1.3 m 8-band VNIR) + 0.8 m LiDAR DSM (~50 km² Jacksonville, ~50 km² Omaha). IEEE DataPort / `pubgeo/dfc2019`.
- **IARPA MVS3DM & CORE3D:** WorldView-3 (0.3 m pan, 1.2 m 8-band VNIR, multi-date sun angles) + LiDAR DSM. AWS (`s3://spacenet-dataset/Hosted-Datasets/`).
- **SpaceNet (2, 4, 6):** WorldView-2/3 (0.3–0.5 m pansharpened 8-band VNIR); Las Vegas (~216 km²), Atlanta 27-angle MVOI, Rotterdam SAR+Optical (~120 km²). AWS (`s3://spacenet-dataset`).

### B. Day / Night & Thermal Infrared Satellites (Coarse Conditioning Priors)
| Dataset | Resolution | Modality | Day / Night | Access |
| :--- | :--- | :--- | :--- | :--- |
| **SDGSAT-1 (GLI + TIS)** | **10 m** pan / **40 m** RGB (night); **30 m** TIR | Nighttime Optical + 3-band Thermal IR | **Day & Night** | Open Science Portal (`sdgsat.ac.cn`) & EU GHSL |
| **ECOSTRESS (ISS)** | **70 m** | 5-band Thermal IR | **Day & Night** (variable diurnal overpass) | NASA Earthdata |
| **ASTER (Terra)** | **90 m** (TIR) | 5-band Thermal IR | **Day & Night** (targeted) | NASA / USGS EarthExplorer |
| **VIIRS DNB** | **~740 m** | Nighttime Radiance | Daily Global Night | NASA LAADS / EOG / GEE |

### C. Global Multispectral, Elevation & All-Weather SAR
- **Sentinel-2 (ESA):** 10 m (RGB + NIR), 20 m (Red-Edge + SWIR), 13 bands, global.
- **Landsat 8/9 (USGS/NASA):** 30 m (RGB + NIR + SWIR) + 100 m TIR, global.
- **Copernicus DEM GLO-30 / FABDEM:** 30 m global DSM / DTM.
- **Umbra & Capella Open Data:** 0.16–1.2 m X-band SAR (day/night all-weather radar backscatter, CC BY 4.0, AWS).

---

## 4. Target Regions & Geofence Registry (`100 km <= r <= 300 km`, Land-Focused)

### Confirmed US Regions (Land-Focused)
| ID | Region | Origin (Lat, Lon) | Radius | Difficulty / Terrain Profile | Primary Data Stack |
| :-: | :--- | :--- | :-: | :--- | :--- |
| **R1** | **South Georgia / North Florida** *(Jacksonville Land-Focused Corridor, USA)* | `31.1500°N, 82.8500°W` | **120 km** | Dense canopy/wetlands + highways + bridges + Jacksonville metro edge (~100% land) | NAIP + USGS 3DEP + US3D (core) + SDGSAT-1 / ECOSTRESS |
| **R2** | **Las Vegas, NV (USA)** | `36.1699°N, 115.1398°W` | **120 km** | Urban-to-desert transition, high-rises, strong night-light contrast | NAIP + USGS 3DEP + SpaceNet 2 (core) + SDGSAT-1 / ECOSTRESS |
| **R3** | **West/Central Mojave Desert, CA (USA)** | `35.0200°N, 117.1800°W` | **110 km** | Pure feature-sparse sand/scrub desert (disjoint from Las Vegas, ~225 km apart) | NAIP + USGS 3DEP + Sentinel-2 / ECOSTRESS |
| **R4** | **Moab / Canyonlands, UT (USA)** | `38.5733°N, 109.5498°W` | **300 km** | Steep red-rock canyons, mesas, high terrain relief (statewide 1 m LiDAR) | NAIP + USGS 3DEP + Sentinel-2 / ECOSTRESS |

### Confirmed Non-US Regions (Land-Focused)
| ID | Region | Origin (Lat, Lon) | Radius | Difficulty / Terrain Profile | Primary Data Stack |
| :-: | :--- | :--- | :-: | :--- | :--- |
| **R5** | **Central Spain: Madrid–Toledo–La Mancha** *(Replaces Khartoum)* | `39.9500°N, 3.6500°W` | **200 km** | Arid/semi-arid ochre plateau + low-contrast urban + dense Madrid metro (~100% Spanish land) | **Spain IGN PNOA** (0.25–0.5 m RGB+NIR + nationwide LiDAR) + Sentinel-2 / SDGSAT-1 / ECOSTRESS |
| **R6** | **Paris–Loire Basin: Orléans–Paris** *(Replaces Rotterdam)* | `47.9000°N, 2.0000°E` | **180 km** | Major European metropolis + Seine industrial river ports/rail hubs + farmland (~100% French land) | **France IGN BD ORTHO** (0.20 m RGB+NIR) + **IGN LiDAR HD** (0.5 m DSM/DTM) + Sentinel-2 / SDGSAT-1 |

### Rejected / Cancelled Candidate Regions (`radius < 100 km`)
- **Port of Rotterdam, Netherlands (Cancelled):** SpaceNet 6 is only ~6 km radius; Netherlands national land footprint cannot sustain a 100 km land-focused radius without clipping into the North Sea and crossing multiple national data boundaries.
- **Khartoum, Sudan (Cancelled):** SpaceNet 2 is only ~15 km radius; no open sub-meter imagery or LiDAR elevation data exists nationwide in Sudan.

---

## 5. On-Demand Ingestion Strategy (For 100–300 km Geofences)

Downloading an entire 150–300 km radius circle at 0.20–0.60 m resolution would exceed multiple terabytes per region. Instead, the pipeline should:
1. Define each region strictly by `(origin_lat, origin_lon, radius_km)` in `REGIONS_AND_PARAMS.json`.
2. Sample each randomized UAV flight trajectory inside the geofence first.
3. **On-demand fetch and clip** only the intersecting Cloud-Optimized GeoTIFF (COG) / WMTS / LiDAR tiles along that trajectory's ground footprint + look-ahead buffer into a standardized local scene bundle (`imagery.tif`, `dsm.tif`, `dtm.tif`, `night_prior.tif`, `metadata.json`).
