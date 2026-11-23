# Log 001: Initial Setup, Dataset Research & Project Scope
- **Date:** 2026-11-23
- **Status:** Completed

## Summary
Completed initial survey of publicly available satellite and overhead geospatial datasets against the four core requirements (depth, multispectral/IR, day/night variation, high resolution), documented the project's purpose and video generation specifications, and initialized the workspace structure.

## Changes Made
- Created `PROJECT_PURPOSE.txt` defining the core mission and requirements:
  1. Origin-centered batch generation of `N` UAV flight videos from a standardized satellite input file.
  2. Randomized flight initialization: starting position (up to `X` km from origin), yaw orientation, flight speed, and altitude within configurable ranges.
  3. Configurable/limitable variations: sun angle, spectral camera models, FoV angles, camera pitch directions, and synthetic footage degradations (cracks, water marks, noise, etc.).
  4. Multi-region origin selection covering diverse visual/geometric difficulty profiles (desert, rural, suburban, dense urban, mountainous/coastal).
- Created `SATELLITE_DATASETS.md` documenting feasible public datasets, hard physical/open-data gaps (no free sub-meter nighttime optical or sub-meter thermal satellite imagery), and a multi-source fusion + synthetic relighting strategy.
- Created `update_logs/` directory with `README.md` template and this initial log entry.

## Key Decisions & Findings
- **True Satellite Benchmarks (3/4 criteria — Depth + 8-band VNIR + High-Res, Daytime):** `DFC2019/US3D`, `IARPA MVS3DM / CORE3D`, and `SpaceNet`.
- **Continental Diversity Proxy:** `USDA NAIP` (0.3–0.6 m RGB+NIR) + `USGS 3DEP` (<1 m LiDAR DSM) provides free high-res coverage across every target difficulty regime (desert, urban, suburban, forest, mountain).
- **Day/Night & Thermal:** Must be handled via coarse public priors (`SDGSAT-1`, `ECOSTRESS`) combined with DSM-based relighting / synthetic night & thermal rendering.

## Next Steps
- [ ] Choose initial target regions across difficulty tiers (e.g., desert, dense urban, suburban, rural) and decide between pure satellite (`US3D`/`MVS3DM`) vs. `NAIP + 3DEP` for the first prototype.
- [ ] Define the exact standardized input bundle schema and video generation config schema (`YAML`/`JSON`).
- [ ] Build data ingestion scripts to fetch/normalize a sample region into the standardized GeoTIFF/COG + DSM format.
- [ ] Prototype the 3D/DSM camera renderer and randomized UAV trajectory generator.
