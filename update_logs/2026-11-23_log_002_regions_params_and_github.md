# Log 002: Geofence Policy, Region Selection, Extended Parameters & GitHub Setup
- **Date:** 2026-11-23
- **Status:** Completed

## Summary
Applied the `100 km <= radius <= 300 km` land-focused geofence policy across candidate locations, cancelled regions failing the minimum coverage threshold, proposed two non-US open-data replacements (Spain IGN PNOA and France IGN BD ORTHO + LiDAR HD), locked in 8 extended variation parameters, and published the repository to GitHub.

## Changes Made
- Updated `PROJECT_PURPOSE.txt` with the geofence policy, confirmed/proposed regions, and 8 approved extended parameters.
- Updated `SATELLITE_DATASETS.md` with:
  - Nationwide open non-US high-res + LiDAR sources (**Spain IGN PNOA** and **France IGN BD ORTHO + LiDAR HD**).
  - Finalized land-focused origins & geofence radii for confirmed US regions (`R1–R4`) and proposed non-US regions (`R5–R6`).
  - Documented cancellations of Rotterdam (~6 km SpaceNet 6 / coastal border limits) and Khartoum (~15 km SpaceNet 2 / no national LiDAR).
  - On-demand corridor tile-clipping ingestion architecture.
- Created `REGIONS_AND_PARAMS.json` as the machine-readable source of truth for regions and parameters.
- Created `README.md` and `.gitignore` for the public GitHub repository.

## Key Decisions
1. **Land-Focused Jacksonville Corridor (`R1`):** Shifted origin inland to `31.1500°N, 82.8500°W` (`r = 120 km`) for ~100% land coverage while keeping Jacksonville metro in-bounds.
2. **Disjoint Southwest Deserts (`R2` & `R3`):** Shifted Mojave origin to `35.0200°N, 117.1800°W` (`r = 110 km`) and set Las Vegas (`36.1699°N, 115.1398°W`) to `r = 120 km` so the two geofences (~225 km apart) do not overlap.
3. **Non-US Replacements Proposed:**
   - `R5`: Central Spain (`39.9500°N, 3.6500°W`, `r = 200 km`) via IGN PNOA (arid/low-contrast urban replacement for Khartoum).
   - `R6`: Paris–Loire Basin, France (`47.9000°N, 2.0000°E`, `r = 180 km`) via IGN BD ORTHO + LiDAR HD (European industrial/river/urban replacement for Rotterdam).
4. **Approved Extended Parameters (8):** Motion blur & rolling shutter (#6), lens distortion (#7), auto-exposure & DR clipping (#8), sensor noise (#9), atmospheric haze/fog (#12), sun glare/specular glint (#15), color temperature & white balance (#16), and native video resolution variation only (#20, without digital zoom).

## Next Steps
- [ ] Confirm or adjust the two proposed non-US regions (`R5` Spain, `R6` France).
- [ ] Design the Python package architecture (`src/`) for on-demand COG/LiDAR corridor fetching and 3D DSM-based UAV camera rendering.
- [ ] Prototype trajectory generation and render the first test flight clip.
