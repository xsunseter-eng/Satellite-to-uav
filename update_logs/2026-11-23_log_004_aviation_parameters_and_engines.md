# Log 004: Aviation-Grounded Parameter Distributions, Meteorological Visibility Model & Multi-Engine GPU Rendering
- **Date:** 2026-11-23
- **Status:** Completed (30-Photo Multi-Engine & Spectral Review Ready)

## 1. Executive Summary
Finalized the aeronautical operating envelope, statistical sampling distributions, per-region output schema (`MP4` + lossless `PNG` keyframes every 10 frames + structured `XML` telemetry), and GPU-accelerated OpenGL (EGL) 3D DSM rendering architecture. All parameter ranges and statistical moments are justified using standard aeronautical conventions (ISA atmosphere, KTAS/Mach, AGL/FL, VFR/MVFR/IMC meteorological minima, and real-world EO/IR gimbal specifications).

---

## 2. Output Architecture & Resolution Stratification
- **Per-Region Directory Layout:** Each of the 6 confirmed land-focused regions (`R1–R6`) writes into a dedicated directory (`dataset/<REGION_ID>/footage_XXX/`) with `N = 100` planned sorties per region (`600` total sorties).
- **Sortie Temporal Parameters:** Fixed `10.0 s` duration at `30 FPS` (`300` frames per sortie), exporting:
  1. `footage_XXX.mp4` — H.264 (`yuv420p`) video stream.
  2. `frames/frame_000000.png ... frame_000290.png` — Lossless PNG keyframes sampled every `10` frames (`30` keyframes per sortie).
  3. `footage_XXX.xml` — Structured `UAVFootageMetadata` XML document containing region/scene provenance, render engine, flight kinematics, EO/IR optics, lighting/visibility diagnostics, artifact flags, and per-frame 6-DoF telemetry linked to extracted keyframes.
- **Departure Offset (`X`):** Maximum random departure radius equals each region's full confirmed geofence radius (`110–300 km`), enabled via on-demand corridor tile ingestion.
- **Equal Resolution Stratification (4K Excluded):** Output resolutions are stratified equally (`25%` each) across 4 standard 16:9 aviation/UAV EO-IR video formats:
  - `640×360` (`360p`) — Low-bandwidth tactical RF / LWIR thermal downlink
  - `854×480` (`480p`) — Standard-definition (SD) analog/digital UAV downlink
  - `1280×720` (`720p`) — High-definition (HD) tactical ISR video feed
  - `1920×1080` (`1080p`) — Full-HD stabilized EO/IR gimbal output

---

## 3. Aeronautical Justification of Statistical Distributions

### 3.1 True Airspeed (TAS): `[100 km/h – Mach 1.8]`
- **Model:** Truncated Log-Normal distribution on `[100.0, 2205.1] km/h` (ISA sea-level `Mach 1.0 = 1,225.06 km/h`, `Mach 1.8 = 2,205.1 km/h` / `1,191 KTAS` / `612.5 m/s`), parameterized with $\mu_{\ln} = 4.8243,\; \sigma_{\ln} = 0.65$.
- **Moments:** Arithmetic Mean = **`204.0 km/h`** (`110.2 KTAS` / `56.67 m/s`); Median = **`170.0 km/h`** (`91.8 KTAS`); 90th Percentile = `336 km/h` (`181 KTAS`).
- **Operational Justification:** The distribution is anchored with the **Bayraktar TB2** Medium-Altitude Long-Endurance (MALE) tactical UAS operational airspeed envelope (`70–120 KTAS` / `130–222 km/h`) as its central median/mean reference, spanning continuously from the upper boundary of commercial/amateur small UAS (sUAS) and lightweight fixed-wing drones (`100 km/h` / `54 KTAS`) up to the supersonic supercruise envelope of fifth-generation tactical fighter aircraft (**F-22 Raptor**, `Mach 1.8`). Using a right-skewed log-normal distribution preserves realistic sortie density around operational MALE/tactical UAS speeds while maintaining non-zero coverage of fast-mover jet kinematics.

### 3.2 Altitude (AGL): `[120 m – 15,240 m AGL]`
- **Model:** Truncated Log-Normal distribution on `[120.0, 15240.0] m AGL` with $\mu_{\ln} = 7.4046,\; \sigma_{\ln} = 0.95$.
- **Moments:** Median = **`1,631 m AGL`** (`~5,350 ft AGL`); Mean = **`2,400 m AGL`** (`~7,874 ft AGL`); 90th Percentile = **`5,316 m AGL`** (`~17,440 ft AGL`).
- **Operational Justification:**
  - **Lower Bound (`120 m` / `~400 ft AGL`):** Matches the standard civil regulatory ceiling for small/amateur UAS under FAA 14 CFR Part 107 and EASA Open Category operations.
  - **Central Envelope (`1,500–5,500 m` / `5,000–18,000 ft AGL`):** Aligns with tactical ISR patrol altitudes and the **Bayraktar TB2** nominal operational altitude (`18,000 ft` / `5,486 m`, max service ceiling `25,000 ft`).
  - **Upper Bound (`15,240 m` / `50,000 ft` / `FL500`):** Covers High-Altitude Long-Endurance (HALE) ISR platforms (**RQ-4 Global Hawk**) and high-altitude tactical fighter operations (**F-22 Raptor**).

### 3.3 Horizontal Field of View (FoV): `[24° – 90°]`
- **Model:** Truncated Normal distribution ($\mu = 58^\circ,\; \sigma = 16^\circ$, bounds `[24°, 90°]`).
- **Payload Comparison:**
  - `24°–45°`: Corresponds to wide and medium-wide situational awareness modes of stabilized airborne EO/IR turrets such as **ASELSAN CATS** (Bayraktar TB2), **L3Harris WESCAM MX-15 / MX-15D**, and **Lockheed Martin AN/AAQ-33 Sniper ATP** wide-FOV navigation/acquisition modes.
  - `55°–90°`: Corresponds to tactical multirotor and fixed-wing EO payloads (**DJI Zenmuse H20 / P1** `64°–82°`, **Skydio X10** `~70°–90°`, wide-angle cockpit/HUD situational optics).
  - Ultra-narrow spotter optics (`<20°`) are excluded to prevent excessive sub-pixel oversampling relative to `0.20–1.0 m` satellite/overhead source imagery.

### 3.4 Camera Depression (Pitch) Angle: `[-90° – -20°]`
- **Model:** Truncated Normal distribution ($\mu = -55^\circ,\; \sigma = 16^\circ$, bounds `[-90°, -20°]`).
- **Operational Justification:** Pure nadir (`-90°`) represents vertical photogrammetric mapping sorties, whereas operational ISR and target-acquisition UAV gimbals maintain a median oblique depression around `-55°` (`-45°` to `-65°` primary concentration) to balance slant-range standoff, 3D vertical structure visibility, and manageable atmospheric path attenuation (avoiding grazing horizon views `> -20°`).

### 3.5 Solar Elevation & Azimuth
- **Azimuth:** Uniform `[0°, 360°)` across all sortie headings and diurnal bearings.
- **Elevation:** Truncated Normal ($\mu = 42^\circ,\; \sigma = 18^\circ$, bounds `[8°, 78°]`), modeling realistic mid-latitude solar trajectories from golden-hour low-angle illumination (`8°–15°`, long DSM cast shadows) through mid-day sorties (`35°–55°`) up to summer solar noon (`75°`), paired with a nocturnal probability for night/NVG/thermal-inversion sorties.

---

## 4. Mathematical Meteorological Visibility & Haze Model
To replace arbitrary uniform haze sampling with physically consistent atmospheric conditions dependent on **region climatology**, **altitude AGL**, and a **stochastic weather chance parameter**, horizontal meteorological visibility $V_{\text{final}}\;[\text{km}]$ is computed as:

1. **Regional Climatological Surface Baseline ($V_0(r)$, $k_h(r)$, $p_{\text{adv}}(r)$):**
   - `R1_SE_US_LAND_CORRIDOR` (Humid subtropical): $V_0 = 12\text{ km},\; k_h = 1.35,\; p_{\text{adv}} = 0.14$ (radiation/coastal mist)
   - `R2_LAS_VEGAS_NV` (Arid urban basin): $V_0 = 20\text{ km},\; k_h = 0.85,\; p_{\text{adv}} = 0.06$
   - `R3_MOJAVE_DESERT_CA` (Arid desert): $V_0 = 24\text{ km},\; k_h = 0.75,\; p_{\text{adv}} = 0.09$ (stochastic dust events)
   - `R4_MOAB_CANYONLANDS_UT` (High dry plateau): $V_0 = 28\text{ km},\; k_h = 0.65,\; p_{\text{adv}} = 0.05$
   - `R5_CENTRAL_SPAIN_PNOA` (Semi-arid plateau): $V_0 = 19\text{ km},\; k_h = 0.90,\; p_{\text{adv}} = 0.08$ (periodic Calima dust haze)
   - `R6_FRANCE_PARIS_LOIRE_BASIN` (Temperate oceanic): $V_0 = 11\text{ km},\; k_h = 1.40,\; p_{\text{adv}} = 0.16$ (frequent fog/mist)

2. **Atmospheric Boundary Layer (ABL) Altitude Clearing:**
   Aerosol density decays exponentially above the surface boundary layer ($H_{\text{scale}} = 1,800\text{ m}$), yielding deterministic visibility:
   $$V_{\text{det}}(r, z) = V_0(r) \cdot \left(1 + 0.55 \cdot \left(1 - e^{-z_{\text{AGL}} / 1800}\right)\right)$$

3. **Stochastic Weather Perturbation (Chance Parameter):**
   - With regional probability $p_{\text{adv}}(r)$, an adverse MVFR/IMC event (fog, mist, or blowing dust) draws $V_{\text{final}} \sim \mathcal{U}(V_{\text{adv,min}}, V_{\text{adv,max}})$ (typically `0.8–4.5 km`).
   - Otherwise, a multiplicative log-normal meteorological factor $\epsilon \sim \text{LogNormal}(0, (0.40 \cdot k_h(r))^2)$ produces:
     $$V_{\text{final}} = \text{clip}\!\left(V_{\text{det}}(r, z) \cdot \epsilon,\; 0.8\text{ km},\; 40.0\text{ km}\right)$$

4. **Koschmieder Slant-Range Extinction:**
   The extinction coefficient $\beta = \frac{3.912}{V_{\text{final}} \cdot 1000}\;[\text{m}^{-1}]$ (wavelength-scaled by `0.65` in NIR and `0.35` in LWIR thermal) is integrated per pixel against the exact 3D DSM slant depth $d(u,v)$: $T(u,v) = \exp(-\beta \cdot d(u,v))$.

---

## 5. Render Engine Evaluation Architecture
Implemented three interchangeable 3D DSM render engines to benchmark quality and throughput prior to full dataset generation:
1. **`gpu_opengl_raymarch` (Primary Selected Engine):** Headless NVIDIA EGL OpenGL 3.30 GLSL fragment ray-marcher (`56` coarse + `8` binary-search steps per pixel) with native Brown-Conrady lens distortion, hardware mipmaps, and `16x` anisotropic texture filtering.
2. **`gpu_opengl_mesh` (Comparison Engine):** Headless EGL OpenGL 3.30 displaced 3D heightfield triangle mesh rasterizer (`384×384` grid, hardware Z-buffer, `16x` AF) with post-rasterization distortion remap.
3. **`cpu_raymarch` (Portable Fallback Engine):** Vectorized NumPy + OpenCV CPU ray-marcher with sub-grid depth upsampling.

## 6. Code & Test Artifacts Added
- `src/sat2uav/distributions.py`: Aviation statistical samplers & meteorological visibility model.
- `src/sat2uav/gpu_renderer.py`: `GPURaymarchRenderer` & `GPUMeshRenderer` (ModernGL + EGL).
- `src/sat2uav/xml_export.py`: Structured `UAVFootageMetadata` XML serializer.
- `scripts/generate_quality_review_30.py`: 30-photo multi-engine, multi-spectral, multi-region review & HTML gallery builder.
- `tests/test_pipeline.py`: Expanded to 5 test suites (all passing on GTX 1050 EGL + CPU).
