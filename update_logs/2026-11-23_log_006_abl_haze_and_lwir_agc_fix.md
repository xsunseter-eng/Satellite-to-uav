# Update Log 006 — Exponential ABL Atmospheric Haze Integral & LWIR Thermal AGC Fix

**Date:** 2026-11-23  
**Author:** Antigravity Engineering  
**Scope:** Physical atmospheric scattering fix (`R1_v006` fog whiteout root cause), LWIR thermal dynamic range calibration, and MVFR operational visibility floor.

---

## 1. Root Cause Diagnosis (`R1_v006` Faulty Solid-Gray Render)

Quality inspection of `R1_v006_clean_rgb_day_1280x720.mp4` revealed an almost uniform solid-gray frame (`std < 3`) despite valid georeferenced NAIP imagery and 3DEP DSM. Telemetry inspection (`R1_v006_clean_rgb_day_1280x720.xml`) identified the exact physical bug:

- **Flight & Atmospheric State:** `AltitudeAGLMeters = 762.95 m`, `PitchDepressionDeg = -47.234°`, `VisibilityKM = 1.171 km` (`weather_state = adverse_fog_or_dust`), producing a slant optical path of $d \approx 1{,}040\text{ m}$.
- **Modeling Defect:** Both `FastSceneGPURenderer` (`src/sat2uav/fast_gpu_renderer.py`) and `EffectsPipeline._apply_haze` (`src/sat2uav/effects.py`) applied homogeneous surface-level Koschmieder extinction $\beta_0 = 3.912 / V_{\text{m}}$ uniformly along the entire slant ray:
  $$T = \exp\!\left(-\frac{3.912}{1171}\times 1040\right) = e^{-3.47} = 0.031$$
  yielding only **3.1% ground radiance** and **96.9% uniform airlight**. In real airborne EO/IR operations, surface radiation fog and aerosols decay exponentially with altitude above ground level (AGL) rather than extending at full surface density up to cruise altitude.
- **Secondary Audit Finding (`R1_v004`):** Over homogeneous rural pine forest at night, simulated `thermal_lwir` brightness compressed into a narrow grayscale band (`std = 5.4`) without thermal core Automatic Gain Control (AGC) dynamic range expansion.

---

## 2. Engineering Fixes Implemented

1. **Vertical Exponential ABL Optical Depth Integral** ([`fast_gpu_renderer.py`](../src/sat2uav/fast_gpu_renderer.py), [`effects.py`](../src/sat2uav/effects.py)):
   - Modeled surface extinction decay $\beta(z) = \beta_0 e^{-z / H_{\text{layer}}}$ where scale height transitions smoothly from shallow radiation fog ($H \approx 220\text{–}300\text{ m}$) to normal boundary-layer aerosol ($H = 1000\text{ m}$):
     $$d_{\text{eff}} = d \cdot \frac{H_{\text{layer}}}{\Delta z}\left(1 - e^{-\Delta z / H_{\text{layer}}}\right)$$
   - Enforced an operational MVFR optical depth ceiling $\tau = \min(\beta_0 d_{\text{eff}}, 1.05)$ ($T \ge 0.35$) so heavy fog/haze remains challenging for vision algorithms without degenerating into featureless zero-information whiteout frames.
   - Reduced `thermal_lwir` (8–14 $\mu$m) haze extinction multiplier from `0.35` to `0.18` to reflect superior long-wave IR haze penetration.
2. **MVFR Operational Visibility Floor** ([`distributions.py`](../src/sat2uav/distributions.py)):
   - Raised regional adverse meteorological visibility floors from `0.8–1.0 km` (zero-vis IFR) to `2.0–2.8 km` (realistic MVFR electro-optical UAV operating minimums).
3. **LWIR Thermal AGC Dynamic Range Expansion** ([`fast_gpu_renderer.py`](../src/sat2uav/fast_gpu_renderer.py), [`spectral.py`](../src/sat2uav/spectral.py)):
   - Expanded day/night LWIR scene contrast mapping to emulate onboard FLIR/CATS microbolometer AGC so roads, canopy gaps, and terrain boundaries remain distinguishable even over uniform rural forests.
4. **Regression Test** ([`tests/test_pipeline.py`](../tests/test_pipeline.py)):
   - Added `test_high_altitude_adverse_fog_never_washes_out` stressing `alt = 2500 m`, `pitch = -35°`, `visibility = 1.5 km` on `FastSceneGPURenderer` and asserting `frame.std() > 12.0`.

---

## 3. Verification Results

- Re-rendering the exact `R1_v006` pose (`alt = 763 m`, `pitch = -47.2°`) improved image contrast from `std < 3` (solid gray) to **`std = 19.0` (`p5=103, p95=167`)** with crisp visible buildings, roads, and realistic misty atmospheric veil.
- Re-rendering `R1_v004` (`thermal_lwir_night` over South Georgia forest) improved contrast from `std = 5.4` to **`std = 8.0`** with clear warm asphalt roads and plantation row boundaries.
- Prior `R1` videos generated with the homogeneous haze bug were cleared (preserving cached `_corridors/`), and `R1_SE_US_LAND_CORRIDOR` production was restarted in single-region mode (`--regions R1_SE_US_LAND_CORRIDOR`) per user request.
