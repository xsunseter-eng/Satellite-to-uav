"""Virtual camera spectral response models (RGB, Pan, NIR, CIR, Thermal LWIR, NVG)."""

from __future__ import annotations

import numpy as np


def render_spectral_radiance(
    rgba_nir: np.ndarray,
    dsm_z: np.ndarray,
    normals: np.ndarray,
    ndvi: np.ndarray,
    shadow: np.ndarray,
    sky_mask: np.ndarray,
    sun_vec: np.ndarray,
    spectral_mode: str,
    is_night: bool,
    shadow_strength: float,
    z_min_m: float,
    z_max_m: float,
) -> np.ndarray:
    """Convert sampled 4-band surface reflectance + geometry into a 3-channel
    float32 [0, 1] radiance image according to the selected virtual camera model.
    """
    r = rgba_nir[:, :, 0]
    g = rgba_nir[:, :, 1]
    b = rgba_nir[:, :, 2]
    nir = rgba_nir[:, :, 3]

    # Lambertian solar illumination factor from DSM surface normals & cast shadows
    ndotl = np.clip(
        normals[:, :, 0] * sun_vec[0]
        + normals[:, :, 1] * sun_vec[1]
        + normals[:, :, 2] * sun_vec[2],
        0.0,
        1.0,
    )
    direct = ndotl * (1.0 - shadow_strength * shadow)
    # Normalize relative to moderate ambient sky fill so sunlit areas preserve albedo
    sun_elev = float(np.clip(sun_vec[2], 0.12, 1.0))
    day_illum = np.clip(0.32 + 0.78 * (direct / (sun_elev ** 0.45)), 0.15, 1.25)

    # Urban / road / artificial bright surface proxy (high brightness, low vegetation NDVI)
    albedo = 0.299 * r + 0.587 * g + 0.114 * b
    urban_proxy = np.clip((albedo - 0.28) * 2.5, 0.0, 1.0) * np.clip(
        (0.18 - ndvi) * 3.0, 0.0, 1.0
    )

    if spectral_mode == "rgb":
        rgb = np.stack([r, g, b], axis=-1)
        if is_night:
            # Moonlit low-light cool tone + subtle warm artificial urban/road glow
            moon_illum = 0.11 + 0.14 * ndotl
            out = rgb * moon_illum[:, :, None] * np.array([0.62, 0.74, 1.05], dtype=np.float32)
            glow = (urban_proxy ** 1.6)[:, :, None] * np.array([0.85, 0.62, 0.28], dtype=np.float32)
            out = out + 0.55 * glow
        else:
            out = rgb * day_illum[:, :, None]

    elif spectral_mode == "panchromatic":
        pan = 0.26 * r + 0.50 * g + 0.12 * b + 0.12 * nir
        if is_night:
            val = pan * (0.14 + 0.12 * ndotl) + 0.45 * (urban_proxy ** 1.5)
        else:
            val = pan * day_illum
        out = np.repeat(val[:, :, None], 3, axis=-1)

    elif spectral_mode == "nir":
        if is_night:
            val = nir * (0.18 + 0.14 * ndotl) + 0.35 * (urban_proxy ** 1.5)
        else:
            val = nir * day_illum
        out = np.repeat(val[:, :, None], 3, axis=-1)

    elif spectral_mode == "false_color_cir":
        # Standard Color-Infrared (CIR): Red <- NIR, Green <- Red, Blue <- Green
        cir = np.stack([nir, r, g], axis=-1)
        illum = (0.16 + 0.14 * ndotl) if is_night else day_illum
        out = cir * illum[:, :, None]

    elif spectral_mode == "thermal_lwir":
        # Physics-inspired simulated Long-Wave Infrared (white-hot)
        z_norm = (dsm_z - z_min_m) / max(z_max_m - z_min_m, 10.0)
        water_proxy = np.clip((-ndvi - 0.15) * 2.5, 0.0, 1.0) * np.clip(
            (0.25 - albedo) * 4.0, 0.0, 1.0
        )
        veg_proxy = np.clip((ndvi - 0.15) * 2.2, 0.0, 1.0)

        if is_night:
            # Diurnal thermal inversion: water & concrete/urban retain heat; dry soil/high ridges cool
            temp = (
                0.34
                + 0.28 * water_proxy
                + 0.30 * urban_proxy
                + 0.10 * veg_proxy
                - 0.14 * z_norm
                - 0.12 * (1.0 - albedo) * (1.0 - urban_proxy)
            )
        else:
            # Daytime solar heating: dark sunlit rock/asphalt hot; water, shadows, & vegetation cool
            absorptance = 1.0 - 0.55 * albedo
            solar_heat = 0.42 * absorptance * direct
            temp = (
                0.28
                + solar_heat
                + 0.16 * urban_proxy
                - 0.24 * veg_proxy
                - 0.28 * water_proxy
                - 0.12 * z_norm
            )
        val = np.clip(temp, 0.04, 0.98)
        out = np.repeat(val[:, :, None], 3, axis=-1)

    elif spectral_mode == "night_vision":
        # Green-phosphor low-light NVG (heavily sensitive to NIR + urban light sources)
        nvg_sig = np.clip(
            0.55 * nir + 0.35 * albedo + 0.45 * (urban_proxy ** 1.4), 0.0, 1.0
        )
        nvg_sig = np.clip(nvg_sig * (0.55 + 0.45 * ndotl), 0.0, 1.0)
        out = np.stack(
            [0.14 * nvg_sig, 0.96 * nvg_sig, 0.26 * nvg_sig], axis=-1
        )
    else:
        out = np.stack([r, g, b], axis=-1) * day_illum[:, :, None]

    # Fill sky pixels (rays pointing above horizon)
    if np.any(sky_mask):
        sky_color = _sky_color(spectral_mode, is_night)
        out = np.where(sky_mask[:, :, None], sky_color[None, None, :], out)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _sky_color(spectral_mode: str, is_night: bool) -> np.ndarray:
    if spectral_mode == "thermal_lwir":
        return np.array([0.08, 0.08, 0.08], dtype=np.float32)
    if spectral_mode == "night_vision":
        return np.array([0.03, 0.14, 0.04], dtype=np.float32)
    if is_night:
        return np.array([0.03, 0.04, 0.08], dtype=np.float32)
    if spectral_mode in ("panchromatic", "nir"):
        return np.array([0.65, 0.65, 0.65], dtype=np.float32)
    if spectral_mode == "false_color_cir":
        return np.array([0.45, 0.62, 0.78], dtype=np.float32)
    return np.array([0.52, 0.70, 0.90], dtype=np.float32)
