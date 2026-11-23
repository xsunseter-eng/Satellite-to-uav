"""Aviation-grounded statistical distributions and meteorological visibility model.

All flight, optical, solar, and atmospheric parameters are grounded in standard
aeronautical operating envelopes spanning small UAS (sUAS), Medium-Altitude
Long-Endurance (MALE) ISR platforms (Bayraktar TB2 reference), and high-altitude
supersonic tactical aircraft (F-22 Raptor upper envelope).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import truncnorm

# ISA Sea-Level Speed of Sound: 340.294 m/s = 1225.06 km/h
ISA_MACH_1_KMH = 1225.06
SPEED_MIN_KMH = 100.0
SPEED_MAX_KMH = 1.8 * ISA_MACH_1_KMH  # 2205.1 km/h (Mach 1.8 ISA MSL)
# Solved numerically so truncated log-normal on [100, 2205.1] km/h has:
#   Mean   = 204.0 km/h (110.2 KTAS / 56.67 m/s)
#   Median = 170.0 km/h (91.8 KTAS, Bayraktar TB2 operational speed envelope)
SPEED_LOG_MU = 4.8243
SPEED_LOG_SIGMA = 0.65

# Altitude AGL bounds:
#   Lower : 120 m AGL (~400 ft, FAA Part 107 / EASA Open Category sUAS ceiling)
#   Upper : 15,240 m AGL (50,000 ft / FL500, HALE UAV / F-22 operational ceiling)
#   Median: 1,631 m AGL (~5,350 ft) | Mean: 2,400 m | P90: 5,316 m (~18,000 ft TB2 ops)
ALT_MIN_M = 120.0
ALT_MAX_M = 15240.0
ALT_LOG_MU = 7.4046
ALT_LOG_SIGMA = 0.95

# Standard 16:9 aviation/UAV EO-IR resolutions (no 4K; equally weighted 25% each)
STANDARD_RESOLUTIONS: list[tuple[int, int]] = [
    (640, 360),    # 360p — Tactical low-bandwidth / LWIR downlink
    (854, 480),    # 480p — SD EO/IR downlink
    (1280, 720),   # 720p — HD tactical ISR feed
    (1920, 1080),  # 1080p — Full HD stabilized gimbal output
]


@dataclass(frozen=True)
class RegionClimatology:
    """Climatological baseline for regional meteorological visibility modeling."""

    base_visibility_km: float  # Clear-sky surface VFR visibility V0 (km)
    volatility_factor: float   # Scales log-normal weather sigma (humidity/aerosol variance)
    adverse_prob: float        # Probability of IMC/MVFR fog, mist, or blowing dust event
    adverse_range_km: tuple[float, float]
    biome_label: str


REGION_CLIMATOLOGY: dict[str, RegionClimatology] = {
    "R1_SE_US_LAND_CORRIDOR": RegionClimatology(
        base_visibility_km=12.0,
        volatility_factor=1.35,
        adverse_prob=0.14,
        adverse_range_km=(0.9, 3.2),
        biome_label="Humid subtropical coastal/wetland (frequent morning mist & humidity haze)",
    ),
    "R2_LAS_VEGAS_NV": RegionClimatology(
        base_visibility_km=20.0,
        volatility_factor=0.85,
        adverse_prob=0.06,
        adverse_range_km=(2.0, 5.0),
        biome_label="Arid desert basin with urban aerosol layer",
    ),
    "R3_MOJAVE_DESERT_CA": RegionClimatology(
        base_visibility_km=24.0,
        volatility_factor=0.75,
        adverse_prob=0.09,
        adverse_range_km=(1.0, 3.5),
        biome_label="High-clarity arid desert with stochastic blowing dust/sand events",
    ),
    "R4_MOAB_CANYONLANDS_UT": RegionClimatology(
        base_visibility_km=28.0,
        volatility_factor=0.65,
        adverse_prob=0.05,
        adverse_range_km=(2.5, 6.0),
        biome_label="High-altitude dry Colorado Plateau (crisp mountain-desert air)",
    ),
    "R5_CENTRAL_SPAIN_PNOA": RegionClimatology(
        base_visibility_km=19.0,
        volatility_factor=0.90,
        adverse_prob=0.08,
        adverse_range_km=(1.5, 4.5),
        biome_label="Semi-arid Iberian plateau with periodic Saharan dust (Calima) haze",
    ),
    "R6_FRANCE_PARIS_LOIRE_BASIN": RegionClimatology(
        base_visibility_km=11.0,
        volatility_factor=1.40,
        adverse_prob=0.16,
        adverse_range_km=(0.8, 3.0),
        biome_label="Temperate oceanic basin (frequent radiation fog, mist, & urban haze)",
    ),
}

DEFAULT_CLIMATOLOGY = RegionClimatology(
    base_visibility_km=18.0,
    volatility_factor=1.0,
    adverse_prob=0.08,
    adverse_range_km=(1.5, 4.5),
    biome_label="Mid-latitude standard ISA atmosphere",
)


def _sample_truncnorm(
    rng: np.random.Generator, low: float, high: float, mu: float, sigma: float
) -> float:
    a = (low - mu) / sigma
    b = (high - mu) / sigma
    u = float(rng.uniform(1e-6, 1.0 - 1e-6))
    return float(truncnorm.ppf(u, a, b, loc=mu, scale=sigma))


def sample_airspeed_kmh(
    rng: np.random.Generator,
    min_kmh: float = SPEED_MIN_KMH,
    max_kmh: float = SPEED_MAX_KMH,
) -> float:
    """Sample true airspeed (km/h) from truncated log-normal [100 km/h, Mach 1.8].

    Mean = 204.0 km/h (110.2 KTAS); Median = 170.0 km/h (Bayraktar TB2 reference).
    """
    val_ln = _sample_truncnorm(
        rng, math.log(min_kmh), math.log(max_kmh), SPEED_LOG_MU, SPEED_LOG_SIGMA
    )
    return float(math.exp(val_ln))


def sample_altitude_agl_m(
    rng: np.random.Generator,
    min_m: float = ALT_MIN_M,
    max_m: float = ALT_MAX_M,
) -> float:
    """Sample altitude AGL (m) from truncated log-normal [120 m, 15,240 m].

    Spans FAA/EASA sUAS ceiling (400 ft) through Bayraktar TB2 MALE operational
    band (~5,000–18,000 ft) up to HALE/F-22 FL500 (50,000 ft).
    """
    val_ln = _sample_truncnorm(
        rng, math.log(min_m), math.log(max_m), ALT_LOG_MU, ALT_LOG_SIGMA
    )
    return float(math.exp(val_ln))


def sample_fov_deg(
    rng: np.random.Generator,
    min_deg: float = 24.0,
    max_deg: float = 90.0,
    mu_deg: float = 58.0,
    sigma_deg: float = 16.0,
) -> float:
    """Sample horizontal FoV (deg) from truncated normal anchored on EO/IR gimbal optics."""
    return _sample_truncnorm(rng, min_deg, max_deg, mu_deg, sigma_deg)


def sample_pitch_deg(
    rng: np.random.Generator,
    min_deg: float = -90.0,
    max_deg: float = -20.0,
    mu_deg: float = -55.0,
    sigma_deg: float = 16.0,
) -> float:
    """Sample camera depression/pitch angle (deg) from truncated normal around -55 deg."""
    return _sample_truncnorm(rng, min_deg, max_deg, mu_deg, sigma_deg)


def sample_sun_elevation_deg(
    rng: np.random.Generator,
    min_deg: float = 8.0,
    max_deg: float = 78.0,
    mu_deg: float = 42.0,
    sigma_deg: float = 18.0,
) -> float:
    """Sample solar elevation angle (deg) from truncated normal around mid-latitude arc."""
    return _sample_truncnorm(rng, min_deg, max_deg, mu_deg, sigma_deg)


def sample_visibility_km(
    region_id: str | None,
    altitude_agl_m: float,
    rng: np.random.Generator,
    h_scale_m: float = 1800.0,
) -> tuple[float, dict[str, Any]]:
    """Compute meteorological visibility (km) from region biome, altitude AGL, and chance.

    Mathematical Model:
        1. Regional surface baseline V_0(r) from Köppen-Geiger / METAR climatology.
        2. Atmospheric Boundary Layer (ABL) altitude clearing factor:
               V_det(r, z) = V_0(r) * (1 + 0.55 * (1 - exp(-z_AGL / H_scale)))
           where H_scale = 1800 m represents aerosol scale height.
        3. Stochastic weather chance parameter:
           - With probability p_adv(r), an adverse IMC/MVFR event (radiation fog,
             mist, or desert dust storm) draws directly from V_adv(r).
           - Otherwise, multiplicative log-normal weather perturbation:
               eps ~ LogNormal(0, (0.40 * k_h(r))^2)
               V_final = clip(V_det * eps, 0.8 km, 40.0 km)
    """
    clim = REGION_CLIMATOLOGY.get(region_id or "", DEFAULT_CLIMATOLOGY)
    abl_factor = 1.0 + 0.55 * (1.0 - math.exp(-max(altitude_agl_m, 0.0) / h_scale_m))
    v_det = clim.base_visibility_km * abl_factor

    is_adverse = bool(rng.random() < clim.adverse_prob)
    if is_adverse:
        v_final = float(rng.uniform(*clim.adverse_range_km))
        weather_state = "adverse_fog_or_dust"
        stochastic_factor = v_final / max(v_det, 1e-3)
    else:
        sigma_w = 0.40 * clim.volatility_factor
        stochastic_factor = float(math.exp(rng.normal(0.0, sigma_w)))
        v_final = float(np.clip(v_det * stochastic_factor, 0.8, 40.0))
        weather_state = "vfr_clear" if v_final >= 8.0 else "mvfr_haze"

    diag = {
        "region_base_v0_km": clim.base_visibility_km,
        "abl_altitude_factor": round(abl_factor, 4),
        "deterministic_v_km": round(v_det, 3),
        "stochastic_factor": round(stochastic_factor, 4),
        "weather_state": weather_state,
        "is_adverse_event": is_adverse,
        "final_visibility_km": round(v_final, 3),
    }
    return v_final, diag


def pick_equal_resolution(video_index: int, resolutions: list[tuple[int, int]] | None = None) -> tuple[int, int]:
    """Deterministically stratify resolutions equally across a batch (25% each for 4 resolutions)."""
    pool = resolutions or STANDARD_RESOLUTIONS
    return pool[video_index % len(pool)]
