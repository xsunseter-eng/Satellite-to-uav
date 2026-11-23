"""Standardized georeferenced satellite/overhead scene representation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject


@dataclass
class SceneBundle:
    """In-memory metric-projected satellite imagery + 3D DSM scene."""

    imagery: np.ndarray  # (H, W, 4) float32 in [0, 1] -> [R, G, B, NIR]
    dsm: np.ndarray  # (H, W) float32 elevation in meters (MSL)
    normals: np.ndarray  # (H, W, 3) float32 unit normals (East, North, Up)
    ndvi: np.ndarray  # (H, W) float32 in [-1, 1]
    gsd_m: float  # Ground Sample Distance in meters/pixel
    crs: str  # Projected metric CRS string (e.g., EPSG:32612)
    origin_x_m: float  # Scene origin Easting in meters
    origin_y_m: float  # Scene origin Northing in meters
    x_min_m: float  # Raster west bound (m)
    y_min_m: float  # Raster south bound (m)
    x_max_m: float  # Raster east bound (m)
    y_max_m: float  # Raster north bound (m)
    z_min_m: float
    z_max_m: float
    has_real_nir: bool
    has_real_dsm: bool
    metadata: dict[str, Any]

    @property
    def height_px(self) -> int:
        return int(self.dsm.shape[0])

    @property
    def width_px(self) -> int:
        return int(self.dsm.shape[1])

    def world_to_pixel(self, x_m: np.ndarray | float, y_m: np.ndarray | float):
        """Convert metric ENU coordinates (x=East, y=North) to raster (col, row)."""
        col = (x_m - self.x_min_m) / self.gsd_m
        row = (self.y_max_m - y_m) / self.gsd_m
        return col, row

    def sample_elevation(self, x_m: float, y_m: float) -> float:
        """Bilinearly sample DSM elevation (meters MSL) at a single world point."""
        col, row = self.world_to_pixel(x_m, y_m)
        c = float(np.clip(col, 0.0, self.width_px - 1.001))
        r = float(np.clip(row, 0.0, self.height_px - 1.001))
        c0, r0 = int(c), int(r)
        dc, dr = c - c0, r - r0
        z00 = self.dsm[r0, c0]
        z10 = self.dsm[r0, c0 + 1]
        z01 = self.dsm[r0 + 1, c0]
        z11 = self.dsm[r0 + 1, c0 + 1]
        return float(
            (1 - dr) * ((1 - dc) * z00 + dc * z10)
            + dr * ((1 - dc) * z01 + dc * z11)
        )

    def to_latlon(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Convert projected scene metric coordinates to WGS84 (lat, lon)."""
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x_m, y_m)
        return float(lat), float(lon)

    def compute_shadow_map(
        self,
        sun_azimuth_deg: float,
        sun_elevation_deg: float,
        max_steps: int = 80,
    ) -> np.ndarray:
        """Compute DSM ray-casted cast-shadow mask in [0, 1] (1 = in shadow)."""
        if not self.has_real_dsm or (self.z_max_m - self.z_min_m) < 0.5:
            return np.zeros((self.height_px, self.width_px), dtype=np.float32)

        # Compute on a slightly downsampled grid for speed, then upsample smoothly
        scale = 2 if max(self.height_px, self.width_px) > 1200 else 1
        if scale > 1:
            dsm_s = cv2.resize(
                self.dsm,
                (self.width_px // scale, self.height_px // scale),
                interpolation=cv2.INTER_AREA,
            )
        else:
            dsm_s = self.dsm

        h_s, w_s = dsm_s.shape
        gsd_s = self.gsd_m * scale

        az = np.deg2rad(sun_azimuth_deg)
        el = np.deg2rad(np.clip(sun_elevation_deg, 5.0, 89.0))
        # Direction TOWARD the sun in pixel (col=+East, row=-North) space
        d_col = np.sin(az)
        d_row = -np.cos(az)
        dz_per_px = np.tan(el) * gsd_s

        max_relief = float(np.percentile(dsm_s, 99.5) - np.percentile(dsm_s, 0.5))
        max_dist_px = min(max_steps * 2.0, max(12.0, max_relief / max(dz_per_px, 1e-4)))
        step_px = max(1.0, max_dist_px / max_steps)

        rows = np.arange(h_s, dtype=np.float32)[:, None]
        cols = np.arange(w_s, dtype=np.float32)[None, :]
        in_shadow = np.zeros((h_s, w_s), dtype=np.float32)

        for i in range(1, max_steps + 1):
            dist = i * step_px
            sc = cols + d_col * dist
            sr = rows + d_row * dist
            valid = (sc >= 0) & (sc < w_s - 1) & (sr >= 0) & (sr < h_s - 1)
            if not np.any(valid):
                break
            sc_i = np.clip(sc, 0, w_s - 1).astype(np.int32)
            sr_i = np.clip(sr, 0, h_s - 1).astype(np.int32)
            ray_z = dsm_s + dist * dz_per_px
            occluded = valid & (dsm_s[sr_i, sc_i] > ray_z + 0.35)
            in_shadow = np.maximum(in_shadow, occluded.astype(np.float32))

        # Slight Gaussian softening for realistic penumbra
        in_shadow = cv2.GaussianBlur(in_shadow, (5, 5), 1.0)
        if scale > 1:
            in_shadow = cv2.resize(
                in_shadow, (self.width_px, self.height_px), interpolation=cv2.INTER_LINEAR
            )
        return np.clip(in_shadow, 0.0, 1.0).astype(np.float32)


def _normalize_imagery(raw: np.ndarray) -> tuple[np.ndarray, bool]:
    """Convert rasterio (C, H, W) imagery into float32 (H, W, 4) [R, G, B, NIR] in [0, 1]."""
    if raw.ndim == 2:
        raw = raw[None, :, :]
    c, h, w = raw.shape
    arr = raw.astype(np.float32)

    if raw.dtype == np.uint8:
        arr /= 255.0
    elif raw.dtype == np.uint16:
        p995 = float(np.percentile(arr, 99.5))
        denom = 10000.0 if p995 <= 12000.0 else (4095.0 if p995 <= 4200.0 else 65535.0)
        arr = np.clip(arr / max(denom, 1.0), 0.0, 1.0)
    else:
        max_v = float(np.nanmax(arr)) if arr.size else 1.0
        if max_v > 1.5:
            arr = arr / max_v
        arr = np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)

    has_real_nir = c >= 4
    out = np.zeros((h, w, 4), dtype=np.float32)
    if c >= 4:
        out[:, :, 0] = arr[0]
        out[:, :, 1] = arr[1]
        out[:, :, 2] = arr[2]
        out[:, :, 3] = arr[3]
    elif c == 3:
        out[:, :, 0] = arr[0]
        out[:, :, 1] = arr[1]
        out[:, :, 2] = arr[2]
        # Synthetic NIR proxy when only RGB is supplied (vegetation-sensitive green emphasis)
        out[:, :, 3] = np.clip(0.55 * arr[1] + 0.30 * arr[0] + 0.15 * arr[2], 0.0, 1.0)
    else:
        for idx in range(4):
            out[:, :, idx] = arr[0]

    return out, has_real_nir


def _compute_normals(dsm: np.ndarray, gsd_m: float) -> np.ndarray:
    """Compute unit surface normals (nx=East, ny=North, nz=Up) from DSM heightfield."""
    # np.gradient axis 0 = row (South = -North), axis 1 = col (East = +East)
    dz_drow, dz_dcol = np.gradient(dsm.astype(np.float32), gsd_m)
    dz_dx = dz_dcol
    dz_dy = -dz_drow
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(dsm, dtype=np.float32)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    return np.stack([nx / norm, ny / norm, nz / norm], axis=-1).astype(np.float32)


def load_scene(scene_path: str | Path, dsm_path: str | Path | None = None) -> SceneBundle:
    """Load a standardized scene directory (imagery.tif + dsm.tif + metadata.json)
    or a standalone georeferenced GeoTIFF file.
    """
    p = Path(scene_path)
    metadata: dict[str, Any] = {}

    if p.is_dir():
        img_path = p / "imagery.tif"
        if not img_path.exists():
            tifs = sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))
            if not tifs:
                raise FileNotFoundError(f"No GeoTIFF found in scene directory: {p}")
            img_path = tifs[0]
        if dsm_path is None and (p / "dsm.tif").exists():
            dsm_path = p / "dsm.tif"
        meta_file = p / "metadata.json"
        if meta_file.exists():
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    else:
        img_path = p
        if dsm_path is None and (p.parent / "dsm.tif").exists():
            dsm_path = p.parent / "dsm.tif"

    with rasterio.open(img_path) as src:
        raw_img = src.read()
        img_transform: Affine = src.transform
        img_crs = src.crs
        h, w = src.height, src.width
        bounds = src.bounds

    # Ensure metric CRS; if geographic (EPSG:4326), convert approximate GSD in meters
    if img_crs is not None and img_crs.is_projected:
        crs_str = img_crs.to_string()
        gsd_m = float((abs(img_transform.a) + abs(img_transform.e)) * 0.5)
        x_min_m, y_min_m, x_max_m, y_max_m = (
            float(bounds.left),
            float(bounds.bottom),
            float(bounds.right),
            float(bounds.top),
        )
    else:
        # Fallback for geographic GeoTIFF: reproject to local UTM automatically
        center_lon = 0.5 * (bounds.left + bounds.right)
        center_lat = 0.5 * (bounds.bottom + bounds.top)
        utm_zone = int((center_lon + 180.0) / 6.0) + 1
        epsg = (32600 if center_lat >= 0 else 32700) + utm_zone
        crs_str = f"EPSG:{epsg}"
        t_to_utm = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        x_min_m, y_min_m = t_to_utm.transform(bounds.left, bounds.bottom)
        x_max_m, y_max_m = t_to_utm.transform(bounds.right, bounds.top)
        gsd_m = float(((x_max_m - x_min_m) / w + (y_max_m - y_min_m) / h) * 0.5)

    imagery, has_real_nir = _normalize_imagery(raw_img)

    # Load and align DSM to exact imagery grid
    has_real_dsm = False
    if dsm_path is not None and Path(dsm_path).exists():
        with rasterio.open(dsm_path) as dsrc:
            dsm_aligned = np.empty((h, w), dtype=np.float32)
            reproject(
                source=rasterio.band(dsrc, 1),
                destination=dsm_aligned,
                src_transform=dsrc.transform,
                src_crs=dsrc.crs,
                dst_transform=img_transform,
                dst_crs=img_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
            valid_mask = np.isfinite(dsm_aligned) & (dsm_aligned > -500.0) & (dsm_aligned < 9000.0)
            if np.any(valid_mask):
                fill_val = float(np.median(dsm_aligned[valid_mask]))
                dsm_aligned = np.where(valid_mask, dsm_aligned, fill_val).astype(np.float32)
                has_real_dsm = True
            else:
                dsm_aligned = np.zeros((h, w), dtype=np.float32)
    else:
        dsm_aligned = np.zeros((h, w), dtype=np.float32)

    normals = _compute_normals(dsm_aligned, gsd_m)
    red = imagery[:, :, 0]
    nir = imagery[:, :, 3]
    ndvi = np.clip((nir - red) / (nir + red + 1e-6), -1.0, 1.0).astype(np.float32)

    # Determine scene origin in metric coordinates
    origin_x_m = 0.5 * (x_min_m + x_max_m)
    origin_y_m = 0.5 * (y_min_m + y_max_m)
    if "origin_lat" in metadata and "origin_lon" in metadata:
        try:
            tr = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
            ox, oy = tr.transform(metadata["origin_lon"], metadata["origin_lat"])
            if x_min_m <= ox <= x_max_m and y_min_m <= oy <= y_max_m:
                origin_x_m, origin_y_m = float(ox), float(oy)
        except Exception:
            pass

    return SceneBundle(
        imagery=imagery,
        dsm=dsm_aligned,
        normals=normals,
        ndvi=ndvi,
        gsd_m=gsd_m,
        crs=crs_str,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        x_min_m=x_min_m,
        y_min_m=y_min_m,
        x_max_m=x_max_m,
        y_max_m=y_max_m,
        z_min_m=float(np.min(dsm_aligned)),
        z_max_m=float(np.max(dsm_aligned)),
        has_real_nir=has_real_nir,
        has_real_dsm=has_real_dsm,
        metadata=metadata,
    )
