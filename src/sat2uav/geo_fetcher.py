"""On-demand geospatial scene fetcher for confirmed US, Spain, and France regions.

Fetches only the requested corridor (e.g., 2–5 km box) into a standardized
GeoTIFF bundle (`imagery.tif` + `dsm.tif` + `metadata.json`) in a local metric
UTM CRS so 100–300 km geofences never require terabyte downloads.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import requests
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

from .config import load_regions_registry

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _utm_epsg(lat: float, lon: float) -> str:
    zone = int((lon + 180.0) / 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def sample_point_in_geofence(
    origin_lat: float,
    origin_lon: float,
    radius_km: float,
    rng: np.random.Generator,
    max_radius_km: float | None = None,
) -> tuple[float, float, float]:
    """Sample a random (lat, lon) uniformly inside the region geofence disk."""
    eff_r_km = min(radius_km, max_radius_km) if max_radius_km is not None else radius_km
    epsg = _utm_epsg(origin_lat, origin_lon)
    to_utm = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    to_wgs = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)

    ox, oy = to_utm.transform(origin_lon, origin_lat)
    r_m = math.sqrt(float(rng.random())) * (eff_r_km * 1000.0)
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    lon, lat = to_wgs.transform(ox + r_m * math.cos(theta), oy + r_m * math.sin(theta))
    return float(lat), float(lon), float(r_m / 1000.0)


def _fetch_stac_mosaic_to_grid(
    catalog: pystac_client.Client,
    collection: str,
    asset_key: str,
    bbox_wgs84: list[float],
    dst_crs: str,
    dst_transform,
    width: int,
    height: int,
    num_bands: int,
    dtype: Any,
    resampling: Resampling = Resampling.bilinear,
    max_items: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """Query Planetary Computer STAC and reproject matching COG assets onto a target grid."""
    search = catalog.search(
        collections=[collection],
        bbox=bbox_wgs84,
        max_items=max_items,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No STAC items found in collection '{collection}' for bbox {bbox_wgs84}"
        )
    # Prefer most recent imagery
    items.sort(key=lambda it: str(it.datetime or ""), reverse=True)

    if np.issubdtype(np.dtype(dtype), np.floating):
        mosaic = np.full((num_bands, height, width), np.nan, dtype=dtype)
        nodata_val = np.nan
    else:
        mosaic = np.zeros((num_bands, height, width), dtype=dtype)
        nodata_val = 0

    used_ids: list[str] = []
    for item in items:
        signed = planetary_computer.sign(item)
        if asset_key not in signed.assets:
            continue
        href = signed.assets[asset_key].href
        tmp = (
            np.full((num_bands, height, width), np.nan, dtype=dtype)
            if np.issubdtype(np.dtype(dtype), np.floating)
            else np.zeros((num_bands, height, width), dtype=dtype)
        )
        with rasterio.open(href) as src:
            bands_to_read = min(num_bands, src.count)
            for b in range(1, bands_to_read + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=tmp[b - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    dst_nodata=nodata_val,
                )

        if np.issubdtype(np.dtype(dtype), np.floating):
            empty = ~np.isfinite(mosaic)
            has_data = np.isfinite(tmp)
        else:
            empty = mosaic == 0
            has_data = tmp != 0

        fill = empty & has_data
        if np.any(fill):
            mosaic = np.where(fill, tmp, mosaic)
            used_ids.append(item.id)

        # Stop once >99.5% of pixels are populated
        coverage = float(np.mean(np.isfinite(mosaic[0]) if np.issubdtype(np.dtype(dtype), np.floating) else mosaic[0] > 0))
        if coverage >= 0.995:
            break

    return mosaic, used_ids


def _fetch_wms_image(
    url: str,
    layer: str,
    bbox_metric: tuple[float, float, float, float],
    crs: str,
    width: int,
    height: int,
    version: str = "1.1.1",
) -> np.ndarray:
    """Fetch a georeferenced RGB image from an open national OGC WMS endpoint."""
    minx, miny, maxx, maxy = bbox_metric
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": version,
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/jpeg",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "SRS" if version == "1.1.1" else "CRS": crs,
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    from io import BytesIO

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def fetch_scene_corridor(
    lat: float,
    lon: float,
    output_dir: str | Path,
    size_km: float = 2.0,
    gsd_m: float = 1.0,
    region_id: str | None = None,
    country: str = "USA",
) -> Path:
    """Fetch a standardized metric GeoTIFF bundle (`imagery.tif`, `dsm.tif`, `metadata.json`)
    centered at `(lat, lon)` with width/height `size_km` and resolution `gsd_m`.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dst_crs = _utm_epsg(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    to_wgs = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)

    cx, cy = to_utm.transform(lon, lat)
    half_m = (size_km * 1000.0) * 0.5
    x_min, y_min, x_max, y_max = cx - half_m, cy - half_m, cx + half_m, cy + half_m

    width = max(64, int(round((x_max - x_min) / gsd_m)))
    height = max(64, int(round((y_max - y_min) / gsd_m)))
    dst_transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    lon_min, lat_min = to_wgs.transform(x_min, y_min)
    lon_max, lat_max = to_wgs.transform(x_max, y_max)
    bbox_wgs84 = [lon_min, lat_min, lon_max, lat_max]

    catalog = pystac_client.Client.open(PC_STAC_URL)
    imagery_sources: list[str] = []
    dsm_sources: list[str] = []

    country_u = country.upper()
    if country_u == "USA":
        imagery, imagery_sources = _fetch_stac_mosaic_to_grid(
            catalog=catalog,
            collection="naip",
            asset_key="image",
            bbox_wgs84=bbox_wgs84,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            width=width,
            height=height,
            num_bands=4,
            dtype=np.uint8,
            resampling=Resampling.bilinear,
        )
        # Prefer USGS 3DEP Seamless DEM; fall back to Copernicus GLO-30 if needed
        try:
            dsm, dsm_sources = _fetch_stac_mosaic_to_grid(
                catalog=catalog,
                collection="3dep-seamless",
                asset_key="data",
                bbox_wgs84=bbox_wgs84,
                dst_crs=dst_crs,
                dst_transform=dst_transform,
                width=width,
                height=height,
                num_bands=1,
                dtype=np.float32,
                resampling=Resampling.bilinear,
            )
            dsm_collection = "3dep-seamless"
        except Exception:
            dsm, dsm_sources = _fetch_stac_mosaic_to_grid(
                catalog=catalog,
                collection="cop-dem-glo-30",
                asset_key="data",
                bbox_wgs84=bbox_wgs84,
                dst_crs=dst_crs,
                dst_transform=dst_transform,
                width=width,
                height=height,
                num_bands=1,
                dtype=np.float32,
                resampling=Resampling.bilinear,
            )
            dsm_collection = "cop-dem-glo-30"
        imagery_collection = "naip"

    elif country_u == "SPAIN":
        rgb = _fetch_wms_image(
            url="https://www.ign.es/wms-inspire/pnoa-ma",
            layer="OI.OrthoimageCoverage",
            bbox_metric=(x_min, y_min, x_max, y_max),
            crs=dst_crs,
            width=width,
            height=height,
        )
        # Approximate NIR channel if separate IRC layer isn't returned cleanly
        nir = np.clip(0.55 * rgb[:, :, 1] + 0.30 * rgb[:, :, 0] + 0.15 * rgb[:, :, 2], 0, 255).astype(np.uint8)
        imagery = np.stack([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], nir], axis=0)
        imagery_collection = "spain-ign-pnoa-wms"
        imagery_sources = ["OI.OrthoimageCoverage"]
        dsm, dsm_sources = _fetch_stac_mosaic_to_grid(
            catalog=catalog,
            collection="cop-dem-glo-30",
            asset_key="data",
            bbox_wgs84=bbox_wgs84,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            width=width,
            height=height,
            num_bands=1,
            dtype=np.float32,
        )
        dsm_collection = "cop-dem-glo-30"

    elif country_u == "FRANCE":
        # IGN Géoplateforme WMS 1.3.0 accepts EPSG:3857 with max 2048x2048 per request
        to_3857 = Transformer.from_crs(dst_crs, "EPSG:3857", always_xy=True)
        bx0, by0 = to_3857.transform(x_min, y_min)
        bx1, by1 = to_3857.transform(x_max, y_max)
        req_w = min(width, 2048)
        req_h = min(height, 2048)
        rgb = _fetch_wms_image(
            url="https://data.geopf.fr/wms-r",
            layer="ORTHOIMAGERY.ORTHOPHOTOS",
            bbox_metric=(bx0, by0, bx1, by1),
            crs="EPSG:3857",
            width=req_w,
            height=req_h,
            version="1.3.0",
        )
        if (req_w, req_h) != (width, height):
            rgb = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)
        try:
            irc = _fetch_wms_image(
                url="https://data.geopf.fr/wms-r",
                layer="ORTHOIMAGERY.ORTHOPHOTOS.IRC",
                bbox_metric=(bx0, by0, bx1, by1),
                crs="EPSG:3857",
                width=req_w,
                height=req_h,
                version="1.3.0",
            )
            if (req_w, req_h) != (width, height):
                irc = np.asarray(Image.fromarray(irc).resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)
            nir = irc[:, :, 0]  # Red channel of IRC carries Near-Infrared
        except Exception:
            nir = np.clip(0.55 * rgb[:, :, 1] + 0.30 * rgb[:, :, 0] + 0.15 * rgb[:, :, 2], 0, 255).astype(np.uint8)
        imagery = np.stack([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], nir], axis=0)
        imagery_collection = "france-ign-bd-ortho-wms"
        imagery_sources = ["ORTHOIMAGERY.ORTHOPHOTOS", "ORTHOIMAGERY.ORTHOPHOTOS.IRC"]
        dsm, dsm_sources = _fetch_stac_mosaic_to_grid(
            catalog=catalog,
            collection="cop-dem-glo-30",
            asset_key="data",
            bbox_wgs84=bbox_wgs84,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            width=width,
            height=height,
            num_bands=1,
            dtype=np.float32,
        )
        dsm_collection = "cop-dem-glo-30"
    else:
        raise ValueError(f"Unsupported country '{country}'. Supported: USA, Spain, France.")

    # Write standardized imagery.tif (4-band RGB+NIR)
    img_out = out_dir / "imagery.tif"
    with rasterio.open(
        img_out,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="uint8",
        crs=dst_crs,
        transform=dst_transform,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(imagery)
        dst.descriptions = ("Red", "Green", "Blue", "NIR")

    # Fill any tiny NaN gaps in DSM before saving
    dsm_arr = dsm[0]
    valid = np.isfinite(dsm_arr)
    if np.any(valid):
        dsm_arr = np.where(valid, dsm_arr, float(np.median(dsm_arr[valid]))).astype(np.float32)
    else:
        dsm_arr = np.zeros((height, width), dtype=np.float32)

    dsm_out = out_dir / "dsm.tif"
    with rasterio.open(
        dsm_out,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=dst_crs,
        transform=dst_transform,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(dsm_arr, 1)

    meta: dict[str, Any] = {
        "region_id": region_id,
        "country": country,
        "origin_lat": lat,
        "origin_lon": lon,
        "size_km": size_km,
        "gsd_m": gsd_m,
        "crs": dst_crs,
        "width_px": width,
        "height_px": height,
        "imagery_collection": imagery_collection,
        "imagery_sources": imagery_sources,
        "dsm_collection": dsm_collection,
        "dsm_sources": dsm_sources,
        "elevation_min_m": float(np.min(dsm_arr)),
        "elevation_max_m": float(np.max(dsm_arr)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def fetch_region_sample(
    region_id: str,
    output_dir: str | Path,
    size_km: float = 2.0,
    gsd_m: float = 1.0,
    random_within_geofence: bool = False,
    seed: int = 42,
    lat: float | None = None,
    lon: float | None = None,
) -> Path:
    """Fetch a corridor from one of the confirmed regions in REGIONS_AND_PARAMS.json."""
    regions = load_regions_registry()
    if region_id not in regions:
        raise KeyError(
            f"Unknown region_id '{region_id}'. Available: {list(regions.keys())}"
        )
    reg = regions[region_id]
    if lat is None or lon is None:
        if random_within_geofence:
            rng = np.random.default_rng(seed)
            lat, lon, _ = sample_point_in_geofence(
                reg["origin"]["lat"],
                reg["origin"]["lon"],
                reg["geofence_radius_km"],
                rng,
            )
        else:
            lat = float(reg["origin"]["lat"])
            lon = float(reg["origin"]["lon"])

    return fetch_scene_corridor(
        lat=lat,
        lon=lon,
        output_dir=output_dir,
        size_km=size_km,
        gsd_m=gsd_m,
        region_id=region_id,
        country=reg["country"],
    )


def create_synthetic_test_scene(
    output_dir: str | Path,
    width: int = 640,
    height: int = 640,
    gsd_m: float = 1.5,
) -> Path:
    """Create a realistic offline GeoTIFF test scene (4-band RGB+NIR + 3D terrain/buildings DSM)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crs = "EPSG:32612"  # UTM Zone 12N (Utah)
    x_min, y_min = 620000.0, 4270000.0
    x_max = x_min + width * gsd_m
    y_max = y_min + height * gsd_m
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = xx / width
    yn = yy / height

    # Procedural canyon + hills + urban block DSM
    dsm = (
        1200.0
        + 45.0 * np.sin(xn * 2.0 * math.pi * 1.5) * np.cos(yn * 2.0 * math.pi * 1.2)
        + 22.0 * np.sin((xn + yn) * 14.0)
        - 35.0 * np.exp(-(((xn - 0.52) + 0.25 * np.sin(yn * 6.0)) ** 2) / 0.008)
    ).astype(np.float32)

    # Add synthetic urban buildings with crisp 18–35 m height
    bldg_mask = np.zeros((height, width), dtype=bool)
    for bx in range(int(0.18 * width), int(0.42 * width), 36):
        for by in range(int(0.20 * height), int(0.75 * height), 42):
            dsm[by : by + 22, bx : bx + 20] += 24.0
            bldg_mask[by : by + 22, bx : bx + 20] = True

    # 4-band RGB + NIR texture
    veg_mask = (xn > 0.62) & (dsm > 1190.0)
    river_mask = dsm < 1172.0

    r = np.full((height, width), 175.0, dtype=np.float32)
    g = np.full((height, width), 135.0, dtype=np.float32)
    b = np.full((height, width), 102.0, dtype=np.float32)
    nir = np.full((height, width), 145.0, dtype=np.float32)

    # Fine surface grain
    rng = np.random.default_rng(1234)
    grain = rng.normal(0.0, 10.0, size=(height, width)).astype(np.float32)
    r, g, b, nir = r + grain, g + grain, b + grain, nir + grain

    # Vegetation (bright NIR, green RGB)
    r[veg_mask], g[veg_mask], b[veg_mask], nir[veg_mask] = 52.0, 118.0, 48.0, 225.0
    # Water / canyon river (dark NIR, blue-teal RGB)
    r[river_mask], g[river_mask], b[river_mask], nir[river_mask] = 28.0, 62.0, 96.0, 18.0
    # Buildings (bright concrete roofs, low NIR)
    r[bldg_mask], g[bldg_mask], b[bldg_mask], nir[bldg_mask] = 212.0, 208.0, 200.0, 120.0

    imagery = np.clip(np.stack([r, g, b, nir], axis=0), 0, 255).astype(np.uint8)

    with rasterio.open(
        out_dir / "imagery.tif",
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(imagery)

    with rasterio.open(
        out_dir / "dsm.tif",
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(dsm, 1)

    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    olon, olat = to_wgs.transform(0.5 * (x_min + x_max), 0.5 * (y_min + y_max))
    meta = {
        "region_id": "SYNTHETIC_TEST",
        "country": "TEST",
        "origin_lat": float(olat),
        "origin_lon": float(olon),
        "size_km": round((width * gsd_m) / 1000.0, 3),
        "gsd_m": gsd_m,
        "crs": crs,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir
