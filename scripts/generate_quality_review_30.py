"""Generate 30 multi-engine, multi-spectral, multi-region test photos and an interactive HTML review gallery."""

from __future__ import annotations

import html
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sat2uav.config import GenerationConfig, VideoSampleParams, load_generation_config
from sat2uav.distributions import (
    STANDARD_RESOLUTIONS,
    sample_airspeed_kmh,
    sample_altitude_agl_m,
    sample_fov_deg,
    sample_pitch_deg,
    sample_sun_elevation_deg,
    sample_visibility_km,
)
from sat2uav.effects import EffectsPipeline
from sat2uav.geo_fetcher import fetch_region_sample
from sat2uav.gpu_renderer import EngineType, create_renderer
from sat2uav.scene import SceneBundle, load_scene
from sat2uav.trajectory import FramePose, sample_valid_flight
from sat2uav.xml_export import save_footage_xml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "quality_review_30"
SCENES_DIR = OUT_DIR / "scenes"
PHOTOS_DIR = OUT_DIR / "photos"
XML_DIR = OUT_DIR / "xml"

ENGINES: list[EngineType] = ["gpu_opengl_raymarch", "gpu_opengl_mesh", "cpu_raymarch"]
SPECTRAL_MODES = ["rgb", "panchromatic", "nir", "false_color_cir", "thermal_lwir", "night_vision"]

# Visually rich coordinates inside each confirmed region's geofence (all 100% land)
REGION_FETCH_SPECS: list[tuple[str, float, float, float, float]] = [
    # (region_id, lat, lon, size_km, gsd_m)
    ("R1_SE_US_LAND_CORRIDOR", 30.8327, -83.2785, 2.8, 1.0),   # Valdosta / South GA urban-forest-highway
    ("R2_LAS_VEGAS_NV", 36.1147, -115.1728, 2.8, 1.0),         # Las Vegas Strip urban-desert transition
    ("R3_MOJAVE_DESERT_CA", 34.9150, -117.0150, 2.8, 1.0),     # Mojave arid desert & dry washes
    ("R4_MOAB_CANYONLANDS_UT", 38.5733, -109.5498, 2.0, 1.0),  # Moab red-rock canyon & town (reuses cache)
    ("R5_CENTRAL_SPAIN_PNOA", 39.8628, -4.0273, 2.5, 1.2),     # Toledo / Iberian arid plateau & historic city
    ("R6_FRANCE_PARIS_LOIRE_BASIN", 47.9029, 1.9093, 2.5, 1.2),# Orléans / Loire river, bridges & European city
]


def prepare_scenes() -> dict[str, SceneBundle]:
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, SceneBundle] = {}

    # Reuse existing Moab cache if present
    existing_moab = ROOT / "outputs" / "moab_live_demo" / "scene"
    moab_target = SCENES_DIR / "R4_MOAB_CANYONLANDS_UT"
    if existing_moab.exists() and not (moab_target / "imagery.tif").exists():
        shutil.copytree(existing_moab, moab_target, dirs_exist_ok=True)

    for rid, lat, lon, size_km, gsd_m in REGION_FETCH_SPECS:
        target = SCENES_DIR / rid
        if not (target / "imagery.tif").exists():
            print(f"[fetch] {rid} @ ({lat:.4f}, {lon:.4f}), {size_km} km, {gsd_m} m GSD...")
            try:
                fetch_region_sample(
                    region_id=rid,
                    output_dir=target,
                    size_km=size_km,
                    gsd_m=gsd_m,
                    lat=lat,
                    lon=lon,
                )
            except Exception as exc:
                print(f"  [warn] Failed fetching {rid}: {exc}")
                continue
        else:
            print(f"[cache] Reusing {rid}")

        try:
            loaded[rid] = load_scene(target)
        except Exception as exc:
            print(f"  [warn] Failed loading {rid}: {exc}")

    return loaded


def render_single_photo(
    scene: SceneBundle,
    params: VideoSampleParams,
    pose: FramePose,
    engine: EngineType,
    out_png: Path,
    out_xml: Path | None = None,
) -> tuple[float, float]:
    """Render one photo and return (renderer_ms, total_ms_including_effects)."""
    t_total0 = time.perf_counter()
    renderer = create_renderer(engine, scene, params)
    try:
        # Warmup GPU shader pipeline for accurate per-frame timing
        _ = renderer.render_frame(pose)
        t_r0 = time.perf_counter()
        render_out = renderer.render_frame(pose)
        render_ms = (time.perf_counter() - t_r0) * 1000.0

        effects = EffectsPipeline(params)
        frame_rgb = effects.apply(render_out, pose, renderer.focal_px)
    finally:
        if hasattr(renderer, "close"):
            renderer.close()

    total_ms = (time.perf_counter() - t_total0) * 1000.0
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    if out_xml is not None:
        save_footage_xml(
            output_xml=out_xml,
            video_filename=out_png.name,
            engine_name=engine,
            scene=scene,
            params=params,
            poses=[pose],
            keyframe_interval=10,
            keyframe_files={0: out_png.name},
            visibility_diagnostics=params.visibility_diagnostics,
        )
    return render_ms, total_ms


def build_fixed_pose(
    scene: SceneBundle,
    yaw_deg: float,
    pitch_deg: float,
    altitude_agl_m: float,
    speed_mps: float,
    dx_m: float = 0.0,
    dy_m: float = 0.0,
) -> FramePose:
    x = scene.origin_x_m + dx_m
    y = scene.origin_y_m + dy_m
    gz = scene.sample_elevation(x, y)
    lat, lon = scene.to_latlon(x, y)
    return FramePose(
        frame_idx=0,
        time_sec=0.0,
        x_m=round(x, 3),
        y_m=round(y, 3),
        z_msl_m=round(gz + altitude_agl_m, 3),
        z_agl_m=round(altitude_agl_m, 3),
        ground_z_m=round(gz, 3),
        lat=round(lat, 7),
        lon=round(lon, 7),
        yaw_deg=round(yaw_deg, 2),
        pitch_deg=round(pitch_deg, 2),
        speed_mps=round(speed_mps, 2),
    )


def build_html_gallery(records: list[dict[str, Any]], engine_stats: dict[str, dict[str, float]]) -> Path:
    html_path = OUT_DIR / "index.html"
    records_json = json.dumps(records)
    regions_seen = sorted({r["region_id"] for r in records})

    engine_cards = []
    engine_desc = {
        "gpu_opengl_raymarch": "NVIDIA GTX 1050 GLSL 3.30 3D DSM Ray-Marcher + 16x Anisotropic Mipmaps + Native Lens Distortion",
        "gpu_opengl_mesh": "NVIDIA GTX 1050 3D Displaced Heightfield Triangle Mesh + Hardware Z-Buffer + 16x AF",
        "cpu_raymarch": "NumPy + OpenCV CPU 3D DSM Ray-Marcher (Sub-Grid Depth Upsampling, Zero-GPU Fallback)",
    }
    for eng in ENGINES:
        st = engine_stats.get(eng, {"avg_ms": 0.0, "fps": 0.0})
        engine_cards.append(
            f"""
            <div class="stat-card">
              <div class="stat-label">{html.escape(eng)}</div>
              <div class="stat-val">{st['avg_ms']:.1f} ms <span class="stat-sub">/ frame ({st['fps']:.1f} fps)</span></div>
              <div class="stat-desc">{html.escape(engine_desc[eng])}</div>
            </div>
            """
        )

    # Build Side-by-Side 3-Engine Comparison rows for the 18 matrix photos
    matrix_rows = []
    for mode in SPECTRAL_MODES:
        cells = []
        for eng in ENGINES:
            rec = next((r for r in records if r["section"] == "engine_matrix" and r["spectral_mode"] == mode and r["engine"] == eng), None)
            if rec:
                cells.append(
                    f"""
                    <div class="cmp-cell" onclick="openLightbox({rec['id']})">
                      <img src="{html.escape(rec['rel_path'])}" loading="lazy" alt="{html.escape(rec['title'])}" />
                      <div class="cmp-meta">
                        <span class="badge eng">{html.escape(eng)}</span>
                        <span class="badge ms">{rec['render_ms']:.1f} ms</span>
                      </div>
                    </div>
                    """
                )
        matrix_rows.append(
            f"""
            <div class="cmp-row">
              <div class="cmp-mode-title">Spectral Mode: <strong>{html.escape(mode.upper())}</strong></div>
              <div class="cmp-grid">{''.join(cells)}</div>
            </div>
            """
        )

    region_options = "".join(f'<option value="{html.escape(r)}">{html.escape(r)}</option>' for r in regions_seen)
    mode_options = "".join(f'<option value="{html.escape(m)}">{html.escape(m)}</option>' for m in SPECTRAL_MODES)
    eng_options = "".join(f'<option value="{html.escape(e)}">{html.escape(e)}</option>' for e in ENGINES)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Satellite-to-UAV | 30-Photo Render Engine & Spectral Quality Review</title>
<style>
  :root {{
    --bg: #0b0f17;
    --surface: #131a26;
    --surface-2: #1b2434;
    --border: #273449;
    --text: #e6edf7;
    --muted: #93a4be;
    --accent: #38bdf8;
    --green: #34d399;
    --amber: #fbbf24;
    --purple: #a78bfa;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 24px 32px 64px;
  }}
  header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 18px;
    margin-bottom: 24px;
  }}
  h1 {{ font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }}
  .subtitle {{ color: var(--muted); margin-top: 6px; font-size: 0.95rem; }}
  h2 {{
    font-size: 1.25rem;
    margin: 30px 0 14px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
  }}
  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
  }}
  .stat-label {{ color: var(--accent); font-weight: 600; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  .stat-val {{ font-size: 1.6rem; font-weight: 700; margin: 6px 0; }}
  .stat-sub {{ font-size: 0.9rem; font-weight: 400; color: var(--green); }}
  .stat-desc {{ font-size: 0.82rem; color: var(--muted); }}

  .cmp-row {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 14px;
  }}
  .cmp-mode-title {{ margin-bottom: 10px; font-size: 0.95rem; color: var(--amber); }}
  .cmp-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }}
  .cmp-cell {{
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.15s ease, border-color 0.15s ease;
  }}
  .cmp-cell:hover {{ transform: translateY(-2px); border-color: var(--accent); }}
  .cmp-cell img {{ width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover; }}
  .cmp-meta {{ padding: 8px 10px; display: flex; justify-content: space-between; align-items: center; }}

  .filters {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 18px;
  }}
  .filters label {{ font-size: 0.85rem; color: var(--muted); display: flex; align-items: center; gap: 6px; }}
  select, button {{
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 0.85rem;
    cursor: pointer;
  }}
  button:hover {{ border-color: var(--accent); }}
  #countLabel {{ margin-left: auto; font-size: 0.85rem; color: var(--accent); font-weight: 600; }}

  .gallery {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.15s ease, border-color 0.15s ease;
  }}
  .card:hover {{ transform: translateY(-3px); border-color: var(--accent); }}
  .card img {{ width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover; background: #000; }}
  .card-body {{ padding: 12px 14px; }}
  .card-title {{ font-weight: 600; font-size: 0.95rem; margin-bottom: 6px; }}
  .badges {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 8px; }}
  .badge {{
    font-size: 0.72rem;
    padding: 2px 7px;
    border-radius: 999px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text);
  }}
  .badge.eng {{ border-color: #38bdf866; color: var(--accent); }}
  .badge.mode {{ border-color: #a78bfa66; color: var(--purple); }}
  .badge.ms {{ border-color: #34d39966; color: var(--green); }}
  .badge.night {{ border-color: #fbbf2466; color: var(--amber); }}
  .telemetry {{ font-size: 0.78rem; color: var(--muted); line-height: 1.55; }}

  .lightbox {{
    position: fixed;
    inset: 0;
    background: rgba(5, 8, 14, 0.94);
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    padding: 24px;
  }}
  .lightbox.open {{ display: flex; }}
  .lightbox img {{ max-width: 92vw; max-height: 76vh; border-radius: 8px; border: 1px solid var(--border); }}
  .lb-info {{ margin-top: 14px; max-width: 1000px; text-align: center; font-size: 0.9rem; color: var(--text); }}
  .lb-sub {{ color: var(--muted); font-size: 0.82rem; margin-top: 4px; }}
  .lb-controls {{ margin-top: 12px; display: flex; gap: 10px; }}
</style>
</head>
<body>
  <header>
    <h1>Satellite-to-UAV — 30-Photo Multi-Engine &amp; Spectral Quality Review</h1>
    <div class="subtitle">
      Side-by-side comparison of 3 render engines (NVIDIA GTX 1050 OpenGL Ray-March, OpenGL 3D Mesh, CPU Ray-March),
      all 6 spectral camera models, 4 equal aviation resolutions (360p / 480p / 720p / 1080p — no 4K),
      and all {len(regions_seen)} confirmed global regions. Click any photo to inspect full resolution.
    </div>
  </header>

  <h2>Render Engine Latency Benchmark (GTX 1050 EGL vs. CPU)</h2>
  <div class="stats-grid">
    {''.join(engine_cards)}
  </div>

  <h2>Part 1 — Direct 3-Engine × 6-Spectral Comparison (18 Photos, Identical Pose)</h2>
  {''.join(matrix_rows)}

  <h2>Part 2 — All 30 Test Photos (Filterable Gallery + Regional Aviation Envelope Showcase)</h2>
  <div class="filters">
    <label>Section:
      <select id="fSection" onchange="renderGrid()">
        <option value="all">All (30)</option>
        <option value="engine_matrix">Engine × Spectral Matrix (18)</option>
        <option value="regional_showcase">Multi-Region Aviation Showcase (12)</option>
      </select>
    </label>
    <label>Engine:
      <select id="fEngine" onchange="renderGrid()">
        <option value="all">All</option>
        {eng_options}
      </select>
    </label>
    <label>Spectral:
      <select id="fMode" onchange="renderGrid()">
        <option value="all">All</option>
        {mode_options}
      </select>
    </label>
    <label>Region:
      <select id="fRegion" onchange="renderGrid()">
        <option value="all">All</option>
        {region_options}
      </select>
    </label>
    <button onclick="resetFilters()">Reset</button>
    <span id="countLabel">Showing 30 / 30</span>
  </div>

  <div class="gallery" id="gallery"></div>

  <div class="lightbox" id="lightbox" onclick="if(event.target===this) closeLightbox()">
    <img id="lbImg" src="" alt="" />
    <div class="lb-info" id="lbTitle"></div>
    <div class="lb-sub" id="lbTelemetry"></div>
    <div class="lb-controls">
      <button onclick="stepLightbox(-1)">&#8592; Prev</button>
      <button onclick="closeLightbox()">Close (Esc)</button>
      <button onclick="stepLightbox(1)">Next &#8594;</button>
    </div>
  </div>

<script>
const RECORDS = {records_json};
let currentIdx = 0;

function renderGrid() {{
  const sec = document.getElementById('fSection').value;
  const eng = document.getElementById('fEngine').value;
  const mode = document.getElementById('fMode').value;
  const reg = document.getElementById('fRegion').value;

  const filtered = RECORDS.filter(r =>
    (sec === 'all' || r.section === sec) &&
    (eng === 'all' || r.engine === eng) &&
    (mode === 'all' || r.spectral_mode === mode) &&
    (reg === 'all' || r.region_id === reg)
  );
  document.getElementById('countLabel').textContent = `Showing ${{filtered.length}} / ${{RECORDS.length}}`;

  const container = document.getElementById('gallery');
  container.innerHTML = filtered.map(r => `
    <div class="card" onclick="openLightbox(${{r.id}})">
      <img src="${{r.rel_path}}" loading="lazy" alt="${{r.title}}" />
      <div class="card-body">
        <div class="card-title">#${{String(r.id + 1).padStart(2, '0')}} — ${{r.title}}</div>
        <div class="badges">
          <span class="badge eng">${{r.engine}}</span>
          <span class="badge mode">${{r.spectral_mode}}</span>
          <span class="badge">${{r.resolution}}</span>
          <span class="badge ms">${{r.render_ms.toFixed(1)}} ms</span>
          ${{r.is_night ? '<span class="badge night">NIGHT</span>' : '<span class="badge">DAY</span>'}}
          ${{r.has_crack ? '<span class="badge night">CRACK</span>' : ''}}
          ${{r.has_droplets ? '<span class="badge night">WATERMARKS</span>' : ''}}
        </div>
        <div class="telemetry">
          <strong>${{r.region_id}}</strong><br/>
          Airspeed: ${{r.speed_kmh}} km/h (M${{r.mach}}) &bull; Alt: ${{r.alt_m}} m AGL (${{r.alt_ft}} ft)<br/>
          Pitch: ${{r.pitch_deg}}&deg; &bull; FoV: ${{r.fov_deg}}&deg; &bull; Sun El: ${{r.sun_el_deg}}&deg; &bull; Vis: ${{r.vis_km}} km (${{r.weather_state}})
        </div>
      </div>
    </div>
  `).join('');
}}

function resetFilters() {{
  document.getElementById('fSection').value = 'all';
  document.getElementById('fEngine').value = 'all';
  document.getElementById('fMode').value = 'all';
  document.getElementById('fRegion').value = 'all';
  renderGrid();
}}

function openLightbox(id) {{
  currentIdx = RECORDS.findIndex(r => r.id === id);
  updateLightbox();
  document.getElementById('lightbox').classList.add('open');
}}
function closeLightbox() {{ document.getElementById('lightbox').classList.remove('open'); }}
function stepLightbox(delta) {{
  currentIdx = (currentIdx + delta + RECORDS.length) % RECORDS.length;
  updateLightbox();
}}
function updateLightbox() {{
  const r = RECORDS[currentIdx];
  document.getElementById('lbImg').src = r.rel_path;
  document.getElementById('lbTitle').textContent = `#${{String(r.id+1).padStart(2,'0')}} — ${{r.title}} (${{r.engine}} | ${{r.spectral_mode}} | ${{r.resolution}} | ${{r.render_ms.toFixed(1)}} ms)`;
  document.getElementById('lbTelemetry').textContent =
    `${{r.region_id}} | Airspeed: ${{r.speed_kmh}} km/h (Mach ${{r.mach}}) | Alt: ${{r.alt_m}} m AGL (${{r.alt_ft}} ft) | Pitch: ${{r.pitch_deg}}° | FoV: ${{r.fov_deg}}° | Sun: Az ${{r.sun_az_deg}}° El ${{r.sun_el_deg}}° | Vis: ${{r.vis_km}} km (${{r.weather_state}}) | k1=${{r.k1}} | Crack=${{r.has_crack}} | Droplets=${{r.has_droplets}}`;
}}

document.addEventListener('keydown', e => {{
  if (!document.getElementById('lightbox').classList.contains('open')) return;
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowRight') stepLightbox(1);
  if (e.key === 'ArrowLeft') stepLightbox(-1);
}});

renderGrid();
</script>
</body>
</html>
"""
    html_path.write_text(doc, encoding="utf-8")
    return html_path


def main() -> None:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    XML_DIR.mkdir(parents=True, exist_ok=True)

    scenes = prepare_scenes()
    if not scenes:
        raise RuntimeError("No regions could be fetched or loaded.")

    available_rids = list(scenes.keys())
    print(f"[info] Loaded {len(available_rids)} real regions: {available_rids}")

    cfg = load_generation_config(ROOT / "configs" / "default_generation.yaml")
    rng = np.random.default_rng(2026)

    records: list[dict[str, Any]] = []
    engine_timings: dict[str, list[float]] = {e: [] for e in ENGINES}
    photo_id = 0

    # ------------------------------------------------------------------
    # PART 1: 18 Photos — 3 Engines x 6 Spectral Modes (Identical Pose!)
    # ------------------------------------------------------------------
    base_rid = "R2_LAS_VEGAS_NV" if "R2_LAS_VEGAS_NV" in scenes else available_rids[0]
    alt_rid = "R4_MOAB_CANYONLANDS_UT" if "R4_MOAB_CANYONLANDS_UT" in scenes else base_rid

    for m_idx, mode in enumerate(SPECTRAL_MODES):
        rid = base_rid if m_idx % 2 == 0 else alt_rid
        scene = scenes[rid]
        is_night = mode in ("night_vision", "thermal_lwir")
        alt_m = 260.0 if rid == "R4_MOAB_CANYONLANDS_UT" else 340.0
        speed_kmh = 185.0  # Bayraktar TB2 operational reference
        yaw_deg = (35.0 + m_idx * 55.0) % 360.0
        pitch_deg = -52.0
        fov_deg = 62.0
        vis_km, vis_diag = sample_visibility_km(rid, alt_m, np.random.default_rng(100 + m_idx))

        pose = build_fixed_pose(
            scene,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            altitude_agl_m=alt_m,
            speed_mps=speed_kmh / 3.6,
        )

        for eng in ENGINES:
            params = VideoSampleParams(
                video_index=photo_id,
                seed=5000 + m_idx,  # Identical RNG seed across the 3 engines for fair comparison
                width=1280,
                height=720,
                fps=30,
                duration_sec=10.0,
                start_x_m=pose.x_m,
                start_y_m=pose.y_m,
                start_offset_from_origin_m=0.0,
                yaw_deg=yaw_deg,
                speed_mps=speed_kmh / 3.6,
                altitude_agl_m=alt_m,
                fov_deg=fov_deg,
                pitch_deg=pitch_deg,
                spectral_mode=mode,
                k1=-0.10,
                k2=0.02,
                p1=0.0,
                p2=0.0,
                sun_azimuth_deg=140.0,
                sun_elevation_deg=34.0,
                is_night=is_night,
                shadow_strength=0.55,
                color_temp_k=5600.0,
                wb_tint=0.0,
                haze_enabled=True,
                visibility_km=vis_km,
                sun_glare_enabled=not is_night,
                glare_intensity=0.35,
                specular_glint=0.30,
                motion_blur_enabled=True,
                shutter_sec=0.005,
                rolling_shutter_strength=0.15,
                auto_exposure_enabled=True,
                ae_target_luma=0.50,
                ae_lag_alpha=0.20,
                dr_clip_strength=0.12,
                sensor_noise_enabled=True,
                iso_noise=0.014,
                thermal_fpn=0.018,
                has_lens_crack=(mode == "panchromatic"),
                has_watermark_droplets=(mode == "false_color_cir"),
                visibility_diagnostics=vis_diag,
            )

            fname = f"{photo_id:02d}_matrix_{mode}_{eng}.png"
            out_png = PHOTOS_DIR / fname
            out_xml = XML_DIR / f"{photo_id:02d}_matrix_{mode}_{eng}.xml"
            render_ms, _ = render_single_photo(scene, params, pose, eng, out_png, out_xml)
            engine_timings[eng].append(render_ms)

            records.append(
                {
                    "id": photo_id,
                    "section": "engine_matrix",
                    "title": f"Matrix: {mode.upper()} ({rid.split('_')[0]})",
                    "rel_path": f"photos/{fname}",
                    "region_id": rid,
                    "engine": eng,
                    "spectral_mode": mode,
                    "resolution": f"{params.width}x{params.height}",
                    "render_ms": round(render_ms, 2),
                    "is_night": params.is_night,
                    "speed_kmh": round(speed_kmh, 1),
                    "mach": round(speed_kmh / 1225.06, 2),
                    "alt_m": round(alt_m, 0),
                    "alt_ft": round(alt_m * 3.28084, 0),
                    "pitch_deg": round(pitch_deg, 1),
                    "fov_deg": round(fov_deg, 1),
                    "sun_az_deg": round(params.sun_azimuth_deg, 1),
                    "sun_el_deg": round(params.sun_elevation_deg, 1),
                    "vis_km": round(vis_km, 2),
                    "weather_state": vis_diag.get("weather_state", "vfr"),
                    "k1": round(params.k1, 3),
                    "has_crack": params.has_lens_crack,
                    "has_droplets": params.has_watermark_droplets,
                }
            )
            print(f"  [{photo_id + 1:02d}/30] {fname:45s} | {render_ms:6.1f} ms")
            photo_id += 1

    # ------------------------------------------------------------------
    # PART 2: 12 Photos — Multi-Region, 4 Equal Resolutions & Aviation Envelope
    # ------------------------------------------------------------------
    showcase_specs = [
        # (region_idx, mode, is_night, engine, res_idx, alt_m, speed_kmh, pitch, fov, sun_el, force_crack, force_drop, label)
        (0, "rgb", False, "gpu_opengl_raymarch", 3, 280.0, 145.0, -55.0, 64.0, 26.0, False, True, "South GA/FL Humid Haze + Droplets (1080p)"),
        (0, "nir", False, "gpu_opengl_mesh", 2, 420.0, 210.0, -68.0, 52.0, 48.0, False, False, "South GA/FL Forest NIR Canopy (720p)"),
        (1, "rgb", True, "gpu_opengl_raymarch", 3, 380.0, 195.0, -48.0, 72.0, 18.0, False, False, "Las Vegas Night Urban Glow (1080p)"),
        (1, "thermal_lwir", False, "gpu_opengl_raymarch", 0, 520.0, 310.0, -60.0, 45.0, 62.0, True, False, "Las Vegas Daytime LWIR + Crack (360p)"),
        (2, "rgb", False, "gpu_opengl_raymarch", 2, 300.0, 240.0, -45.0, 78.0, 16.0, False, False, "Mojave Low-Sun Desert Haze (720p)"),
        (2, "panchromatic", False, "gpu_opengl_mesh", 1, 550.0, 680.0, -75.0, 38.0, 55.0, False, False, "Mojave Fast-Mover Mach 0.55 Pan (480p)"),
        (3, "rgb", False, "gpu_opengl_raymarch", 3, 240.0, 170.0, -50.0, 66.0, 22.0, False, False, "Moab Golden-Hour Red-Rock Relief (1080p)"),
        (3, "false_color_cir", False, "gpu_opengl_raymarch", 1, 360.0, 204.0, -64.0, 58.0, 44.0, False, True, "Moab Color-IR + Watermarks (480p)"),
        (4, "rgb", False, "gpu_opengl_raymarch", 2, 320.0, 180.0, -54.0, 60.0, 38.0, False, False, "Spain IGN PNOA Toledo Plateau RGB (720p)"),
        (4, "night_vision", True, "gpu_opengl_mesh", 0, 290.0, 165.0, -58.0, 65.0, 15.0, True, False, "Spain Toledo NVG + Lens Crack (360p)"),
        (5, "rgb", False, "gpu_opengl_raymarch", 3, 350.0, 204.0, -50.0, 56.0, 32.0, False, True, "France IGN BD ORTHO Loire Mist + Droplets (1080p)"),
        (5, "false_color_cir", False, "gpu_opengl_raymarch", 1, 410.0, 450.0, -66.0, 48.0, 50.0, False, False, "France Orléans Real IRC-NIR (480p)"),
    ]

    for s_idx, spec in enumerate(showcase_specs):
        (
            r_idx,
            mode,
            is_night,
            eng,
            res_idx,
            alt_m,
            speed_kmh,
            pitch_deg,
            fov_deg,
            sun_el,
            force_crack,
            force_drop,
            label,
        ) = spec
        rid = available_rids[r_idx % len(available_rids)]
        scene = scenes[rid]
        w, h = STANDARD_RESOLUTIONS[res_idx]
        vis_km, vis_diag = sample_visibility_km(rid, alt_m, np.random.default_rng(900 + s_idx))

        # Clamp altitude slightly if scene footprint is compact so frustum stays 100% inside
        max_span = min(scene.x_max_m - scene.x_min_m, scene.y_max_m - scene.y_min_m)
        safe_alt = min(alt_m, max_span * 0.28)
        yaw_deg = float((25.0 + s_idx * 31.0) % 360.0)
        pose = build_fixed_pose(
            scene,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            altitude_agl_m=safe_alt,
            speed_mps=speed_kmh / 3.6,
        )

        params = VideoSampleParams(
            video_index=photo_id,
            seed=7000 + s_idx,
            width=w,
            height=h,
            fps=30,
            duration_sec=10.0,
            start_x_m=pose.x_m,
            start_y_m=pose.y_m,
            start_offset_from_origin_m=0.0,
            yaw_deg=yaw_deg,
            speed_mps=speed_kmh / 3.6,
            altitude_agl_m=safe_alt,
            fov_deg=fov_deg,
            pitch_deg=pitch_deg,
            spectral_mode=mode,
            k1=float(rng.uniform(-0.18, 0.08)),
            k2=float(rng.uniform(-0.03, 0.03)),
            p1=0.0,
            p2=0.0,
            sun_azimuth_deg=float((60.0 + s_idx * 40.0) % 360.0),
            sun_elevation_deg=sun_el,
            is_night=is_night,
            shadow_strength=0.55,
            color_temp_k=float(rng.uniform(4100.0, 7600.0)),
            wb_tint=float(rng.uniform(-0.04, 0.04)),
            haze_enabled=True,
            visibility_km=vis_km,
            sun_glare_enabled=not is_night,
            glare_intensity=0.45,
            specular_glint=0.35,
            motion_blur_enabled=True,
            shutter_sec=0.004,
            rolling_shutter_strength=0.22,
            auto_exposure_enabled=True,
            ae_target_luma=0.50,
            ae_lag_alpha=0.20,
            dr_clip_strength=0.14,
            sensor_noise_enabled=True,
            iso_noise=0.015,
            thermal_fpn=0.020,
            has_lens_crack=force_crack,
            has_watermark_droplets=force_drop,
            visibility_diagnostics=vis_diag,
        )

        fname = f"{photo_id:02d}_region_{rid.split('_')[0]}_{mode}_{w}x{h}.png"
        out_png = PHOTOS_DIR / fname
        out_xml = XML_DIR / f"{photo_id:02d}_region_{rid.split('_')[0]}_{mode}.xml"
        render_ms, _ = render_single_photo(scene, params, pose, eng, out_png, out_xml)
        engine_timings[eng].append(render_ms)

        records.append(
            {
                "id": photo_id,
                "section": "regional_showcase",
                "title": label,
                "rel_path": f"photos/{fname}",
                "region_id": rid,
                "engine": eng,
                "spectral_mode": mode,
                "resolution": f"{w}x{h}",
                "render_ms": round(render_ms, 2),
                "is_night": params.is_night,
                "speed_kmh": round(speed_kmh, 1),
                "mach": round(speed_kmh / 1225.06, 2),
                "alt_m": round(safe_alt, 0),
                "alt_ft": round(safe_alt * 3.28084, 0),
                "pitch_deg": round(pitch_deg, 1),
                "fov_deg": round(fov_deg, 1),
                "sun_az_deg": round(params.sun_azimuth_deg, 1),
                "sun_el_deg": round(params.sun_elevation_deg, 1),
                "vis_km": round(vis_km, 2),
                "weather_state": vis_diag.get("weather_state", "vfr"),
                "k1": round(params.k1, 3),
                "has_crack": params.has_lens_crack,
                "has_droplets": params.has_watermark_droplets,
            }
        )
        print(f"  [{photo_id + 1:02d}/30] {fname:45s} | {render_ms:6.1f} ms")
        photo_id += 1

    engine_stats = {}
    for eng, times in engine_timings.items():
        avg_ms = float(np.mean(times)) if times else 0.0
        engine_stats[eng] = {"avg_ms": avg_ms, "fps": 1000.0 / avg_ms if avg_ms > 0 else 0.0}

    html_path = build_html_gallery(records, engine_stats)
    (OUT_DIR / "review_manifest.json").write_text(
        json.dumps({"engine_stats": engine_stats, "photos": records}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[done] Generated {len(records)} test photos & HTML gallery -> {html_path}")
    for eng, st in engine_stats.items():
        print(f"  Engine {eng:22s}: avg {st['avg_ms']:6.1f} ms/frame ({st['fps']:5.1f} fps)")


if __name__ == "__main__":
    main()
