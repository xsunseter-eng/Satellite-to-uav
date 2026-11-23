"""Optical, atmospheric, sensor, and footage-defect post-processing pipeline.

Implements all approved variations:
- #6  Motion blur (velocity-coupled) & CMOS rolling-shutter skew
- #8  Auto-exposure temporal lag & dynamic range clipping
- #9  Sensor noise (optical ISO/shot noise & thermal FPN striping)
- #12 Atmospheric haze / fog / visibility (Beer-Lambert using exact 3D depth)
- #15 Sun glare / lens flare & Blinn-Phong surface specular glint
- #16 Color temperature (Kelvin) & white balance shift
- Baseline defects: procedural lens cracks & water marks / droplets
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import VideoSampleParams
from .renderer import RenderOutput
from .trajectory import FramePose


def _kelvin_to_rgb_gains(kelvin: float) -> np.ndarray:
    """Approximate Planckian color temperature white-balance gains relative to 6500K."""
    k = float(np.clip(kelvin, 2500.0, 12000.0)) / 100.0
    if k <= 66.0:
        r = 255.0
        g = 99.4708025861 * math.log(max(k, 1.0)) - 161.1195681661
        b = 0.0 if k <= 19.0 else 138.5177312231 * math.log(k - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * ((k - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((k - 60.0) ** -0.0755148492)
        b = 255.0

    rgb = np.clip(np.array([r, g, b], dtype=np.float32), 1.0, 255.0)
    rgb /= np.mean(rgb)
    # Dampen extremes so shifts look like realistic UAV white balance
    return (0.45 + 0.55 * rgb).astype(np.float32)


def _generate_lens_crack_overlay(
    height: int, width: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate persistent procedural lens crack masks: (dark_alpha, glint_alpha)."""
    dark = np.zeros((height, width), dtype=np.float32)
    glint = np.zeros((height, width), dtype=np.float32)

    # Impact origin near a corner or off-center edge
    ox = int(rng.uniform(0.12 * width, 0.88 * width))
    oy = int(rng.uniform(0.12 * height, 0.88 * height))
    num_branches = int(rng.integers(4, 9))
    base_thick = max(1, int(round(min(width, height) / 480.0)))

    for _ in range(num_branches):
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        cx, cy = float(ox), float(oy)
        steps = int(rng.integers(8, 18))
        seg_len = float(rng.uniform(0.025, 0.065) * max(width, height))
        for s in range(steps):
            angle += float(rng.uniform(-0.35, 0.35))
            nx = cx + math.cos(angle) * seg_len
            ny = cy + math.sin(angle) * seg_len
            thick = max(1, base_thick if s < 4 else 1)
            cv2.line(dark, (int(cx), int(cy)), (int(nx), int(ny)), 0.72, thick, cv2.LINE_AA)
            cv2.line(
                glint,
                (int(cx + 1), int(cy + 1)),
                (int(nx + 1), int(ny + 1)),
                0.45,
                1,
                cv2.LINE_AA,
            )
            # Occasional secondary fork
            if s > 2 and rng.random() < 0.28:
                fa = angle + float(rng.choice([-1.0, 1.0]) * rng.uniform(0.4, 0.9))
                fx = nx + math.cos(fa) * seg_len * 1.4
                fy = ny + math.sin(fa) * seg_len * 1.4
                cv2.line(dark, (int(nx), int(ny)), (int(fx), int(fy)), 0.50, 1, cv2.LINE_AA)
            cx, cy = nx, ny
            if cx < 0 or cx >= width or cy < 0 or cy >= height:
                break

    # Tiny impact chip at origin
    cv2.circle(dark, (ox, oy), max(2, base_thick * 2), 0.85, -1, cv2.LINE_AA)
    dark = cv2.GaussianBlur(dark, (3, 3), 0.6)
    glint = cv2.GaussianBlur(glint, (3, 3), 0.7)
    return np.clip(dark, 0.0, 1.0), np.clip(glint, 0.0, 1.0)


def _generate_water_droplets_overlay(
    height: int, width: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate persistent lens water droplets / watermarks: (blur_mask, rim_dark, rim_bright)."""
    blur_mask = np.zeros((height, width), dtype=np.float32)
    rim_dark = np.zeros((height, width), dtype=np.float32)
    rim_bright = np.zeros((height, width), dtype=np.float32)

    num_drops = int(rng.integers(8, 22))
    scale = min(width, height)

    for _ in range(num_drops):
        cx = int(rng.uniform(0.05 * width, 0.95 * width))
        cy = int(rng.uniform(0.05 * height, 0.95 * height))
        rx = int(rng.uniform(0.012 * scale, 0.048 * scale))
        ry = int(rx * rng.uniform(0.80, 1.25))
        rot = float(rng.uniform(0.0, 180.0))
        strength = float(rng.uniform(0.35, 0.85))

        cv2.ellipse(blur_mask, (cx, cy), (rx, ry), rot, 0, 360, strength, -1, cv2.LINE_AA)
        cv2.ellipse(
            rim_dark,
            (cx, cy),
            (rx, ry),
            rot,
            0,
            360,
            strength * 0.22,
            max(1, rx // 7),
            cv2.LINE_AA,
        )
        # Subtle bright specular crescent inside droplet
        cv2.ellipse(
            rim_bright,
            (cx - rx // 4, cy - ry // 4),
            (max(1, rx // 3), max(1, ry // 4)),
            rot,
            0,
            360,
            strength * 0.25,
            -1,
            cv2.LINE_AA,
        )

    blur_mask = cv2.GaussianBlur(blur_mask, (9, 9), 2.5)
    rim_dark = cv2.GaussianBlur(rim_dark, (5, 5), 1.2)
    rim_bright = cv2.GaussianBlur(rim_bright, (7, 7), 1.8)
    return (
        np.clip(blur_mask, 0.0, 1.0),
        np.clip(rim_dark, 0.0, 1.0),
        np.clip(rim_bright, 0.0, 1.0),
    )


@dataclass
class _AutoExposureState:
    current_gain: float = 1.0
    initialized: bool = False


class EffectsPipeline:
    """Stateful per-video sensor, atmosphere, optical, and artifact pipeline."""

    def __init__(self, params: VideoSampleParams) -> None:
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self.ae_state = _AutoExposureState()

        h, w = params.height, params.width

        # Precompute persistent thermal Fixed-Pattern Noise (FPN) column + row striping (#9)
        col_fpn = self.rng.normal(0.0, params.thermal_fpn, size=(1, w, 1)).astype(np.float32)
        row_fpn = self.rng.normal(0.0, params.thermal_fpn * 0.4, size=(h, 1, 1)).astype(np.float32)
        self.thermal_fpn_map = col_fpn + row_fpn

        # Precompute persistent lens cracks / water droplet masks if active
        self.crack_overlay = (
            _generate_lens_crack_overlay(h, w, self.rng) if params.has_lens_crack else None
        )
        self.droplet_overlay = (
            _generate_water_droplets_overlay(h, w, self.rng)
            if params.has_watermark_droplets
            else None
        )

        # Precompute Kelvin white-balance gain (#16)
        self.wb_gains = _kelvin_to_rgb_gains(params.color_temp_k)
        # Apply green/magenta tint
        self.wb_gains[1] *= float(1.0 + params.wb_tint)
        self.wb_gains /= float(np.mean(self.wb_gains))

        # Precompute normalized screen grid for sun flare (#15)
        u = np.linspace(0.0, 1.0, w, dtype=np.float32)
        v = np.linspace(0.0, 1.0, h, dtype=np.float32)
        self.uu, self.vv = np.meshgrid(u, v)

        # Precompute base pixel coordinate maps for fast rolling shutter (#6)
        self._base_map_x = np.repeat(np.arange(w, dtype=np.float32)[None, :], h, axis=0)
        self._base_map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
        self._row_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)

        # Precompute a padded Gaussian noise bank for zero-copy random crop slicing (<1 ms/frame)
        noise_ch = 1 if params.spectral_mode in ("panchromatic", "nir", "thermal_lwir") else 3
        self._noise_pad = 64
        self._noise_bank = self.rng.normal(
            0.0, 1.0, size=(h + self._noise_pad, w + self._noise_pad, noise_ch)
        ).astype(np.float32)

    def apply(
        self,
        render_out: RenderOutput,
        pose: FramePose,
        focal_px: float,
    ) -> np.ndarray:
        """Apply all enabled effects and return uint8 RGB image (H, W, 3)."""
        p = self.params
        img = render_out.image.copy()

        # 1. Atmospheric Haze / Fog (#12) using exact 3D depth buffer + ABL scale height
        if p.haze_enabled:
            img = self._apply_haze(img, render_out.depth_m, render_out.sky_mask, pose.z_agl_m)

        # 2. Sun Glare, Lens Flare & Specular Glint (#15)
        if p.sun_glare_enabled and p.spectral_mode not in ("thermal_lwir", "night_vision"):
            img = self._apply_glare_and_glint(img, render_out)

        # 3. Color Temperature & White Balance Shift (#16)
        if p.spectral_mode in ("rgb", "false_color_cir"):
            img = np.clip(img * self.wb_gains[None, None, :], 0.0, 1.0)

        # 4. Velocity-Coupled Motion Blur & Rolling Shutter (#6)
        if p.motion_blur_enabled:
            img = self._apply_motion_blur_and_rolling_shutter(img, pose, focal_px)

        # 5. Auto-Exposure with Temporal Lag & Dynamic Range Clipping (#8)
        if p.auto_exposure_enabled:
            img = self._apply_auto_exposure(img)

        # 6. Sensor Noise (#9: ISO photon/read noise or Thermal FPN)
        if p.sensor_noise_enabled:
            img = self._apply_sensor_noise(img)

        # 7. Baseline Footage Defects: Water Marks / Droplets & Lens Cracks
        if self.droplet_overlay is not None:
            img = self._apply_water_droplets(img)
        if self.crack_overlay is not None:
            img = self._apply_lens_crack(img)

        return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)

    def apply_post_only(
        self,
        img_float: np.ndarray,
        pose: FramePose,
        focal_px: float,
    ) -> np.ndarray:
        """Apply screen-space post effects (#6, #8, #9, cracks, droplets) when
        spectral radiance, haze (#12), glare (#15), and WB (#16) ran on GPU.
        """
        p = self.params
        img = img_float
        if p.motion_blur_enabled:
            img = self._apply_motion_blur_and_rolling_shutter(img, pose, focal_px)
        if p.auto_exposure_enabled:
            img = self._apply_auto_exposure(img)
        if p.sensor_noise_enabled:
            img = self._apply_sensor_noise(img)
        if self.droplet_overlay is not None:
            img = self._apply_water_droplets(img)
        if self.crack_overlay is not None:
            img = self._apply_lens_crack(img)
        return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)

    def _apply_haze(
        self, img: np.ndarray, depth_m: np.ndarray, sky_mask: np.ndarray, z_agl_m: float = 250.0
    ) -> np.ndarray:
        p = self.params
        vis_km = max(1.0, p.visibility_km)
        vis_m = vis_km * 1000.0
        # Surface Koschmieder extinction beta_0 = 3.912 / V_m integrated over exponential ABL scale height
        beta = 3.912 / vis_m
        # Thermal LWIR (8-14 um) penetrates haze/mist much better than visible wavelengths
        if p.spectral_mode == "thermal_lwir":
            beta *= 0.18
        elif p.spectral_mode == "nir":
            beta *= 0.65

        # Shallow fog/mist layer (~250-450m) vs thicker clear-air boundary-layer aerosol (~1000m)
        scale_h = 220.0 + 780.0 * float(np.clip((vis_km - 1.5) / 6.5, 0.0, 1.0))
        dz = max(float(z_agl_m), 1.0)
        abl_ratio = (scale_h / dz) * (1.0 - math.exp(-dz / scale_h))
        d_eff = np.clip(depth_m, 0.0, 20000.0) * abl_ratio
        # Enforce MVFR operational optical depth ceiling (tau <= 1.05 -> transmittance >= 0.35)
        tau = np.minimum(beta * d_eff, 1.05)
        trans = np.exp(-tau).astype(np.float32)
        trans = np.where(sky_mask, 0.0, trans)

        if p.spectral_mode == "thermal_lwir":
            airlight = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        elif p.spectral_mode == "night_vision":
            airlight = np.array([0.05, 0.24, 0.07], dtype=np.float32)
        elif p.is_night:
            airlight = np.array([0.06, 0.08, 0.13], dtype=np.float32)
        elif p.spectral_mode in ("panchromatic", "nir"):
            airlight = np.array([0.74, 0.74, 0.74], dtype=np.float32)
        else:
            airlight = np.array([0.72, 0.78, 0.86], dtype=np.float32)

        return img * trans[:, :, None] + airlight[None, None, :] * (1.0 - trans[:, :, None])

    def _apply_glare_and_glint(
        self, img: np.ndarray, render_out: RenderOutput
    ) -> np.ndarray:
        p = self.params

        # Blinn-Phong specular glint on flat/reflective terrain patches
        if p.specular_glint > 0.01:
            view_to_cam = -render_out.ray_dirs_world
            half_vec = view_to_cam + render_out.sun_vec_world[None, None, :]
            half_vec = half_vec / (np.linalg.norm(half_vec, axis=-1, keepdims=True) + 1e-8)
            ndoth = np.clip(
                np.sum(render_out.normals_hit * half_vec, axis=-1), 0.0, 1.0
            )
            spec = (ndoth ** 42.0) * p.specular_glint * (~render_out.sky_mask).astype(np.float32)
            # Modulate by local flatness & brightness
            spec = np.clip(spec, 0.0, 0.85)
            tint = np.array([1.0, 0.96, 0.86], dtype=np.float32)
            img = np.clip(img + spec[:, :, None] * tint[None, None, :], 0.0, 1.0)

        # Optical sun glare / lens flare when sun lies in or near forward frustum
        if p.glare_intensity > 0.01 and render_out.sun_uv_norm is not None:
            su, sv = render_out.sun_uv_norm
            # Active even slightly outside frame bounds (off-axis veiling glare)
            if -0.45 <= su <= 1.45 and -0.45 <= sv <= 1.45:
                aspect = p.width / float(p.height)
                du = (self.uu - su) * aspect
                dv = self.vv - sv
                dist = np.sqrt(du * du + dv * dv)
                halo = np.exp(-dist * 3.8) * p.glare_intensity * 0.65
                core = np.exp(-(dist ** 2) * 45.0) * p.glare_intensity * 0.55

                # Subtle secondary lens ghost opposite optical center
                gu, gv = 1.0 - su, 1.0 - sv
                gdist = np.sqrt(((self.uu - gu) * aspect) ** 2 + (self.vv - gv) ** 2)
                ghost = np.exp(-(gdist ** 2) * 32.0) * p.glare_intensity * 0.22

                flare_warm = np.array([1.0, 0.92, 0.78], dtype=np.float32)
                ghost_cool = np.array([0.75, 0.88, 1.0], dtype=np.float32)
                img = np.clip(
                    img
                    + (halo + core)[:, :, None] * flare_warm[None, None, :]
                    + ghost[:, :, None] * ghost_cool[None, None, :],
                    0.0,
                    1.0,
                )
        return img

    def _apply_motion_blur_and_rolling_shutter(
        self, img: np.ndarray, pose: FramePose, focal_px: float
    ) -> np.ndarray:
        p = self.params
        h, w = img.shape[:2]

        # Estimate vertical screen-space ground flow (pixels/frame) from forward flight
        pitch_rad = abs(math.radians(pose.pitch_deg))
        mean_slant = max(25.0, pose.z_agl_m / max(math.sin(pitch_rad), 0.25))
        px_per_sec = (pose.speed_mps / mean_slant) * focal_px * max(math.sin(pitch_rad), 0.35)
        blur_len_px = float(np.clip(px_per_sec * p.shutter_sec, 0.0, 15.0))

        if blur_len_px >= 1.5:
            ksize = int(round(blur_len_px)) | 1  # Odd kernel size >= 3
            ksize = max(3, min(ksize, 15))
            # Exact vertical 1D motion blur using O(1) separable cv2.blur (15x faster than 2D filter2D)
            img = cv2.blur(img, (1, ksize), borderType=cv2.BORDER_REPLICATE)

        # CMOS rolling-shutter subtle row-wise horizontal wobble (#6)
        if p.rolling_shutter_strength > 0.02:
            t_phase = pose.time_sec * 18.0
            amp_px = p.rolling_shutter_strength * (w / 640.0) * 1.8
            shift_x = (amp_px * np.sin(2.0 * math.pi * (self._row_norm * 1.5 + t_phase))).astype(np.float32)
            map_x = self._base_map_x + shift_x[:, None]
            img = cv2.remap(
                img,
                map_x,
                self._base_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        return img

    def _apply_auto_exposure(self, img: np.ndarray) -> np.ndarray:
        p = self.params
        sub = img[::4, ::4]
        mean_luma = float(
            np.clip(
                0.299 * np.mean(sub[:, :, 0])
                + 0.587 * np.mean(sub[:, :, 1])
                + 0.114 * np.mean(sub[:, :, 2]),
                0.04,
                0.95,
            )
        )

        # Night & NVG modes target lower luma so highlights don't clip into neon/daytime
        if p.spectral_mode == "night_vision":
            target = p.ae_target_luma * 0.62
        elif p.is_night and p.spectral_mode == "rgb":
            target = p.ae_target_luma * 0.45
        else:
            target = p.ae_target_luma
        desired_gain = float(np.clip(target / mean_luma, 0.45, 2.4))

        if not self.ae_state.initialized:
            # Start slightly off-target to exercise realistic initial AE convergence
            self.ae_state.current_gain = float(
                np.clip(desired_gain * self.rng.uniform(0.85, 1.15), 0.45, 2.4)
            )
            self.ae_state.initialized = True
        else:
            a = p.ae_lag_alpha
            self.ae_state.current_gain = (1.0 - a) * self.ae_state.current_gain + a * desired_gain

        img = img * self.ae_state.current_gain

        # Dynamic range clipping / contrast punch (simulates limited UAV sensor DR)
        if p.dr_clip_strength > 0.01:
            black_pt = 0.08 * p.dr_clip_strength
            white_pt = 1.0 - 0.14 * p.dr_clip_strength
            img = (img - black_pt) * (1.0 / max(white_pt - black_pt, 0.2))

        return np.clip(img, 0.0, 1.0, out=img)

    def _apply_sensor_noise(self, img: np.ndarray) -> np.ndarray:
        p = self.params
        h, w, _ = img.shape
        sy = int(self.rng.integers(0, self._noise_pad))
        sx = int(self.rng.integers(0, self._noise_pad))
        # Zero-copy slice from pre-generated Gaussian bank
        noise = self._noise_bank[sy : sy + h, sx : sx + w]

        if p.spectral_mode == "thermal_lwir":
            img = img + self.thermal_fpn_map + noise * (p.thermal_fpn * 0.55)
            return np.clip(img, 0.0, 1.0, out=img)

        iso = p.iso_noise * (1.8 if p.is_night else 1.0)
        if p.spectral_mode in ("panchromatic", "nir"):
            scale = (np.sqrt(np.clip(img[:, :, :1], 0.01, 1.0)) * (iso * 0.7)) + (iso * 0.5)
            img = img + noise * scale
            return np.clip(img, 0.0, 1.0, out=img)

        scale = (np.sqrt(np.clip(img, 0.01, 1.0)) * (iso * 0.7)) + (iso * 0.5)
        img = img + noise * scale
        return np.clip(img, 0.0, 1.0, out=img)

    def _apply_water_droplets(self, img: np.ndarray) -> np.ndarray:
        assert self.droplet_overlay is not None
        blur_mask, rim_dark, rim_bright = self.droplet_overlay
        blurred = cv2.GaussianBlur(img, (15, 15), 4.5)
        out = img * (1.0 - blur_mask[:, :, None]) + blurred * blur_mask[:, :, None]
        out = out * (1.0 - rim_dark[:, :, None]) + rim_bright[:, :, None] * 0.35
        return np.clip(out, 0.0, 1.0)

    def _apply_lens_crack(self, img: np.ndarray) -> np.ndarray:
        assert self.crack_overlay is not None
        dark, glint = self.crack_overlay
        out = img * (1.0 - dark[:, :, None]) + glint[:, :, None] * 0.55
        return np.clip(out, 0.0, 1.0)
