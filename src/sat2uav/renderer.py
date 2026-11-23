"""Vectorized 3D DSM heightfield ray-marching UAV camera renderer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import VideoSampleParams
from .scene import SceneBundle
from .spectral import render_spectral_radiance
from .trajectory import FramePose


@dataclass
class RenderOutput:
    """Outputs of rendering a single 3D UAV camera frame."""

    image: np.ndarray  # (H, W, 3) float32 in [0, 1]
    depth_m: np.ndarray  # (H, W) float32 3D slant distance in meters
    sky_mask: np.ndarray  # (H, W) bool
    ray_dirs_world: np.ndarray  # (H, W, 3) float32 unit world rays
    normals_hit: np.ndarray  # (H, W, 3) float32 surface normals at hit points
    sun_vec_world: np.ndarray  # (3,) float32 unit sun vector
    sun_uv_norm: tuple[float, float] | None  # Sun screen position in [0, 1]^2 if in front


class DSMCameraRenderer:
    """Physically grounded 3D camera renderer over a metric DSM heightfield.

    Supports exact horizontal FoV, oblique pitch/yaw, radial & tangential lens
    distortion (#7), true 3D terrain parallax/occlusion via coarse-to-fine ray
    marching, and per-pixel depth buffer generation.
    """

    def __init__(
        self,
        scene: SceneBundle,
        params: VideoSampleParams,
        coarse_steps: int = 36,
        refine_steps: int = 6,
    ) -> None:
        self.scene = scene
        self.params = params
        self.coarse_steps = coarse_steps
        self.refine_steps = refine_steps

        # Precompute static DSM cast shadow map for this video's sun angle
        self.shadow_map = scene.compute_shadow_map(
            sun_azimuth_deg=params.sun_azimuth_deg,
            sun_elevation_deg=params.sun_elevation_deg,
        )

        # Sun direction unit vector (ENU: +East, +North, +Up)
        az = math.radians(params.sun_azimuth_deg)
        el = math.radians(params.sun_elevation_deg)
        self.sun_vec = np.array(
            [math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)],
            dtype=np.float32,
        )

        # Precompute camera-space ray directions including radial/tangential distortion (#7)
        self.cam_rays, self.focal_px = self._build_camera_rays(
            width=params.width,
            height=params.height,
            fov_deg=params.fov_deg,
            k1=params.k1,
            k2=params.k2,
            p1=params.p1,
            p2=params.p2,
        )

        # Precompute camera-to-world rotation matrix for fixed yaw/pitch
        self.rot_c2w = self._camera_to_world_matrix(params.yaw_deg, params.pitch_deg)
        # Rotate all rays to world space once (since yaw/pitch are fixed per clip)
        self.world_rays = np.einsum("ij,hwj->hwi", self.rot_c2w, self.cam_rays).astype(
            np.float32
        )

        # Ray-march depth on a sub-grid for HD/4K (4x-16x faster) while sampling
        # textures at 100% native full resolution.
        if params.width >= 3000:
            self.ray_scale = 0.25
        elif params.width >= 1000:
            self.ray_scale = 0.5
        else:
            self.ray_scale = 1.0

        if self.ray_scale < 1.0:
            ws = max(64, int(round(params.width * self.ray_scale)))
            hs = max(64, int(round(params.height * self.ray_scale)))
            sub = cv2.resize(self.world_rays, (ws, hs), interpolation=cv2.INTER_LINEAR)
            sub /= np.linalg.norm(sub, axis=-1, keepdims=True) + 1e-8
            self.march_rays_world = sub.astype(np.float32)
        else:
            self.march_rays_world = self.world_rays

        # Project sun vector into camera frame to check for lens flare alignment (#15)
        sun_cam = self.rot_c2w.T @ self.sun_vec
        if sun_cam[2] > 0.05:
            u = 0.5 + (sun_cam[0] / sun_cam[2]) * (self.focal_px / params.width)
            v = 0.5 + (sun_cam[1] / sun_cam[2]) * (self.focal_px / params.height)
            self.sun_uv_norm: tuple[float, float] | None = (float(u), float(v))
        else:
            self.sun_uv_norm = None

    @staticmethod
    def _build_camera_rays(
        width: int,
        height: int,
        fov_deg: float,
        k1: float,
        k2: float,
        p1: float,
        p2: float,
    ) -> tuple[np.ndarray, float]:
        """Build unit rays in OpenCV camera frame (+X right, +Y down, +Z forward)
        with Brown-Conrady radial & tangential lens distortion applied.
        """
        fx = 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)
        cx = 0.5 * (width - 1)
        cy = 0.5 * (height - 1)

        u = np.arange(width, dtype=np.float32)
        v = np.arange(height, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        x = (uu - cx) / fx
        y = (vv - cy) / fx

        # Apply Brown-Conrady distortion model (#7)
        if abs(k1) > 1e-6 or abs(k2) > 1e-6 or abs(p1) > 1e-6 or abs(p2) > 1e-6:
            r2 = x * x + y * y
            r4 = r2 * r2
            radial = 1.0 + k1 * r2 + k2 * r4
            xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        else:
            xd, yd = x, y

        z = np.ones_like(xd, dtype=np.float32)
        rays = np.stack([xd, yd, z], axis=-1)
        norm = np.linalg.norm(rays, axis=-1, keepdims=True) + 1e-8
        return (rays / norm).astype(np.float32), float(fx)

    @staticmethod
    def _camera_to_world_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
        """Build 3x3 rotation matrix from OpenCV camera frame (+X right, +Y down, +Z forward)
        to world ENU frame (+X East, +Y North, +Z Up).

        yaw_deg   : 0 = North, 90 = East (clockwise)
        pitch_deg : -90 = straight down (nadir), 0 = horizontal forward
        """
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)

        # Forward optical axis (+Z_cam) in ENU
        fwd = np.array(
            [
                math.cos(pitch) * math.sin(yaw),
                math.cos(pitch) * math.cos(yaw),
                math.sin(pitch),
            ],
            dtype=np.float32,
        )
        # Right axis (+X_cam) stays level with horizon (zero roll)
        right = np.array([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float32)
        # Down axis (+Y_cam) = cross(fwd, right) in right-handed ENU? Check:
        # At yaw=0 (North: fwd=[0,1,0]), right=[1,0,0]: cross(fwd, right)=[0,0,-1] (Down!).
        # At pitch=-90 (nadir: fwd=[0,0,-1]), right=[1,0,0]: cross(fwd, right)=[0,-1,0] (South = bottom of screen!).
        down = np.cross(fwd, right)
        down = down / (np.linalg.norm(down) + 1e-8)

        return np.stack([right, down, fwd], axis=1).astype(np.float32)

    def _march_rays(
        self, cam_x: float, cam_y: float, cam_z: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect all camera rays with the 3D DSM heightfield.

        Returns (hit_col, hit_row, depth_m, sky_mask).
        """
        sc = self.scene
        dx_full = self.world_rays[:, :, 0]
        dy_full = self.world_rays[:, :, 1]
        dz_full = self.world_rays[:, :, 2]

        h_full, w_full = dz_full.shape
        sky_mask_full = dz_full >= -0.015

        # Fast flat-plane analytic solution when DSM has negligible relief
        if not sc.has_real_dsm or (sc.z_max_m - sc.z_min_m) < 0.5:
            safe_dz = np.where(sky_mask_full, -1.0, dz_full)
            t_hit = np.maximum(1.0, (sc.z_min_m - cam_z) / safe_dz)
            t_hit = np.where(sky_mask_full, 50000.0, t_hit)
            hx = cam_x + t_hit * dx_full
            hy = cam_y + t_hit * dy_full
            col, row = sc.world_to_pixel(hx, hy)
            col = np.clip(col, 0.0, sc.width_px - 1.0).astype(np.float32)
            row = np.clip(row, 0.0, sc.height_px - 1.0).astype(np.float32)
            return col, row, t_hit.astype(np.float32), sky_mask_full

        # March rays on march_rays_world (sub-grid for HD/4K)
        dx = self.march_rays_world[:, :, 0]
        dy = self.march_rays_world[:, :, 1]
        dz = self.march_rays_world[:, :, 2]
        h_out, w_out = dz.shape
        sky_mask = dz >= -0.015

        # Compute bounding-slab intersection [z_min, z_max] for downward rays
        safe_dz = np.where(sky_mask, -1.0, dz)
        t_enter = np.maximum(1.0, (sc.z_max_m - cam_z) / safe_dz)
        t_exit = np.minimum(12000.0, np.maximum(t_enter + 5.0, (sc.z_min_m - cam_z) / safe_dz))

        dsm = sc.dsm
        h_dsm, w_dsm = dsm.shape
        inv_gsd = 1.0 / sc.gsd_m
        x_min = sc.x_min_m
        y_max = sc.y_max_m

        # Coarse vectorized ray-march to bracket first surface hit
        t_lo = t_enter.copy()
        t_hi = t_exit.copy()
        hit_found = np.zeros((h_out, w_out), dtype=bool)
        dt = (t_exit - t_enter) / float(self.coarse_steps)

        t_prev = t_enter.copy()
        for step in range(1, self.coarse_steps + 1):
            t_cur = t_enter + step * dt
            active = (~sky_mask) & (~hit_found)
            if not np.any(active):
                break
            px = cam_x + t_cur * dx
            py = cam_y + t_cur * dy
            pz = cam_z + t_cur * dz

            c_i = np.clip(((px - x_min) * inv_gsd).astype(np.int32), 0, w_dsm - 1)
            r_i = np.clip(((y_max - py) * inv_gsd).astype(np.int32), 0, h_dsm - 1)
            terrain_z = dsm[r_i, c_i]

            crossed = active & (pz <= terrain_z)
            t_lo = np.where(crossed, t_prev, t_lo)
            t_hi = np.where(crossed, t_cur, t_hi)
            hit_found |= crossed
            t_prev = t_cur

        # Any downward ray that didn't cross inside slab hits bottom plane
        t_lo = np.where((~sky_mask) & (~hit_found), t_exit - dt, t_lo)
        t_hi = np.where((~sky_mask) & (~hit_found), t_exit, t_hi)

        # Binary-search refinement for crisp 3D ridges, canyons & roofs
        for _ in range(self.refine_steps):
            t_mid = 0.5 * (t_lo + t_hi)
            px = cam_x + t_mid * dx
            py = cam_y + t_mid * dy
            pz = cam_z + t_mid * dz

            c_i = np.clip(((px - x_min) * inv_gsd).astype(np.int32), 0, w_dsm - 1)
            r_i = np.clip(((y_max - py) * inv_gsd).astype(np.int32), 0, h_dsm - 1)
            below = pz <= dsm[r_i, c_i]

            t_hi = np.where(below, t_mid, t_hi)
            t_lo = np.where(below, t_lo, t_mid)

        # Keep finite t_exit on sky rays before upsampling to prevent horizon bleed
        t_sub = np.where(sky_mask, t_exit, 0.5 * (t_lo + t_hi)).astype(np.float32)
        if self.ray_scale < 1.0:
            t_full = cv2.resize(t_sub, (w_full, h_full), interpolation=cv2.INTER_LINEAR)
        else:
            t_full = t_sub

        t_final = np.where(sky_mask_full, 50000.0, t_full).astype(np.float32)
        hx = cam_x + t_final * dx_full
        hy = cam_y + t_final * dy_full

        col = np.clip((hx - x_min) * inv_gsd, 0.0, w_dsm - 1.0).astype(np.float32)
        row = np.clip((y_max - hy) * inv_gsd, 0.0, h_dsm - 1.0).astype(np.float32)
        return col, row, t_final, sky_mask_full

    def render_frame(self, pose: FramePose) -> RenderOutput:
        """Render a single 3D camera frame at the given UAV pose."""
        col, row, depth_m, sky_mask = self._march_rays(
            pose.x_m, pose.y_m, pose.z_msl_m
        )

        # Bilinearly sample multispectral imagery & geometry maps via cv2.remap
        rgba_nir = cv2.remap(
            self.scene.imagery,
            col,
            row,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        dsm_z = cv2.remap(
            self.scene.dsm,
            col,
            row,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        normals_hit = cv2.remap(
            self.scene.normals,
            col,
            row,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        ndvi_hit = cv2.remap(
            self.scene.ndvi,
            col,
            row,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        shadow_hit = cv2.remap(
            self.shadow_map,
            col,
            row,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        image = render_spectral_radiance(
            rgba_nir=rgba_nir,
            dsm_z=dsm_z,
            normals=normals_hit,
            ndvi=ndvi_hit,
            shadow=shadow_hit,
            sky_mask=sky_mask,
            sun_vec=self.sun_vec,
            spectral_mode=self.params.spectral_mode,
            is_night=self.params.is_night,
            shadow_strength=self.params.shadow_strength,
            z_min_m=self.scene.z_min_m,
            z_max_m=self.scene.z_max_m,
        )

        return RenderOutput(
            image=image,
            depth_m=depth_m,
            sky_mask=sky_mask,
            ray_dirs_world=self.world_rays,
            normals_hit=normals_hit,
            sun_vec_world=self.sun_vec,
            sun_uv_norm=self.sun_uv_norm,
        )
