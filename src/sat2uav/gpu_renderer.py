"""Headless GPU OpenGL (ModernGL + EGL) 3D DSM Render Engines.

Provides two hardware-accelerated rendering engines on NVIDIA/EGL GPUs:
1. `gpu_opengl_raymarch`: Full-resolution GLSL 3.30 3D DSM heightfield ray-marching
   shader with exact per-pixel Brown-Conrady lens distortion, hardware mipmaps, and
   16x anisotropic texture filtering.
2. `gpu_opengl_mesh`: True 3D displaced heightfield triangle mesh rasterizer using
   OpenGL hardware Z-buffer + mipmaps + 16x anisotropic filtering, followed by GPU
   lens-distortion mapping.
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import moderngl
import numpy as np

from .config import VideoSampleParams
from .renderer import DSMCameraRenderer, RenderOutput
from .scene import SceneBundle
from .spectral import render_spectral_radiance
from .trajectory import FramePose

EngineType = Literal["gpu_opengl_raymarch", "gpu_opengl_mesh", "cpu_raymarch"]

_QUAD_VERT = """
#version 330 core
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_RAYMARCH_FRAG = """
#version 330 core
in vec2 v_uv;

layout(location = 0) out vec4 out_rgba_nir;
layout(location = 1) out vec4 out_geom;   // (dsm_z, depth_m, shadow, sky_flag)
layout(location = 2) out vec4 out_norm;   // (nx, ny, nz, ndvi)

uniform sampler2D u_imagery;
uniform sampler2D u_dsm;
uniform sampler2D u_normals;
uniform sampler2D u_ndvi;
uniform sampler2D u_shadow;

uniform vec3 u_cam_pos;      // (x_m, y_m, z_msl_m)
uniform mat3 u_rot_c2w;      // camera (+X right, +Y down, +Z fwd) -> world ENU
uniform vec2 u_res;          // (width, height)
uniform float u_fx;          // focal length in pixels
uniform vec4 u_dist;         // (k1, k2, p1, p2)

uniform vec4 u_bounds;       // (x_min, y_min, x_max, y_max)
uniform vec2 u_zbounds;      // (z_min, z_max)
uniform int u_has_dsm;
uniform int u_coarse_steps;
uniform int u_refine_steps;

vec2 world_to_uv(vec2 xy) {
    float u = (xy.x - u_bounds.x) / max(u_bounds.z - u_bounds.x, 1e-4);
    // OpenGL texture row 0 = top of raster (y_max) since we upload top-down without flipping
    float v = (u_bounds.w - xy.y) / max(u_bounds.w - u_bounds.y, 1e-4);
    return clamp(vec2(u, v), vec2(0.0005), vec2(0.9995));
}

void main() {
    // v_uv.y = 0 at bottom in OpenGL; map to OpenCV top-down pixel coords
    float px_u = v_uv.x * u_res.x - 0.5;
    float px_v = (1.0 - v_uv.y) * u_res.y - 0.5;
    float cx = 0.5 * (u_res.x - 1.0);
    float cy = 0.5 * (u_res.y - 1.0);

    float x = (px_u - cx) / u_fx;
    float y = (px_v - cy) / u_fx;

    // Exact Brown-Conrady radial & tangential lens distortion (#7)
    float r2 = x * x + y * y;
    float r4 = r2 * r2;
    float rad = 1.0 + u_dist.x * r2 + u_dist.y * r4;
    float xd = x * rad + 2.0 * u_dist.z * x * y + u_dist.w * (r2 + 2.0 * x * x);
    float yd = y * rad + u_dist.z * (r2 + 2.0 * y * y) + 2.0 * u_dist.w * x * y;

    vec3 ray_cam = normalize(vec3(xd, yd, 1.0));
    vec3 ray_w = normalize(u_rot_c2w * ray_cam);

    if (ray_w.z >= -0.015) {
        out_rgba_nir = vec4(0.0);
        out_geom = vec4(u_zbounds.x, 50000.0, 0.0, 1.0);
        out_norm = vec4(0.0, 0.0, 1.0, 0.0);
        return;
    }

    float t_final = 1.0;
    if (u_has_dsm == 0 || (u_zbounds.y - u_zbounds.x) < 0.5) {
        t_final = max(1.0, (u_zbounds.x - u_cam_pos.z) / ray_w.z);
    } else {
        float t_enter = max(1.0, (u_zbounds.y - u_cam_pos.z) / ray_w.z);
        float t_exit = min(35000.0, max(t_enter + 5.0, (u_zbounds.x - u_cam_pos.z) / ray_w.z));
        float dt = (t_exit - t_enter) / float(u_coarse_steps);

        float t_lo = t_enter;
        float t_hi = t_exit;
        float t_prev = t_enter;
        bool hit = false;

        for (int i = 1; i <= 96; ++i) {
            if (i > u_coarse_steps) break;
            float t_cur = t_enter + float(i) * dt;
            vec3 p = u_cam_pos + t_cur * ray_w;
            float tz = textureLod(u_dsm, world_to_uv(p.xy), 0.0).r;
            if (p.z <= tz) {
                t_lo = t_prev;
                t_hi = t_cur;
                hit = true;
                break;
            }
            t_prev = t_cur;
        }
        if (!hit) {
            t_lo = t_exit - dt;
            t_hi = t_exit;
        }

        for (int j = 0; j < 10; ++j) {
            if (j >= u_refine_steps) break;
            float t_mid = 0.5 * (t_lo + t_hi);
            vec3 p = u_cam_pos + t_mid * ray_w;
            float tz = textureLod(u_dsm, world_to_uv(p.xy), 0.0).r;
            if (p.z <= tz) {
                t_hi = t_mid;
            } else {
                t_lo = t_mid;
            }
        }
        t_final = 0.5 * (t_lo + t_hi);
    }

    vec3 hit_p = u_cam_pos + t_final * ray_w;
    vec2 uv = world_to_uv(hit_p.xy);

    // Hardware mipmapped + 16x anisotropic texture sampling!
    vec4 tex = texture(u_imagery, uv);
    float dsm_z = texture(u_dsm, uv).r;
    vec3 norm_w = normalize(texture(u_normals, uv).rgb);
    float ndvi = texture(u_ndvi, uv).r;
    float shd = texture(u_shadow, uv).r;

    out_rgba_nir = tex;
    out_geom = vec4(dsm_z, t_final, shd, 0.0);
    out_norm = vec4(norm_w, ndvi);
}
"""

_MESH_VERT = """
#version 330 core
in vec2 in_uv;       // [0, 1]^2 across scene raster

uniform sampler2D u_dsm;
uniform vec4 u_bounds;   // (x_min, y_min, x_max, y_max)
uniform mat4 u_mvp;
uniform vec3 u_cam_pos;

out vec2 v_uv;
out float v_depth;
out float v_z;

void main() {
    float wx = mix(u_bounds.x, u_bounds.z, in_uv.x);
    float wy = mix(u_bounds.w, u_bounds.y, in_uv.y);
    float wz = textureLod(u_dsm, in_uv, 0.0).r;
    vec3 wp = vec3(wx, wy, wz);
    v_uv = in_uv;
    v_z = wz;
    v_depth = length(wp - u_cam_pos);
    gl_Position = u_mvp * vec4(wp, 1.0);
}
"""

_MESH_FRAG = """
#version 330 core
in vec2 v_uv;
in float v_depth;
in float v_z;

layout(location = 0) out vec4 out_rgba_nir;
layout(location = 1) out vec4 out_geom;
layout(location = 2) out vec4 out_norm;

uniform sampler2D u_imagery;
uniform sampler2D u_normals;
uniform sampler2D u_ndvi;
uniform sampler2D u_shadow;

void main() {
    out_rgba_nir = texture(u_imagery, v_uv);
    out_geom = vec4(v_z, v_depth, texture(u_shadow, v_uv).r, 0.0);
    out_norm = vec4(normalize(texture(u_normals, v_uv).rgb), texture(u_ndvi, v_uv).r);
}
"""


def _upload_scene_textures(
    ctx: moderngl.Context, scene: SceneBundle, shadow_map: np.ndarray
) -> dict[str, moderngl.Texture]:
    h, w = scene.height_px, scene.width_px

    # Imagery (4-band float32 -> RGBA8 with mipmaps & 16x anisotropic filtering)
    img_u8 = np.ascontiguousarray(np.clip(scene.imagery * 255.0, 0, 255).astype(np.uint8))
    t_img = ctx.texture((w, h), 4, img_u8.tobytes(), dtype="f1")
    t_img.build_mipmaps()
    t_img.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    t_img.anisotropy = 16.0
    t_img.repeat_x = False
    t_img.repeat_y = False

    t_dsm = ctx.texture((w, h), 1, np.ascontiguousarray(scene.dsm, dtype=np.float32).tobytes(), dtype="f4")
    t_dsm.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_dsm.repeat_x = False
    t_dsm.repeat_y = False

    t_norm = ctx.texture((w, h), 3, np.ascontiguousarray(scene.normals, dtype=np.float32).tobytes(), dtype="f4")
    t_norm.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_norm.repeat_x = False
    t_norm.repeat_y = False

    t_ndvi = ctx.texture((w, h), 1, np.ascontiguousarray(scene.ndvi, dtype=np.float32).tobytes(), dtype="f4")
    t_ndvi.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_ndvi.repeat_x = False
    t_ndvi.repeat_y = False

    t_shd = ctx.texture((w, h), 1, np.ascontiguousarray(shadow_map, dtype=np.float32).tobytes(), dtype="f4")
    t_shd.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_shd.repeat_x = False
    t_shd.repeat_y = False

    return {
        "imagery": t_img,
        "dsm": t_dsm,
        "normals": t_norm,
        "ndvi": t_ndvi,
        "shadow": t_shd,
    }


def _create_mrt_fbo(ctx: moderngl.Context, w: int, h: int, with_depth: bool = False):
    c0 = ctx.texture((w, h), 4, dtype="f4")
    c1 = ctx.texture((w, h), 4, dtype="f4")
    c2 = ctx.texture((w, h), 4, dtype="f4")
    depth = ctx.depth_renderbuffer((w, h)) if with_depth else None
    fbo = ctx.framebuffer(color_attachments=[c0, c1, c2], depth_attachment=depth)
    return fbo, (c0, c1, c2), depth


class GPURaymarchRenderer:
    """Headless NVIDIA EGL OpenGL 3.30 3D DSM Ray-Marching Renderer."""

    engine_name = "gpu_opengl_raymarch"

    def __init__(
        self,
        scene: SceneBundle,
        params: VideoSampleParams,
        coarse_steps: int = 56,
        refine_steps: int = 8,
    ) -> None:
        self.scene = scene
        self.params = params
        self.coarse_steps = coarse_steps
        self.refine_steps = refine_steps

        self.shadow_map = scene.compute_shadow_map(
            sun_azimuth_deg=params.sun_azimuth_deg,
            sun_elevation_deg=params.sun_elevation_deg,
        )
        az = math.radians(params.sun_azimuth_deg)
        el = math.radians(params.sun_elevation_deg)
        self.sun_vec = np.array(
            [math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)],
            dtype=np.float32,
        )

        self.cam_rays, self.focal_px = DSMCameraRenderer._build_camera_rays(
            params.width, params.height, params.fov_deg,
            params.k1, params.k2, params.p1, params.p2,
        )
        self.rot_c2w = DSMCameraRenderer._camera_to_world_matrix(params.yaw_deg, params.pitch_deg)
        self.world_rays = np.einsum("ij,hwj->hwi", self.rot_c2w, self.cam_rays).astype(np.float32)

        sun_cam = self.rot_c2w.T @ self.sun_vec
        if sun_cam[2] > 0.05:
            u = 0.5 + (sun_cam[0] / sun_cam[2]) * (self.focal_px / params.width)
            v = 0.5 + (sun_cam[1] / sun_cam[2]) * (self.focal_px / params.height)
            self.sun_uv_norm: tuple[float, float] | None = (float(u), float(v))
        else:
            self.sun_uv_norm = None

        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.prog = self.ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_RAYMARCH_FRAG)
        self.textures = _upload_scene_textures(self.ctx, scene, self.shadow_map)
        self.fbo, self.color_texs, _ = _create_mrt_fbo(self.ctx, params.width, params.height, with_depth=False)

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_pos")

        # Bind static uniforms
        self.prog["u_imagery"].value = 0
        self.prog["u_dsm"].value = 1
        self.prog["u_normals"].value = 2
        self.prog["u_ndvi"].value = 3
        self.prog["u_shadow"].value = 4
        self.prog["u_res"].value = (float(params.width), float(params.height))
        self.prog["u_fx"].value = float(self.focal_px)
        self.prog["u_dist"].value = (params.k1, params.k2, params.p1, params.p2)
        self.prog["u_bounds"].value = (scene.x_min_m, scene.y_min_m, scene.x_max_m, scene.y_max_m)
        self.prog["u_zbounds"].value = (scene.z_min_m, scene.z_max_m)
        self.prog["u_has_dsm"].value = 1 if scene.has_real_dsm else 0
        self.prog["u_coarse_steps"].value = coarse_steps
        self.prog["u_refine_steps"].value = refine_steps
        # GLSL mat3 is column-major
        self.prog["u_rot_c2w"].write(np.ascontiguousarray(self.rot_c2w.T, dtype=np.float32).tobytes())

    def render_frame(self, pose: FramePose) -> RenderOutput:
        self.textures["imagery"].use(0)
        self.textures["dsm"].use(1)
        self.textures["normals"].use(2)
        self.textures["ndvi"].use(3)
        self.textures["shadow"].use(4)

        self.prog["u_cam_pos"].value = (float(pose.x_m), float(pose.y_m), float(pose.z_msl_m))

        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        h, w = self.params.height, self.params.width
        # OpenGL framebuffer row 0 is bottom; flip vertically to top-down OpenCV layout
        rgba_nir = np.frombuffer(self.color_texs[0].read(), dtype=np.float32).reshape(h, w, 4)[::-1].copy()
        geom = np.frombuffer(self.color_texs[1].read(), dtype=np.float32).reshape(h, w, 4)[::-1].copy()
        norm_ndvi = np.frombuffer(self.color_texs[2].read(), dtype=np.float32).reshape(h, w, 4)[::-1].copy()

        dsm_z = geom[:, :, 0]
        depth_m = geom[:, :, 1]
        shadow_hit = geom[:, :, 2]
        sky_mask = geom[:, :, 3] > 0.5
        normals_hit = norm_ndvi[:, :, :3]
        ndvi_hit = norm_ndvi[:, :, 3]

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

    def close(self) -> None:
        for t in self.textures.values():
            t.release()
        for t in self.color_texs:
            t.release()
        self.fbo.release()
        self.vao.release()
        self.vbo.release()
        self.prog.release()
        self.ctx.release()

    def __enter__(self) -> "GPURaymarchRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class GPUMeshRenderer:
    """Headless NVIDIA EGL OpenGL 3.30 3D Displaced Heightfield Triangle Mesh Rasterizer."""

    engine_name = "gpu_opengl_mesh"

    def __init__(
        self,
        scene: SceneBundle,
        params: VideoSampleParams,
        grid_res: int = 384,
    ) -> None:
        self.scene = scene
        self.params = params
        self.grid_res = grid_res

        self.shadow_map = scene.compute_shadow_map(
            sun_azimuth_deg=params.sun_azimuth_deg,
            sun_elevation_deg=params.sun_elevation_deg,
        )
        az = math.radians(params.sun_azimuth_deg)
        el = math.radians(params.sun_elevation_deg)
        self.sun_vec = np.array(
            [math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)],
            dtype=np.float32,
        )

        self.cam_rays, self.focal_px = DSMCameraRenderer._build_camera_rays(
            params.width, params.height, params.fov_deg,
            params.k1, params.k2, params.p1, params.p2,
        )
        self.rot_c2w = DSMCameraRenderer._camera_to_world_matrix(params.yaw_deg, params.pitch_deg)
        self.world_rays = np.einsum("ij,hwj->hwi", self.rot_c2w, self.cam_rays).astype(np.float32)

        sun_cam = self.rot_c2w.T @ self.sun_vec
        if sun_cam[2] > 0.05:
            u = 0.5 + (sun_cam[0] / sun_cam[2]) * (self.focal_px / params.width)
            v = 0.5 + (sun_cam[1] / sun_cam[2]) * (self.focal_px / params.height)
            self.sun_uv_norm: tuple[float, float] | None = (float(u), float(v))
        else:
            self.sun_uv_norm = None

        # Precompute post-rasterization lens distortion remap grids (#7)
        self._dist_map_x, self._dist_map_y = self._build_distortion_remap()

        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog = self.ctx.program(vertex_shader=_MESH_VERT, fragment_shader=_MESH_FRAG)
        self.textures = _upload_scene_textures(self.ctx, scene, self.shadow_map)
        self.fbo, self.color_texs, self.depth_rb = _create_mrt_fbo(
            self.ctx, params.width, params.height, with_depth=True
        )

        # Build UV grid triangle mesh (grid_res x grid_res)
        g = grid_res
        us = np.linspace(0.0, 1.0, g, dtype=np.float32)
        vs = np.linspace(0.0, 1.0, g, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)
        verts = np.stack([uu, vv], axis=-1).reshape(-1, 2)

        idx = np.arange(g * g, dtype=np.int32).reshape(g, g)
        i00 = idx[:-1, :-1].ravel()
        i10 = idx[:-1, 1:].ravel()
        i01 = idx[1:, :-1].ravel()
        i11 = idx[1:, 1:].ravel()
        tris = np.stack([i00, i01, i10, i10, i01, i11], axis=-1).ravel()

        self.vbo = self.ctx.buffer(np.ascontiguousarray(verts, dtype=np.float32).tobytes())
        self.ibo = self.ctx.buffer(np.ascontiguousarray(tris, dtype=np.int32).tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_uv")], self.ibo)

        self.prog["u_dsm"].value = 1
        self.prog["u_imagery"].value = 0
        self.prog["u_normals"].value = 2
        self.prog["u_ndvi"].value = 3
        self.prog["u_shadow"].value = 4
        self.prog["u_bounds"].value = (scene.x_min_m, scene.y_min_m, scene.x_max_m, scene.y_max_m)

    def _build_distortion_remap(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        h, w = p.height, p.width
        fx = self.focal_px
        cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        x = (uu - cx) / fx
        y = (vv - cy) / fx
        r2 = x * x + y * y
        rad = 1.0 + p.k1 * r2 + p.k2 * r2 * r2
        xd = x * rad + 2.0 * p.p1 * x * y + p.p2 * (r2 + 2.0 * x * x)
        yd = y * rad + p.p1 * (r2 + 2.0 * y * y) + 2.0 * p.p2 * x * y
        return (xd * fx + cx).astype(np.float32), (yd * fx + cy).astype(np.float32)

    def _build_mvp(self, pose: FramePose) -> np.ndarray:
        """Build OpenGL 4x4 Model-View-Projection matrix from UAV pose."""
        p = self.params
        # OpenGL camera: +X right, +Y up, -Z forward
        right = self.rot_c2w[:, 0]
        up = -self.rot_c2w[:, 1]
        fwd = self.rot_c2w[:, 2]
        eye = np.array([pose.x_m, pose.y_m, pose.z_msl_m], dtype=np.float32)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -fwd
        view[0, 3] = -np.dot(right, eye)
        view[1, 3] = -np.dot(up, eye)
        view[2, 3] = np.dot(fwd, eye)

        aspect = p.width / float(p.height)
        hfov = math.radians(p.fov_deg)
        vfov = 2.0 * math.atan(math.tan(0.5 * hfov) / aspect)
        f = 1.0 / math.tan(0.5 * vfov)
        near, far = 5.0, 45000.0

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return (proj @ view).astype(np.float32)

    def render_frame(self, pose: FramePose) -> RenderOutput:
        self.textures["imagery"].use(0)
        self.textures["dsm"].use(1)
        self.textures["normals"].use(2)
        self.textures["ndvi"].use(3)
        self.textures["shadow"].use(4)

        mvp = self._build_mvp(pose)
        self.prog["u_mvp"].write(np.ascontiguousarray(mvp.T, dtype=np.float32).tobytes())
        self.prog["u_cam_pos"].value = (float(pose.x_m), float(pose.y_m), float(pose.z_msl_m))

        self.fbo.use()
        # Clear sky marker into attachment 1 alpha = 1.0
        self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.vao.render(moderngl.TRIANGLES)

        h, w = self.params.height, self.params.width
        rgba_nir = np.frombuffer(self.color_texs[0].read(), dtype=np.float32).reshape(h, w, 4)[::-1].copy()
        geom = np.frombuffer(self.color_texs[1].read(), dtype=np.float32).reshape(h, w, 4)[::-1].copy()
        norm_ndvi = np.frombuffer(self.color_texs[2].read(), dtype=np.float32).reshape(h, w, 4)[::-1].copy()

        # Apply optical lens distortion (#7)
        p = self.params
        if abs(p.k1) > 1e-5 or abs(p.k2) > 1e-5 or abs(p.p1) > 1e-5 or abs(p.p2) > 1e-5:
            rgba_nir = cv2.remap(rgba_nir, self._dist_map_x, self._dist_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            geom = cv2.remap(geom, self._dist_map_x, self._dist_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            norm_ndvi = cv2.remap(norm_ndvi, self._dist_map_x, self._dist_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        dsm_z = geom[:, :, 0]
        depth_m = np.where(geom[:, :, 3] > 0.5, 50000.0, np.maximum(geom[:, :, 1], 1.0))
        shadow_hit = geom[:, :, 2]
        sky_mask = geom[:, :, 3] > 0.5
        normals_hit = norm_ndvi[:, :, :3]
        ndvi_hit = norm_ndvi[:, :, 3]

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

    def close(self) -> None:
        for t in self.textures.values():
            t.release()
        for t in self.color_texs:
            t.release()
        if self.depth_rb is not None:
            self.depth_rb.release()
        self.fbo.release()
        self.vao.release()
        self.vbo.release()
        self.ibo.release()
        self.prog.release()
        self.ctx.release()

    def __enter__(self) -> "GPUMeshRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def create_renderer(
    engine: EngineType,
    scene: SceneBundle,
    params: VideoSampleParams,
):
    """Factory that constructs the requested render engine with safe CPU fallback."""
    if engine == "gpu_opengl_raymarch":
        return GPURaymarchRenderer(scene, params)
    if engine == "gpu_opengl_mesh":
        return GPUMeshRenderer(scene, params)
    if engine == "cpu_raymarch":
        return DSMCameraRenderer(scene, params)
    raise ValueError(f"Unknown render engine: {engine}")
