"""Ultra-fast all-in-shader NVIDIA GTX 1050 (ModernGL EGL) 3D DSM Renderer.

Computes 3D ray-marching, lens distortion (#7), 16x anisotropic texture sampling,
all 6 spectral camera models, day/night relighting, Koschmieder 3D haze (#12),
Blinn-Phong specular glint & lens flare (#15), and Kelvin white balance (#16)
entirely on the GPU in a single fragment pass, reading back only an RGB8 buffer
(16x less PCIe bandwidth than float32 MRT readbacks).
"""

from __future__ import annotations

import math

import moderngl
import numpy as np

from .config import VideoSampleParams
from .effects import _kelvin_to_rgb_gains
from .renderer import DSMCameraRenderer
from .scene import SceneBundle
from .trajectory import FramePose

_QUAD_VERT = """
#version 330 core
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_ALL_IN_ONE_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_imagery;
uniform sampler2D u_dsm;
uniform sampler2D u_normals;
uniform sampler2D u_ndvi;
uniform sampler2D u_shadow;

uniform vec3 u_cam_pos;
uniform mat3 u_rot_c2w;
uniform vec2 u_res;
uniform float u_fx;
uniform vec4 u_dist;        // (k1, k2, p1, p2)

uniform vec4 u_bounds;      // (x_min, y_min, x_max, y_max)
uniform vec2 u_zbounds;     // (z_min, z_max)
uniform int u_has_dsm;
uniform int u_coarse_steps;
uniform int u_refine_steps;

// Spectral & lighting uniforms
uniform int u_spectral_id;  // 0=rgb, 1=pan, 2=nir, 3=cir, 4=thermal, 5=nvg
uniform int u_is_night;
uniform vec3 u_sun_vec;
uniform float u_shadow_strength;
uniform vec3 u_wb_gains;

// Atmosphere & glare uniforms
uniform int u_haze_enabled;
uniform float u_haze_beta;
uniform int u_glare_enabled;
uniform float u_glare_intensity;
uniform float u_specular_glint;
uniform int u_has_sun_uv;
uniform vec2 u_sun_uv;

vec2 world_to_uv(vec2 xy) {
    float u = (xy.x - u_bounds.x) / max(u_bounds.z - u_bounds.x, 1e-4);
    float v = (u_bounds.w - xy.y) / max(u_bounds.w - u_bounds.y, 1e-4);
    return clamp(vec2(u, v), vec2(0.0005), vec2(0.9995));
}

vec3 sky_color() {
    if (u_spectral_id == 4) return vec3(0.08);
    if (u_spectral_id == 5) return vec3(0.03, 0.14, 0.04);
    if (u_is_night == 1) return vec3(0.03, 0.04, 0.08);
    if (u_spectral_id == 1 || u_spectral_id == 2) return vec3(0.65);
    if (u_spectral_id == 3) return vec3(0.45, 0.62, 0.78);
    return vec3(0.52, 0.70, 0.90);
}

vec3 haze_airlight() {
    if (u_spectral_id == 4) return vec3(0.45);
    if (u_spectral_id == 5) return vec3(0.05, 0.24, 0.07);
    if (u_is_night == 1) return vec3(0.06, 0.08, 0.13);
    if (u_spectral_id == 1 || u_spectral_id == 2) return vec3(0.74);
    return vec3(0.72, 0.78, 0.86);
}

void main() {
    float px_u = v_uv.x * u_res.x - 0.5;
    float px_v = (1.0 - v_uv.y) * u_res.y - 0.5;
    float cx = 0.5 * (u_res.x - 1.0);
    float cy = 0.5 * (u_res.y - 1.0);

    float x = (px_u - cx) / u_fx;
    float y = (px_v - cy) / u_fx;

    // Brown-Conrady lens distortion (#7)
    float r2 = x * x + y * y;
    float rad = 1.0 + u_dist.x * r2 + u_dist.y * r2 * r2;
    float xd = x * rad + 2.0 * u_dist.z * x * y + u_dist.w * (r2 + 2.0 * x * x);
    float yd = y * rad + u_dist.z * (r2 + 2.0 * y * y) + 2.0 * u_dist.w * x * y;

    vec3 ray_w = normalize(u_rot_c2w * normalize(vec3(xd, yd, 1.0)));
    bool is_sky = (ray_w.z >= -0.015);

    vec3 col;
    float depth_m = 50000.0;
    vec3 norm_w = vec3(0.0, 0.0, 1.0);

    if (is_sky) {
        col = sky_color();
    } else {
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
            for (int i = 1; i <= 64; ++i) {
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
            if (!hit) { t_lo = t_exit - dt; t_hi = t_exit; }
            for (int j = 0; j < 8; ++j) {
                if (j >= u_refine_steps) break;
                float t_mid = 0.5 * (t_lo + t_hi);
                vec3 p = u_cam_pos + t_mid * ray_w;
                float tz = textureLod(u_dsm, world_to_uv(p.xy), 0.0).r;
                if (p.z <= tz) { t_hi = t_mid; } else { t_lo = t_mid; }
            }
            t_final = 0.5 * (t_lo + t_hi);
        }
        depth_m = t_final;
        vec3 hit_p = u_cam_pos + t_final * ray_w;
        vec2 uv = world_to_uv(hit_p.xy);

        vec4 tex = texture(u_imagery, uv);
        float dsm_z = texture(u_dsm, uv).r;
        norm_w = normalize(texture(u_normals, uv).rgb);
        float ndvi = texture(u_ndvi, uv).r;
        float shd = texture(u_shadow, uv).r;

        float r = tex.r;
        float g = tex.g;
        float b = tex.b;
        float nir = tex.a;

        float ndotl = clamp(dot(norm_w, u_sun_vec), 0.0, 1.0);
        float direct = ndotl * (1.0 - u_shadow_strength * shd);
        float sun_elev = clamp(u_sun_vec.z, 0.12, 1.0);
        float day_illum = clamp(0.32 + 0.78 * (direct / pow(sun_elev, 0.45)), 0.15, 1.25);

        float albedo = 0.299 * r + 0.587 * g + 0.114 * b;
        float urban = clamp((albedo - 0.28) * 2.5, 0.0, 1.0) * clamp((0.18 - ndvi) * 3.0, 0.0, 1.0);

        if (u_spectral_id == 0) {
            if (u_is_night == 1) {
                float moon = 0.11 + 0.14 * ndotl;
                col = vec3(r, g, b) * moon * vec3(0.62, 0.74, 1.05) + 0.55 * pow(urban, 1.6) * vec3(0.85, 0.62, 0.28);
            } else {
                col = vec3(r, g, b) * day_illum;
            }
        } else if (u_spectral_id == 1) {
            float pan = 0.26 * r + 0.50 * g + 0.12 * b + 0.12 * nir;
            float val = (u_is_night == 1) ? (pan * (0.14 + 0.12 * ndotl) + 0.45 * pow(urban, 1.5)) : (pan * day_illum);
            col = vec3(val);
        } else if (u_spectral_id == 2) {
            float val = (u_is_night == 1) ? (nir * (0.18 + 0.14 * ndotl) + 0.35 * pow(urban, 1.5)) : (nir * day_illum);
            col = vec3(val);
        } else if (u_spectral_id == 3) {
            float illum = (u_is_night == 1) ? (0.16 + 0.14 * ndotl) : day_illum;
            col = vec3(nir, r, g) * illum;
        } else if (u_spectral_id == 4) {
            float z_norm = (dsm_z - u_zbounds.x) / max(u_zbounds.y - u_zbounds.x, 10.0);
            float water = clamp((-ndvi - 0.15) * 2.5, 0.0, 1.0) * clamp((0.25 - albedo) * 4.0, 0.0, 1.0);
            float veg = clamp((ndvi - 0.15) * 2.2, 0.0, 1.0);
            float temp;
            if (u_is_night == 1) {
                temp = 0.34 + 0.28 * water + 0.30 * urban + 0.10 * veg - 0.14 * z_norm - 0.12 * (1.0 - albedo) * (1.0 - urban);
            } else {
                float solar_heat = 0.42 * (1.0 - 0.55 * albedo) * direct;
                temp = 0.28 + solar_heat + 0.16 * urban - 0.24 * veg - 0.28 * water - 0.12 * z_norm;
            }
            col = vec3(clamp(temp, 0.04, 0.98));
        } else {
            float nvg = clamp(0.55 * nir + 0.35 * albedo + 0.45 * pow(urban, 1.4), 0.0, 1.0);
            nvg = clamp(nvg * (0.55 + 0.45 * ndotl), 0.0, 1.0);
            col = vec3(0.14 * nvg, 0.96 * nvg, 0.26 * nvg);
        }

        // Koschmieder 3D atmospheric haze (#12)
        if (u_haze_enabled == 1) {
            float trans = exp(-u_haze_beta * clamp(depth_m, 0.0, 20000.0));
            col = mix(haze_airlight(), col, trans);
        }

        // Blinn-Phong specular glint (#15)
        if (u_glare_enabled == 1 && u_specular_glint > 0.01 && u_spectral_id < 4) {
            vec3 hvec = normalize(-ray_w + u_sun_vec);
            float ndoth = clamp(dot(norm_w, hvec), 0.0, 1.0);
            float spec = clamp(pow(ndoth, 42.0) * u_specular_glint, 0.0, 0.85);
            col = clamp(col + spec * vec3(1.0, 0.96, 0.86), 0.0, 1.0);
        }
    }

    // Optical sun glare / lens flare (#15)
    if (u_glare_enabled == 1 && u_has_sun_uv == 1 && u_glare_intensity > 0.01 && u_spectral_id < 4) {
        vec2 screen_uv = vec2(v_uv.x, 1.0 - v_uv.y);
        float aspect = u_res.x / max(u_res.y, 1.0);
        vec2 d = vec2((screen_uv.x - u_sun_uv.x) * aspect, screen_uv.y - u_sun_uv.y);
        float dist = length(d);
        float halo = exp(-dist * 3.8) * u_glare_intensity * 0.65;
        float core = exp(-dist * dist * 45.0) * u_glare_intensity * 0.55;
        vec2 guv = vec2(1.0 - u_sun_uv.x, 1.0 - u_sun_uv.y);
        vec2 gd = vec2((screen_uv.x - guv.x) * aspect, screen_uv.y - guv.y);
        float gdist = length(gd);
        float ghost = exp(-gdist * gdist * 32.0) * u_glare_intensity * 0.22;
        col = clamp(col + (halo + core) * vec3(1.0, 0.92, 0.78) + ghost * vec3(0.75, 0.88, 1.0), 0.0, 1.0);
    }

    // Kelvin color temperature & white balance (#16) for color optical modes
    if (u_spectral_id == 0 || u_spectral_id == 3) {
        col = clamp(col * u_wb_gains, 0.0, 1.0);
    }

    out_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

_SPECTRAL_ID = {
    "rgb": 0,
    "panchromatic": 1,
    "nir": 2,
    "false_color_cir": 3,
    "thermal_lwir": 4,
    "night_vision": 5,
}


class FastSceneGPURenderer:
    """Persistent GPU scene context that renders multiple videos on one corridor
    without re-uploading VRAM textures, reading back only an RGB8 buffer (~12–25 ms/frame).
    """

    engine_name = "gpu_opengl_raymarch_fast"

    def __init__(self, scene: SceneBundle, coarse_steps: int = 44, refine_steps: int = 7) -> None:
        self.scene = scene
        self.coarse_steps = coarse_steps
        self.refine_steps = refine_steps
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.prog = self.ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_ALL_IN_ONE_FRAG)

        h, w = scene.height_px, scene.width_px
        img_u8 = np.ascontiguousarray(np.clip(scene.imagery * 255.0, 0, 255).astype(np.uint8))
        self.t_img = self.ctx.texture((w, h), 4, img_u8.tobytes(), dtype="f1")
        self.t_img.build_mipmaps()
        self.t_img.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.t_img.anisotropy = 16.0
        self.t_img.repeat_x = False
        self.t_img.repeat_y = False

        self.t_dsm = self.ctx.texture((w, h), 1, np.ascontiguousarray(scene.dsm, dtype=np.float32).tobytes(), dtype="f4")
        self.t_dsm.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_dsm.repeat_x = False
        self.t_dsm.repeat_y = False

        self.t_norm = self.ctx.texture((w, h), 3, np.ascontiguousarray(scene.normals, dtype=np.float32).tobytes(), dtype="f4")
        self.t_norm.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_norm.repeat_x = False
        self.t_norm.repeat_y = False

        self.t_ndvi = self.ctx.texture((w, h), 1, np.ascontiguousarray(scene.ndvi, dtype=np.float32).tobytes(), dtype="f4")
        self.t_ndvi.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_ndvi.repeat_x = False
        self.t_ndvi.repeat_y = False

        # Shadow texture updated per video's sun angle
        self.t_shd = self.ctx.texture((w, h), 1, dtype="f4")
        self.t_shd.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_shd.repeat_x = False
        self.t_shd.repeat_y = False

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_pos")

        self.prog["u_imagery"].value = 0
        self.prog["u_dsm"].value = 1
        self.prog["u_normals"].value = 2
        self.prog["u_ndvi"].value = 3
        self.prog["u_shadow"].value = 4
        self.prog["u_bounds"].value = (scene.x_min_m, scene.y_min_m, scene.x_max_m, scene.y_max_m)
        self.prog["u_zbounds"].value = (scene.z_min_m, scene.z_max_m)
        self.prog["u_has_dsm"].value = 1 if scene.has_real_dsm else 0
        self.prog["u_coarse_steps"].value = coarse_steps
        self.prog["u_refine_steps"].value = refine_steps

        # Cache FBOs per output resolution (360p, 480p, 720p, 1080p)
        self._fbos: dict[tuple[int, int], tuple[moderngl.Framebuffer, moderngl.Texture]] = {}
        self.focal_px: float = 500.0
        self._cur_res: tuple[int, int] = (1280, 720)

    def _get_fbo(self, w: int, h: int) -> tuple[moderngl.Framebuffer, moderngl.Texture]:
        key = (w, h)
        if key not in self._fbos:
            tex = self.ctx.texture((w, h), 3, dtype="f1")
            fbo = self.ctx.framebuffer(color_attachments=[tex])
            self._fbos[key] = (fbo, tex)
        return self._fbos[key]

    def configure_video(self, params: VideoSampleParams) -> None:
        """Update uniforms & shadow texture for a new video on the already-loaded scene."""
        self._cur_res = (params.width, params.height)
        self.focal_px = 0.5 * params.width / math.tan(math.radians(params.fov_deg) * 0.5)

        shadow_map = self.scene.compute_shadow_map(
            sun_azimuth_deg=params.sun_azimuth_deg,
            sun_elevation_deg=params.sun_elevation_deg,
            max_steps=48,
        )
        self.t_shd.write(np.ascontiguousarray(shadow_map, dtype=np.float32).tobytes())

        az = math.radians(params.sun_azimuth_deg)
        el = math.radians(params.sun_elevation_deg)
        sun_vec = np.array(
            [math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)],
            dtype=np.float32,
        )
        rot_c2w = DSMCameraRenderer._camera_to_world_matrix(params.yaw_deg, params.pitch_deg)

        sun_cam = rot_c2w.T @ sun_vec
        if sun_cam[2] > 0.05:
            su = 0.5 + (sun_cam[0] / sun_cam[2]) * (self.focal_px / params.width)
            sv = 0.5 + (sun_cam[1] / sun_cam[2]) * (self.focal_px / params.height)
            has_sun_uv = (-0.45 <= su <= 1.45 and -0.45 <= sv <= 1.45)
        else:
            su, sv, has_sun_uv = 0.0, 0.0, False

        wb = _kelvin_to_rgb_gains(params.color_temp_k)
        wb[1] *= float(1.0 + params.wb_tint)
        wb /= float(np.mean(wb))

        vis_m = max(300.0, params.visibility_km * 1000.0)
        beta = 3.912 / vis_m
        if params.spectral_mode == "thermal_lwir":
            beta *= 0.35
        elif params.spectral_mode == "nir":
            beta *= 0.65

        self.prog["u_res"].value = (float(params.width), float(params.height))
        self.prog["u_fx"].value = float(self.focal_px)
        self.prog["u_dist"].value = (params.k1, params.k2, params.p1, params.p2)
        self.prog["u_rot_c2w"].write(np.ascontiguousarray(rot_c2w.T, dtype=np.float32).tobytes())
        self.prog["u_spectral_id"].value = _SPECTRAL_ID.get(params.spectral_mode, 0)
        self.prog["u_is_night"].value = 1 if params.is_night else 0
        self.prog["u_sun_vec"].value = (float(sun_vec[0]), float(sun_vec[1]), float(sun_vec[2]))
        self.prog["u_shadow_strength"].value = float(params.shadow_strength)
        self.prog["u_wb_gains"].value = (float(wb[0]), float(wb[1]), float(wb[2]))
        self.prog["u_haze_enabled"].value = 1 if params.haze_enabled else 0
        self.prog["u_haze_beta"].value = float(beta)
        self.prog["u_glare_enabled"].value = 1 if (params.sun_glare_enabled and not params.is_night) else 0
        self.prog["u_glare_intensity"].value = float(params.glare_intensity)
        self.prog["u_specular_glint"].value = float(params.specular_glint)
        self.prog["u_has_sun_uv"].value = 1 if has_sun_uv else 0
        self.prog["u_sun_uv"].value = (float(su), float(sv))

    def render_frame_rgb_float(self, pose: FramePose) -> np.ndarray:
        """Render frame on GPU and return float32 RGB image (H, W, 3) in [0, 1]."""
        w, h = self._cur_res
        fbo, tex = self._get_fbo(w, h)

        self.t_img.use(0)
        self.t_dsm.use(1)
        self.t_norm.use(2)
        self.t_ndvi.use(3)
        self.t_shd.use(4)

        self.prog["u_cam_pos"].value = (float(pose.x_m), float(pose.y_m), float(pose.z_msl_m))
        fbo.use()
        self.vao.render(moderngl.TRIANGLE_STRIP)

        raw = np.frombuffer(tex.read(alignment=1), dtype=np.uint8).reshape(h, w, 3)[::-1]
        return raw.astype(np.float32) * (1.0 / 255.0)

    def close(self) -> None:
        for fbo, tex in self._fbos.values():
            fbo.release()
            tex.release()
        self.t_img.release()
        self.t_dsm.release()
        self.t_norm.release()
        self.t_ndvi.release()
        self.t_shd.release()
        self.vao.release()
        self.vbo.release()
        self.prog.release()
        self.ctx.release()

    def __enter__(self) -> "FastSceneGPURenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
