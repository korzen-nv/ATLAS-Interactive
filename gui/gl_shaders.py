"""GLSL shader sources for the OpenGL canvas widget.

All visualization modes are handled in a single fragment shader controlled
by the u_mode uniform.  This avoids shader switching and keeps the GL state
machine simple.
"""

# ── Vis-mode constants (must match the u_mode uniform values) ─────────────
MODE_IMAGE = 0
MODE_MASK = 1
MODE_DAVIS = 2
MODE_FADE = 3
MODE_LIGHT = 4
MODE_POPUP = 5
MODE_LAYER = 6
MODE_RGBA = 7
MODE_PRECOMPOSITED = 8

MODE_FROM_NAME = {
    'image': MODE_IMAGE,
    'mask': MODE_MASK,
    'davis': MODE_DAVIS,
    'fade': MODE_FADE,
    'light': MODE_LIGHT,
    'popup': MODE_POPUP,
    'layer': MODE_LAYER,
    'rgba': MODE_RGBA,
    '_precomposited': MODE_PRECOMPOSITED,
}

# ── Vertex shader ─────────────────────────────────────────────────────────
VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texcoord;

out vec2 v_uv;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_uv = a_texcoord;
}
"""

# ── Fragment shader ───────────────────────────────────────────────────────
# Inputs:
#   tex_image     – GL_TEXTURE_2D,       RGBA float [0,1]   (unit 0)
#   tex_mask      – GL_TEXTURE_2D,       R    uint8→float   (unit 1)
#   tex_color_map – GL_TEXTURE_1D,       RGB  float [0,1]   (unit 2)
#   tex_overlay   – GL_TEXTURE_2D,       RGBA float [0,1]   (unit 3)
#   tex_prob      – GL_TEXTURE_2D_ARRAY, R    float [0,1]   (unit 4)
#
# The shader supports two mask sources:
#   - Hard mask via tex_mask (uint8 class index, used for CPU/scrubbing path)
#   - Soft prob via tex_prob  (per-class probability, used for GPU path)
# u_use_soft_prob selects which one.

FRAGMENT_SHADER = """
#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D   tex_image;       // unit 0
uniform sampler2D   tex_mask;        // unit 1 – hard mask (R channel = class/255)
uniform sampler1D   tex_color_map;   // unit 2 – 256 RGB entries
uniform sampler2D   tex_overlay;     // unit 3 – RGBA overlay layer
uniform sampler2DArray tex_prob;     // unit 4 – soft probabilities

uniform int   u_mode;               // vis mode (see MODE_* constants)
uniform float u_alpha;              // blend alpha for davis/light modes
uniform int   u_num_classes;        // K+1 (including background)
uniform bool  u_use_soft_prob;      // true = use tex_prob, false = use tex_mask
uniform int   u_target_objects[256]; // 1 if object is a target, 0 otherwise
uniform int   u_num_targets;        // number of target objects set

// Grayscale weights matching existing code
const vec3 GRAY_W = vec3(0.3, 0.59, 0.11);

// ── Helpers ──────────────────────────────────────────────────────────────

// Get hard mask class index from the mask texture (0-255)
int get_hard_class() {
    float r = texture(tex_mask, v_uv).r;
    return int(r * 255.0 + 0.5);
}

// Get color for a class index from the 1D color map
vec3 class_color(int cls) {
    // texelFetch avoids filtering artifacts on the 1D palette
    return texelFetch(tex_color_map, cls, 0).rgb;
}

// Soft-prob path: find argmax class and its probability
void soft_argmax(out int best_cls, out float best_p) {
    best_cls = 0;
    best_p = 0.0;
    for (int k = 0; k < u_num_classes; k++) {
        float p = texture(tex_prob, vec3(v_uv, float(k))).r;
        if (p > best_p) {
            best_p = p;
            best_cls = k;
        }
    }
}

// Soft foreground mask: sum of probs for target objects
float soft_foreground() {
    float fg = 0.0;
    for (int i = 0; i < u_num_targets; i++) {
        int k = u_target_objects[i];
        fg += texture(tex_prob, vec3(v_uv, float(k))).r;
    }
    return clamp(fg, 0.0, 1.0);
}

// Hard foreground mask: 1 if class is in target_objects, 0 otherwise
float hard_foreground() {
    int cls = get_hard_class();
    for (int i = 0; i < u_num_targets; i++) {
        if (u_target_objects[i] == cls) return 1.0;
    }
    return 0.0;
}

// ── Mode implementations ─────────────────────────────────────────────────

void main() {
    vec3 img = texture(tex_image, v_uv).rgb;

    // MODE_IMAGE / MODE_PRECOMPOSITED: raw image passthrough
    if (u_mode == 0 || u_mode == 8) {
        frag_color = vec4(img, 1.0);
        return;
    }

    // Determine mask class and foreground weight
    int cls;
    float fg;

    if (u_use_soft_prob) {
        float best_p;
        soft_argmax(cls, best_p);
        fg = soft_foreground();
    } else {
        cls = get_hard_class();
        fg = hard_foreground();
    }

    vec3 mask_color = class_color(cls);

    // MODE_MASK: colormap only
    if (u_mode == 1) {
        frag_color = vec4(mask_color, 1.0);
        return;
    }

    // MODE_DAVIS (2), MODE_FADE (3), MODE_LIGHT (4)
    if (u_mode == 2 || u_mode == 3 || u_mode == 4) {
        float a = u_alpha;
        vec3 blended = img * a + mask_color * (1.0 - a);
        // Mix blended into foreground regions (class > 0)
        float is_fg = (cls > 0) ? 1.0 : 0.0;
        vec3 result = mix(img, blended, is_fg);
        // Fade background for MODE_FADE
        if (u_mode == 3) {
            result = mix(result * 0.6, result, is_fg);
        }
        frag_color = vec4(result, 1.0);
        return;
    }

    // MODE_POPUP: foreground colored, background grayscale
    if (u_mode == 5) {
        float gray = dot(img, GRAY_W);
        vec3 result = mix(vec3(gray), img, fg);
        frag_color = vec4(result, 1.0);
        return;
    }

    // MODE_LAYER: insert overlay layer between foreground and background
    if (u_mode == 6) {
        vec4 layer = texture(tex_overlay, v_uv);
        vec3 layer_rgb = layer.rgb;
        float layer_a = layer.a;
        float bg_alpha = (1.0 - fg) * (1.0 - layer_a);
        vec3 result = img * bg_alpha + layer_rgb * (1.0 - fg) * layer_a + img * fg;
        frag_color = vec4(clamp(result, 0.0, 1.0), 1.0);
        return;
    }

    // MODE_RGBA: image with alpha = foreground mask
    if (u_mode == 7) {
        frag_color = vec4(img, fg);
        return;
    }

    // fallback
    frag_color = vec4(img, 1.0);
}
"""
