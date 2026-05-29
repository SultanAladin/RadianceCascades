#version 450
// Fullscreen triangle. gl_VertexIndex 0..2 → screen-space corners that fully
// cover the [-1, +1] NDC square with one extra triangle of overdraw (Bevel
// trick). v_uv lands in [0, 1] for the on-screen pixels.
layout(location = 0) out vec2 v_uv;

void main() {
    vec2 pos;
    if (gl_VertexIndex == 0)      pos = vec2(-1.0, -1.0);
    else if (gl_VertexIndex == 1) pos = vec2( 3.0, -1.0);
    else                          pos = vec2(-1.0,  3.0);

    gl_Position = vec4(pos, 0.0, 1.0);
    v_uv        = pos * 0.5 + 0.5;
}
