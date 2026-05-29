#version 450
// Fullscreen triangle covering NDC. Same vertex trick as GridPass — no
// vertex buffer, gl_VertexIndex drives clip-space.
layout(location = 0) out vec2 vUV;

void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vUV         = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
