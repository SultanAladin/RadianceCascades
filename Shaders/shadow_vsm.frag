#version 450
// Phase 11 VSM moment-write fragment shader. Pairs with shadow_csm.vert; the
// vertex shader already emits gl_Position into light-clip space, so we read the
// depth from gl_FragCoord.z and emit (depth, depth²). RG16F is enough range
// for the [0,1] depth values produced by the orthographic CSM projection.
//
// Bias against the squared term (Lauritzen 2007) — adding 0.25 * |dDepth|² to
// E[x²] absorbs the moment-quantisation error introduced by linear filtering,
// which substantially cuts light bleed at silhouette edges. Cheap, no slider.

layout(location = 0) out vec2 outMoments;

void main() {
    float depth = gl_FragCoord.z;
    float dx = dFdx(depth);
    float dy = dFdy(depth);
    float momentSq = depth * depth + 0.25 * (dx * dx + dy * dy);
    outMoments = vec2(depth, momentSq);
}
