#version 450
// Screen-space ray-marched infinite ground grid. Reconstructs world-space ray
// from `invViewProj`, intersects the y=0 plane, and shades a major/minor line
// pattern with distance fade + red/blue axis highlights at the world origin.
//
// Ported from SolidArc/Shaders/pigment_grid.frag with the paint-texture overlay
// removed (RS owns no 2D paint layer) and a sky/ground split background so the
// horizon line is visible at glancing pitches.
layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 o_color;

layout(set = 0, binding = 0) uniform samplerCube uSky;

layout(push_constant) uniform GridPush {
    mat4 InvViewProj;
    vec4 MajorColorAndSize;       // rgb = major colour, a = finite grid extent (unused when w=1 in next)
    vec4 MinorColorAndInfinite;   // rgb = minor colour, a = 1 → infinite grid, 0 → finite
    vec4 SpacingAndThickness;     // x = major spacing, y = minor spacing, z = major weight, w = minor weight
    vec4 FadeAndOpacity;          // x = fade distance, y = falloff start, z = falloff curve, w = signed opacity (sign toggles axis highlight)
    vec4 SkyAndGround;            // rgb = sky tint (multiplier on sky cube),   a = unused
    vec4 GroundColor;             // rgb = ground tint blended at y=0,           a = unused
    vec4 CheckerLightAndSpacing;  // rgb = light tile, a = spacing (metres)
    vec4 CheckerDarkAndStrength;  // rgb = dark tile,  a = strength [0,1]
} pc;

float GridLineMask(vec2 worldXZ, float spacing, float thickness) {
    vec2  scaled     = worldXZ / max(spacing, 1e-4);
    vec2  derivative = max(fwidth(scaled), vec2(1e-4));
    vec2  g          = abs(fract(scaled - 0.5) - 0.5) / derivative;
    float line       = 1.0 - min(min(g.x, g.y), 1.0);
    float shaped     = pow(line, max(0.2, 1.0 / max(thickness, 0.2)));
    return clamp(shaped, 0.0, 1.0);
}

// Classic dev-engine checker. Two-tile pattern on world XZ at `spacing` metres.
// We anti-alias the cell boundaries with `fwidth` so the checker degrades to
// the average colour at grazing angles / large distances instead of aliasing.
float CheckerMask(vec2 worldXZ, float spacing) {
    vec2  scaled = worldXZ / max(spacing, 1e-4);
    vec2  cell   = floor(scaled);
    return mod(cell.x + cell.y, 2.0);   // 0 or 1, constant per cell
}

void main() {
    vec2 ndc = vec2(v_uv.x * 2.0 - 1.0, v_uv.y * 2.0 - 1.0);

    vec4 nearWorld = pc.InvViewProj * vec4(ndc, 0.0, 1.0);
    vec4 farWorld  = pc.InvViewProj * vec4(ndc, 1.0, 1.0);
    nearWorld /= max(nearWorld.w, 1e-6);
    farWorld  /= max(farWorld.w,  1e-6);

    vec3 rayOrigin    = nearWorld.xyz;
    vec3 rayDirection = normalize(farWorld.xyz - nearWorld.xyz);

    // Phase 8: background is the sky cubemap (tinted by SkyAndGround.rgb so
    // the panel can adjust without re-baking the cube), with a thin ground
    // smear below the horizon for visual stability.
    vec3 skySample = texture(uSky, rayDirection).rgb * pc.SkyAndGround.rgb;
    float horizonT = smoothstep(-0.08, 0.08, rayDirection.y);
    vec3  background = mix(pc.GroundColor.rgb, skySample, horizonT);

    if (abs(rayDirection.y) < 1e-5) {
        o_color = vec4(background, 1.0);
        return;
    }
    float t = -rayOrigin.y / rayDirection.y;
    if (t < 0.0) {
        o_color = vec4(background, 1.0);
        return;
    }

    vec3 worldHit = rayOrigin + rayDirection * t;
    vec2 worldXZ  = worldHit.xz;

    float minorMask = GridLineMask(worldXZ, pc.SpacingAndThickness.y, pc.SpacingAndThickness.w);
    float majorMask = GridLineMask(worldXZ, pc.SpacingAndThickness.x, pc.SpacingAndThickness.z);

    // Optional finite-extent fade (off when MinorColorAndInfinite.a >= 0.5).
    float finiteFade = 1.0;
    if (pc.MinorColorAndInfinite.a < 0.5) {
        float maxExtent = max(pc.MajorColorAndSize.a, 0.001);
        float extent    = max(abs(worldXZ.x), abs(worldXZ.y));
        finiteFade = 1.0 - smoothstep(maxExtent * 0.96, maxExtent, extent);
    }

    // Distance fade from camera (when enabled by FadeAndOpacity.x > 0).
    float distanceFade = 1.0;
    if (pc.FadeAndOpacity.x > 0.0) {
        float fadeDistance = max(pc.FadeAndOpacity.x, 0.001);
        float falloffStart = clamp(pc.FadeAndOpacity.y, 0.0, fadeDistance - 0.001);
        float viewDistance = length(worldHit - rayOrigin);
        float fadeT        = clamp((viewDistance - falloffStart) /
                                   max(fadeDistance - falloffStart, 0.001), 0.0, 1.0);
        distanceFade = pow(1.0 - fadeT, max(pc.FadeAndOpacity.z, 0.05));
    }

    // Dev-engine checker floor under the grid lines. Strength = 0 falls back
    // to the flat GroundColor (legacy look); strength > 0 paints the classic
    // two-tone tile pattern at CheckerSpacing.
    float checkerParity = CheckerMask(worldXZ, pc.CheckerLightAndSpacing.a);
    vec3  checkerTile   = mix(pc.CheckerDarkAndStrength.rgb,
                              pc.CheckerLightAndSpacing.rgb,
                              checkerParity);
    vec3  floorColor    = mix(pc.GroundColor.rgb, checkerTile,
                              clamp(pc.CheckerDarkAndStrength.a, 0.0, 1.0));

    vec3 composite = floorColor;
    composite = mix(composite, pc.MinorColorAndInfinite.rgb, clamp(minorMask * 0.75, 0.0, 1.0));
    composite = mix(composite, pc.MajorColorAndSize.rgb,     clamp(majorMask,        0.0, 1.0));

    // Red/blue origin-axis highlight (sign of FadeAndOpacity.w gates it).
    if (pc.FadeAndOpacity.w >= 0.0) {
        float axisX = 1.0 - clamp(abs(worldHit.x) / max(fwidth(worldHit.x) * 2.0, 1e-4), 0.0, 1.0);
        float axisZ = 1.0 - clamp(abs(worldHit.z) / max(fwidth(worldHit.z) * 2.0, 1e-4), 0.0, 1.0);
        composite = mix(composite, vec3(0.88, 0.34, 0.34), axisX * 0.85);
        composite = mix(composite, vec3(0.34, 0.62, 0.90), axisZ * 0.85);
    }

    float opacity    = clamp(abs(pc.FadeAndOpacity.w), 0.0, 1.0);
    float visibility = clamp(finiteFade * distanceFade * opacity, 0.0, 1.0);
    vec3  finalColor = mix(background, composite, visibility);

    o_color = vec4(finalColor, 1.0);
}
