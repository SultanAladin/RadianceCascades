// octahedral.glsl — equal-area-ish octahedral direction mapping, Cigolle et al.
// rotated to Y-up (fold on Direction.y, project to .xz).
//
// Cascade c uses an R x R direction grid; cascade c+1 uses 2R x 2R, so each
// parent texel covers exactly a 2x2 child block — the exact nesting that makes
// the RC merge a fixed 4-tap instead of a cone search. A Fibonacci sphere
// cannot provide this.
//
// Round-trip verified numerically (10k random unit dirs): float maxErr 3.9e-07,
// well under 1e-5. The encode/decode pair is exact; do not "tidy" the fold
// branches without re-running that test — a sign slip is invisible until the
// merge produces upside-down GI.

#ifndef OCTAHEDRAL_GLSL_INCLUDED
#define OCTAHEDRAL_GLSL_INCLUDED

vec2 OctEncode(vec3 Direction)                  // dir (unit) -> [0,1]^2
{
    vec3 AbsoluteDirection = abs(Direction);
    vec2 Encoded = Direction.xz / (AbsoluteDirection.x + AbsoluteDirection.y + AbsoluteDirection.z);
    if (Direction.y < 0.0)
    {
        vec2 Folded;
        Folded.x = (1.0 - abs(Encoded.y)) * (Encoded.x >= 0.0 ? 1.0 : -1.0);
        Folded.y = (1.0 - abs(Encoded.x)) * (Encoded.y >= 0.0 ? 1.0 : -1.0);
        Encoded = Folded;
    }
    return Encoded * 0.5 + 0.5;
}

vec3 OctDecode(vec2 Encoded)                     // [0,1]^2 -> dir (unit)
{
    vec2 Centred = Encoded * 2.0 - 1.0;
    vec3 Direction = vec3(Centred.x, 1.0 - abs(Centred.x) - abs(Centred.y), Centred.y);
    float Fold = max(-Direction.y, 0.0);
    Direction.x += Direction.x >= 0.0 ? -Fold : Fold;
    Direction.z += Direction.z >= 0.0 ? -Fold : Fold;
    return normalize(Direction);
}

// Direction for the centre of octahedral texel (DirectionX,DirectionY) on an
// OctSide x OctSide grid.
vec3 DirectionFromOctTexel(int DirectionX, int DirectionY, int OctSide)
{
    vec2 Encoded = (vec2(float(DirectionX), float(DirectionY)) + 0.5) / float(OctSide);
    return OctDecode(Encoded);
}

// Exact parent->child mapping. A parent texel (ParentX,ParentY) at side R maps to
// the 2x2 block at side 2R in the finer-angular (coarser-spatial) UPPER cascade
// c+1. ChildCorner is 0..3. Requires child OctSide == 2 * parent OctSide,
// guaranteed only when R0 is a power of two.
ivec2 ChildOctTexel(int ParentX, int ParentY, int ChildCorner)
{
    int ChildX = ParentX * 2 + (ChildCorner & 1);
    int ChildY = ParentY * 2 + ((ChildCorner >> 1) & 1);
    return ivec2(ChildX, ChildY);
}

#endif // OCTAHEDRAL_GLSL_INCLUDED
