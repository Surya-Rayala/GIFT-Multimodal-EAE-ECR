// Core scaling math for the Scaled Point Viewer.
//
// Problem: the user knows, for each real-world reference point, its
// perpendicular (shortest) distance to its two nearest adjacent walls, plus the
// real-world length of those walls. We must drop the point onto the map image
// at the correct pixel.
//
// Key idea — ONE global scale, real distances, no per-wall ratios:
//   * Every wall the user draws gives a scale estimate `pixelLength / realLength`
//     (pixels per real unit). We aggregate them into a single, pixel-length
//     weighted scale `S`, so long reliable walls dominate and a short corridor
//     stub barely counts. This is what makes placement robust to corridors and
//     short walls: the chosen wall supplies only a *direction*, never the scale.
//   * A point is then found by SOLVING for the pixel that sits at the given
//     perpendicular distance from each of the two wall lines (a 2x2 linear
//     system using the walls' inward normals). This is exact for ANY corner
//     angle — 90 degrees is just the special case; 60 or 110 degree corners work
//     too.

export type Vec2 = [number, number];

export interface Wall {
  id: number;
  p1: Vec2; // image pixels
  p2: Vec2; // image pixels
  realLength: number; // real-world units (> 0)
}

export interface GlobalScale {
  /** Pixels per real unit. 0 when no usable walls exist. */
  scale: number;
  /** Per-wall diagnostics for the consistency badge. */
  perWall: Array<{ id: number; scale: number; ratio: number }>;
}

export interface PlaceResult {
  ok: boolean;
  point?: Vec2;
  /** Present when ok === false. */
  error?: string;
}

// --- tiny vector helpers ---------------------------------------------------
export function sub(a: Vec2, b: Vec2): Vec2 {
  return [a[0] - b[0], a[1] - b[1]];
}
export function add(a: Vec2, b: Vec2): Vec2 {
  return [a[0] + b[0], a[1] + b[1]];
}
export function scaleVec(a: Vec2, s: number): Vec2 {
  return [a[0] * s, a[1] * s];
}
export function dot(a: Vec2, b: Vec2): number {
  return a[0] * b[0] + a[1] * b[1];
}
export function len(a: Vec2): number {
  return Math.hypot(a[0], a[1]);
}
/** 90-degree rotation: (x, y) -> (-y, x). */
export function perp(a: Vec2): Vec2 {
  return [-a[1], a[0]];
}
export function normalize(a: Vec2): Vec2 {
  const l = len(a);
  return l === 0 ? [0, 0] : [a[0] / l, a[1] / l];
}

export function wallPixelLength(w: Wall): number {
  return len(sub(w.p2, w.p1));
}

// --- global scale ----------------------------------------------------------
/**
 * Aggregate all walls into one pixels-per-real-unit scale. Weighted by each
 * wall's pixel length so longer (more precise) walls dominate; `ratio` is each
 * wall's own scale divided by the consensus, for an outlier badge in the UI.
 */
export function computeGlobalScale(walls: Wall[]): GlobalScale {
  let wSum = 0;
  let wsSum = 0;
  const perWall: Array<{ id: number; scale: number; px: number }> = [];
  for (const w of walls) {
    const px = wallPixelLength(w);
    if (px <= 0 || w.realLength <= 0) continue;
    const s = px / w.realLength;
    perWall.push({ id: w.id, scale: s, px });
    wSum += px;
    wsSum += s * px;
  }
  const scale = wSum > 0 ? wsSum / wSum : 0;
  return {
    scale,
    perWall: perWall.map((p) => ({
      id: p.id,
      scale: p.scale,
      ratio: scale > 0 ? p.scale / scale : 1,
    })),
  };
}

// --- corner / direction helpers -------------------------------------------
/**
 * The endpoint shared by both walls (within `eps` pixels), or null when the
 * walls do not meet at a corner.
 */
export function sharedCorner(a: Wall, b: Wall, eps = 6): Vec2 | null {
  const ends: Array<[Vec2, Vec2]> = [
    [a.p1, b.p1],
    [a.p1, b.p2],
    [a.p2, b.p1],
    [a.p2, b.p2],
  ];
  let best: { c: Vec2; d: number } | null = null;
  for (const [ea, eb] of ends) {
    const d = len(sub(ea, eb));
    if (d <= eps && (best === null || d < best.d)) {
      // Use the midpoint so tiny snap mismatches average out.
      best = { c: [(ea[0] + eb[0]) / 2, (ea[1] + eb[1]) / 2], d };
    }
  }
  return best ? best.c : null;
}

/** Unit direction along the wall, pointing away from `corner`. */
export function unitFromCorner(w: Wall, corner: Vec2): Vec2 | null {
  const d1 = len(sub(w.p1, corner));
  const d2 = len(sub(w.p2, corner));
  const far = d1 >= d2 ? w.p1 : w.p2;
  const v = normalize(sub(far, corner));
  return len(v) === 0 ? null : v;
}

// --- the placement solve ---------------------------------------------------
/**
 * Solve for the pixel that is `distFromA` (perpendicular) from wall A's line and
 * `distFromB` from wall B's line, both measured in real units and converted to
 * pixels with the global `scale`. Works at any corner angle.
 *
 * @param corner    shared corner of the two walls (image px)
 * @param uA        unit direction along wall A, from the corner
 * @param uB        unit direction along wall B, from the corner
 */
export function placePoint(
  corner: Vec2,
  uA: Vec2,
  uB: Vec2,
  distFromA: number,
  distFromB: number,
  scale: number,
): PlaceResult {
  if (scale <= 0) {
    return { ok: false, error: 'No map scale yet — add at least one wall with a real length.' };
  }

  // Inward unit normals: orient each so it points to the interior, i.e. toward
  // the side where the OTHER wall lies. This needs no extra user input and is
  // correct for convex room corners.
  let nA = perp(uA);
  if (dot(nA, uB) < 0) nA = scaleVec(nA, -1);
  let nB = perp(uB);
  if (dot(nB, uA) < 0) nB = scaleVec(nB, -1);

  // M (P - C) = rhs, with M's rows the two inward normals.
  const det = nA[0] * nB[1] - nA[1] * nB[0];
  // |det| = |sin(angle between walls)|. Near 0 => walls nearly parallel.
  if (Math.abs(det) < Math.sin((5 * Math.PI) / 180)) {
    return {
      ok: false,
      error: 'The two walls are nearly parallel — pick a clearer corner where the walls meet at an angle.',
    };
  }

  const r0 = distFromA * scale;
  const r1 = distFromB * scale;
  // M^{-1} rhs, with M = [[nA.x, nA.y], [nB.x, nB.y]].
  const qx = (nB[1] * r0 - nA[1] * r1) / det;
  const qy = (-nB[0] * r0 + nA[0] * r1) / det;

  return { ok: true, point: [corner[0] + qx, corner[1] + qy] };
}

/**
 * Convenience: given the two walls and the two perpendicular distances, find
 * the corner + directions and place the point. Returns an error result if the
 * walls are not adjacent.
 */
export function placeFromWalls(
  wallA: Wall,
  wallB: Wall,
  distFromA: number,
  distFromB: number,
  scale: number,
  cornerEps = 6,
): PlaceResult {
  const corner = sharedCorner(wallA, wallB, cornerEps);
  if (!corner) {
    return { ok: false, error: 'Walls A and B must meet at a corner (share an endpoint).' };
  }
  const uA = unitFromCorner(wallA, corner);
  const uB = unitFromCorner(wallB, corner);
  if (!uA || !uB) {
    return { ok: false, error: 'Could not determine wall directions from the corner.' };
  }
  return placePoint(corner, uA, uB, distFromA, distFromB, scale);
}
