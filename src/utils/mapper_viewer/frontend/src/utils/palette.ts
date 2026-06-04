// Distinct, well-saturated colors used to differentiate the mapping pairs,
// entry polygons, and POD points. Each item picks `colors[i % len]`.
export const trackPalette: ReadonlyArray<string> = [
  '#ef4444', // red
  '#22c55e', // green
  '#3b82f6', // blue
  '#eab308', // yellow
  '#a855f7', // purple
  '#06b6d4', // cyan
  '#f97316', // orange
  '#ec4899', // pink
  '#84cc16', // lime
  '#14b8a6', // teal
  '#f43f5e', // rose
  '#8b5cf6', // violet
];

/** Solid pin color for POD targets — same for every pin, distinct enough on a map. */
export const POD_COLOR = '#0ea5e9';

/** Polygon stroke color for the active draft (unconfirmed). */
export const DRAFT_COLOR = '#ef4444';

/** Polygon stroke color for confirmed (saved) shapes. */
export const CONFIRMED_COLOR = '#22c55e';
