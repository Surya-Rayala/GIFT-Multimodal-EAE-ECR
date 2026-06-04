// Shared types for the scaled point viewer.

import type { Vec2 } from '@/utils/scaling';

export type { Vec2 } from '@/utils/scaling';

export interface SaveRequest {
  output_dir: string;
  project_name: string;
  map_image_path: string;
  points: Vec2[];
  project: Record<string, unknown>;
}

export interface SaveResponse {
  ok: boolean;
  points_path: string;
  image_path: string;
  project_path: string;
  count: number;
}

// Wizard steps, in order. Numeric ids so they sort.
export const STEPS = [
  { id: 1, key: 'setup', title: 'Setup', subtitle: 'Pick the room map image and where to save.' },
  { id: 2, key: 'walls', title: 'Draw Walls', subtitle: 'Trace walls and type their real-world lengths.' },
  { id: 3, key: 'points', title: 'Place Points', subtitle: 'Add points from their distance to two walls.' },
] as const;

export type StepKey = (typeof STEPS)[number]['key'];
