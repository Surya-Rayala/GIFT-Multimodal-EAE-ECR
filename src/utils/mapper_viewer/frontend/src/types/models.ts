// Shared types for the mapper viewer.

export type Point = [number, number];

export interface VideoProbe {
  path: string;
  frame_count: number;
  fps: number;
  width: number;
  height: number;
}

export interface MappingPair {
  fx: number;
  fy: number;
  mx: number | null;
  my: number | null;
}

export type ArtifactKind = 'mapping' | 'entry_polygons' | 'pod_points' | 'room_boundary';

export interface SaveArtifactRequest {
  output_dir: string;
  project_name: string;
  kind: ArtifactKind;
  mapping?: { fx: number; fy: number; mx: number; my: number }[];
  polygons?: Point[][];
  points?: Point[];
}

export interface SaveArtifactResponse {
  ok: boolean;
  path: string;
  lines: number;
}

export type Tool = 'click' | 'drag';

// Steps in the wizard, in order. Numeric ids so they sort.
export const STEPS = [
  { id: 1, key: 'setup',     title: 'Setup',                  subtitle: 'Pick your map image, camera reference, and where to save.' },
  { id: 2, key: 'align',     title: 'Align Camera to Map',    subtitle: 'Match points so we can map tracker positions onto the room.' },
  { id: 3, key: 'entries',   title: 'Mark Entry Zones',       subtitle: 'Outline the door region(s) on the map.' },
  { id: 4, key: 'pods',      title: 'Mark POD Targets',       subtitle: 'Drop a pin on each Point-Of-Domination.' },
  { id: 5, key: 'boundary',  title: 'Outline the Room',       subtitle: 'Trace the walls / floor boundary.' },
] as const;

export type StepKey = (typeof STEPS)[number]['key'];
