// TypeScript mirror of the v2 Analysis JSON schema.
// Source of truth: src/analysis.py (build_analysis_session + _convert_to_v2_schema)
// + src/utils/analysis_viewer/backend/loader.py (which injects transcription,
// drill_window, session_json_path, resolved_video_path at /session load).

export type SessionInfo = {
  video_basename: string;
  analysis_mode: string;
  start_time: number | null;
};

export type VideoInfo = {
  video_path: string;
  frame_rate: number;
  total_frames: number;
  duration_sec: number;
};

export type EntryItemData = {
  entry_number: number;
  track_id: number;
  start_xy: [number, number];
  sample_count: number;
};

export type VectorItemData = {
  entry_number: number;
  track_id: number;
  start_xy: [number, number];
  end_xy: [number, number];
  dx: number;
  dy: number;
  z_cross: number;
  sign: -1 | 0 | 1;
  direction_label: 'NEG' | 'POS' | 'UNKNOWN';
  valid: boolean;
  sample_count: number;
  window_sec: number;
};

export type PairGapItemData = {
  pair_number: number;
  from_entry_number: number;
  to_entry_number: number;
  from_track_id: number;
  to_track_id: number;
  gap_frames: number;
  gap_sec: number;
  allowed_gap_sec: number;
  violates_time_limit: boolean;
};

export type DurationItemData = {
  entry_count: number;
  pair_count: number;
  duration_frames: number;
  duration_sec: number;
  derived_allowed_duration_sec: number;
  violates_total_entry_limit: boolean;
};

export type EntryTimelineItem = {
  item_id: string;
  metric_id: null;
  kind: 'entry';
  label: string;
  flag_ids: string[];
  frame: number;
  time_sec: number;
  data: EntryItemData;
};

export type VectorTimelineItem = {
  item_id: string;
  metric_id: 'entrance_vectors';
  kind: 'vector';
  label: string;
  flag_ids: string[];
  frame: number;
  time_sec: number;
  data: VectorItemData;
};

export type PairGapTimelineItem = {
  item_id: string;
  metric_id: 'entrance_hesitation';
  kind: 'pair_gap';
  label: string;
  flag_ids: string[];
  start_frame: number;
  end_frame: number;
  start_time_sec: number;
  end_time_sec: number;
  data: PairGapItemData;
};

export type DurationTimelineItem = {
  item_id: string;
  metric_id: 'total_time_of_entry';
  kind: 'duration';
  label: string;
  flag_ids: string[];
  start_frame: number;
  end_frame: number;
  start_time_sec: number;
  end_time_sec: number;
  data: DurationItemData;
};

export type WallExcursionItemData = {
  track_id: number;
  entry_number: number | null;
  label_kind: 'too_close' | 'too_far';
  duration_sec: number;
  L_map?: number;
};

export type WallExcursionTimelineItem = {
  item_id: string;
  metric_id: 'move_along_wall';
  kind: 'wall_excursion';
  label: string;
  flag_ids: string[];
  frame: number;
  time_sec: number;
  start_frame: number;
  end_frame: number;
  start_time_sec: number | null;
  end_time_sec: number | null;
  data: WallExcursionItemData;
};

export type TimelineItem =
  | EntryTimelineItem
  | VectorTimelineItem
  | PairGapTimelineItem
  | DurationTimelineItem
  | WallExcursionTimelineItem;

export type EntranceVectorsSummary = {
  vector_count: number;
  valid_vector_count: number;
  alternation_count: number | null;
  transition_count: number | null;
  window_sec: number | null;
  centroid_xy: [number, number] | null;
};

export type EntranceHesitationSummary = {
  pair_count: number;
  violation_count: number;
};

export type TotalEntryTimeSummary = {
  duration_sec: number;
  limit_sec: number | null;
  violates_total_entry_limit: boolean;
};

export type MoveAlongWallEntrant = {
  track_id: number;
  L_map?: number;
  L_source?: string;
  score?: number;
  observed_frames?: number;
  too_close_frames?: number;
  too_far_frames?: number;
  too_close_time_sec?: number;
  too_far_time_sec?: number;
  entry_frame?: number;
  end_frame?: number;
  /**
   * Inward-buffered room boundary at this entrant's L_map. Drawn together
   * with the session-level ``boundary`` (even-odd fill rule) to produce
   * the safe-band annulus on the map-video overlay. Optional — older
   * session JSONs without this field gracefully skip the overlay.
   */
  band_inner_polygon?: Array<[number, number]>;
};

export type MoveAlongWallSummary = {
  excursion_count: number;
  total_too_close_time_sec: number;
  total_too_far_time_sec: number;
  per_entrant?: MoveAlongWallEntrant[];
};

export type MetricRecord =
  | {
      metric_id: 'entrance_vectors';
      label: string;
      score: number;
      summary: EntranceVectorsSummary;
      timeline_item_ids: string[];
      flag_ids: string[];
    }
  | {
      metric_id: 'entrance_hesitation';
      label: string;
      score: number;
      summary: EntranceHesitationSummary;
      timeline_item_ids: string[];
      flag_ids: string[];
    }
  | {
      metric_id: 'total_time_of_entry';
      label: string;
      score: number;
      summary: TotalEntryTimeSummary;
      timeline_item_ids: string[];
      flag_ids: string[];
    }
  | {
      metric_id: 'move_along_wall';
      label: string;
      score: number;
      summary: MoveAlongWallSummary;
      timeline_item_ids: string[];
      flag_ids: string[];
    };

export type FlagType =
  | 'pair_time_violation'
  | 'vector_unknown'
  | 'vector_direction_violation'
  | 'total_entry_time_violation'
  | 'wall_too_close'
  | 'wall_too_far';

export type FlagRecord = {
  flag_id: string;
  metric_id: string;
  linked_item_id: string | null;
  type: FlagType;
  severity: string;
  frame: number | null;
  time_sec: number | null;
  start_frame?: number;
  end_frame?: number;
  start_time_sec?: number;
  end_time_sec?: number;
  title: string;
  message: string;
};

export type Entity = {
  track_id: number;
  entry_number?: number;
  label: string;
};

export type ArtifactsBlock = {
  videos: Partial<Record<'original' | 'motion_camera' | 'motion_map' | 'gaze_camera' | 'gaze_map', string>>;
  images: Partial<Record<'empty_map', string>>;
  data: Partial<Record<'tracker_output' | 'position_cache' | 'gaze_cache' | 'metrics_cache', string>>;
};

export type DrillWindow = {
  start_frame: number;
  end_frame: number;
  start_time_sec: number;
  end_time_sec: number;
  end_uncertain: boolean;
  decision_reason: string;
  matched_segment: string | null;
  candidates: unknown[];
  required_words: string[];
};

export type TranscriptionWord = {
  word: string;
  start: number;
  end: number;
  score?: number;
};

export type TranscriptionSegment = {
  id?: number;
  start: number;
  end: number;
  text: string;
  words?: TranscriptionWord[];
};

export type TranscriptionBlock = {
  language?: string;
  model?: string;
  aligned: boolean;
  schema_version?: string;
  audio_window?: { start_sec: number; end_sec: number | null };
  segments: TranscriptionSegment[];
};

export type AnalysisSession = {
  schema_version: '2.0';
  session: SessionInfo;
  video: VideoInfo;
  timeline: { items: TimelineItem[] };
  metrics: MetricRecord[];
  flags: FlagRecord[];
  entities: { friends: Entity[]; enemies: Entity[] };
  artifacts: ArtifactsBlock;
  drill_window?: DrillWindow;
  transcription?: TranscriptionBlock;
  /**
   * Room boundary polygon in map-image pixel coordinates (the same space
   * the aux/map video frames are rendered in). Used by the wall-band
   * overlay to draw the safe-band annulus on the map. Optional —
   * legacy sessions produced before this field existed will simply not
   * render the overlay.
   */
  boundary?: Array<[number, number]>;
  // Injected by the backend at /session load:
  session_json_path?: string;
  resolved_video_path?: string | null;
  run_info?: RunInfo;
  run_dir?: string;
};

// --- Run manifest + Compare-mode payloads ----------------------------------

export type Role = 'expert' | 'trainee';

export interface RunInfo {
  schema_version: string;
  run_id: string;
  role: Role;
  title: string;
  video_basename: string;
  vmeta_path: string | null;
  created_at: string;
  analysis_file: string;
}

export interface RunSummary {
  run_id: string;
  title: string;
  role: Role;
  path: string;
  created_at: string | null;
  analysis_file_exists: boolean;
}

export interface CompareSide {
  score: number | null;
  role: Role | null;
  run_id: string | null;
}

export interface CompareVisualization {
  image_path: string;
  label?: string;
  caption: string;
}

export interface CompareResult {
  metric_id: string;
  label: string;
  current: CompareSide;
  other: CompareSide;
  visualizations: CompareVisualization[];
  explanation: string;
}
