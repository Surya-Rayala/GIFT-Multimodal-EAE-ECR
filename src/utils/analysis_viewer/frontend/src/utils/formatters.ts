// Generic value formatters for the Detail panel's Summary tab.
// Replaces the polymorphic per-kind formatters from src/utils/gui/detail_panel.py.

export type SummaryRow = { label: string; value: string };

export function formatScore(score: number | null | undefined): string {
  if (typeof score !== 'number' || Number.isNaN(score) || score < 0) return 'N/A';
  return score.toFixed(2);
}

/**
 * Relative-performance band for the Compare view. Grades the current score
 * by its ratio to the comparison score (not by absolute thresholds), because
 * "good" varies by room.
 *
 * Bands:
 *   ratio >= 1.00  → "above"      (dark green)
 *   0.90–1.00      → "at"         (light green)
 *   0.70–0.90      → "near"       (yellow)
 *   < 0.70         → "below"      (red)
 *
 * Edge cases: if `other === 0` we treat any positive `current` as "above",
 * and matching zeros as "at". Missing / non-finite values return null.
 */
export type RelativeBand = 'above' | 'at' | 'near' | 'below';

export function relativeAssessment(
  current: number | null | undefined,
  other: number | null | undefined,
): RelativeBand | null {
  if (
    typeof current !== 'number' ||
    typeof other !== 'number' ||
    !Number.isFinite(current) ||
    !Number.isFinite(other)
  ) {
    return null;
  }
  if (other === 0) {
    if (current === 0) return 'at';
    return current > 0 ? 'above' : 'below';
  }
  const ratio = current / other;
  if (ratio >= 1.0) return 'above';
  if (ratio >= 0.9) return 'at';
  if (ratio >= 0.7) return 'near';
  return 'below';
}

export function formatTime(timeSec: number | null | undefined): string {
  if (typeof timeSec !== 'number' || Number.isNaN(timeSec)) return '—';
  return `${timeSec.toFixed(2)}s`;
}

export function formatFrame(frame: number | null | undefined): string {
  if (typeof frame !== 'number' || Number.isNaN(frame)) return '—';
  return frame.toString();
}

export function formatBool(value: boolean | null | undefined): string {
  if (value === true) return 'Yes';
  if (value === false) return 'No';
  return '—';
}

/**
 * Pretty-print a value of unknown shape: numbers, booleans, arrays of numbers,
 * tuples like [x, y]. Falls back to JSON for anything more complex.
 */
export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'boolean') return formatBool(value);
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toString();
    return value.toFixed(3);
  }
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) {
    if (value.length === 2 && value.every((v) => typeof v === 'number')) {
      const [x, y] = value as [number, number];
      return `(${x.toFixed(1)}, ${y.toFixed(1)})`;
    }
    if (value.every((v) => typeof v === 'number')) {
      return `[${value.map((v) => formatValue(v)).join(', ')}]`;
    }
    return JSON.stringify(value);
  }
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

/** Convert a snake_case_key to "Title Case Label". */
export function formatKey(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b([a-z])/g, (_, c) => c.toUpperCase())
    .replace(/\bXy\b/g, 'XY');
}

/** Walk a flat object and produce {label, value} rows for the Summary tab. */
export function formatObjectRows(obj: Record<string, unknown>): SummaryRow[] {
  const out: SummaryRow[] = [];
  for (const [key, value] of Object.entries(obj)) {
    if (value === null || value === undefined) continue;
    if (typeof value === 'object' && !Array.isArray(value)) {
      // Nested object — skip (the Raw tab shows everything verbatim)
      continue;
    }
    out.push({ label: formatKey(key), value: formatValue(value) });
  }
  return out;
}
