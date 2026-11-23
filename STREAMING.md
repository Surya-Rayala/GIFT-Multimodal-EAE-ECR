# Live stream input

The EAE analyzes a live stream the same way as a recorded video. GIFT streams H.264 + AAC
as **MPEG-TS over UDP** (`udp://127.0.0.1:1234?pkt_size=1316` by default; GoPro feeds use
other addresses). GIFT copies the session vmeta to `stream-<name>.vmeta.xml`, replaces its
`location` with the stream URL, and passes it to `mt_initialize`. Nothing else changes.

## How a stream is recognized

- A vmeta `location` is a stream when it is a URL (`scheme:…`) other than `file:` —
  `udp://`, `rtp://`, `rtsp://`, `srt://`, `tcp://`, `http(s)://`, or anything else ffmpeg
  can open. Paths, including Windows paths (`C:\…`), are files.
- The URL is used as written and is never joined onto the course folder.
- Any address or port works, so several cameras need no extra configuration.
- Output files are named from the vmeta **title**, the same as for a file run.

## Try it with a recorded video

Copy a session vmeta and change only its location:

```xml
<location>udp://127.0.0.1:1234?pkt_size=1316</location>
```

Then use two terminals:

```bash
# 1. The EAE: waits for the stream on that URL
python run_engine_local.py input/stream-run1.vmeta.xml -o output/run1_stream

# 2. Once terminal 1 says "waiting for the stream": send a video the way GIFT does
ffmpeg -re -i input/Videos/Iguana/3-TrimmedV2.mp4 -map 0:v:0 -map 0:a:0 \
  -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 128k \
  -f mpegts "udp://127.0.0.1:1234?pkt_size=1316"
```

Or run both in one command, and compare against a file run of the same video:

```bash
python tools/stream_sim/run_scenario.py --name try1 --out output/stream_sim \
  --source input/Videos/Iguana/3-TrimmedV2.mp4 --vmeta input/run1.vmeta.xml \
  --encode "-map 0:v:0 -map 0:a:0 -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 128k"
python run_engine_local.py input/run1.vmeta.xml -o output/run1_file
python tools/stream_sim/compare_runs.py output/run1_file output/stream_sim/try1/run
```

## What happens to a stream

- **One recorder, both tracks.** An ffmpeg recorder receives the stream in real time and
  saves it losslessly to `{title}_StreamCapture.ts`. Video and audio are both read from
  that file, so the recorder never drops packets, even when processing runs slower than
  real time.
- **Timestamps, not arrival times.** Frame N is always at t = (N−1)/fps:
  - a missing frame (packet loss, a variable frame rate, an outage) is filled by repeating
    the previous frame and its pose result;
  - an extra frame is dropped;
  - a frame whose timestamp jumps more than 0.5 s must be confirmed by the next frames, so
    one stale or corrupt frame cannot shift the timeline;
  - a jump longer than the idle timeout cannot be an outage, so it is treated as a sender
    restart and joined up.
- **Frame rate from the stream.** Taken from the stream's metadata when it agrees with the
  measured timestamps, otherwise measured. The config's `frame_rate` is only a fallback,
  so a 30 fps feed of a room set up at 60 fps works.
- **Audio on the same clock.** Audio is placed by its timestamps relative to frame 1, and
  lost audio becomes silence of the right length. The speech recognizer hears exactly what
  was received.
- **A normal video afterwards.** `{title}.mp4` (the processed frames plus the stream audio)
  becomes the run's video for overlays and the Analysis Viewer.
- **Several streams.** `mt_initialize` starts a recorder for every stream vmeta in the
  request before it processes the first one, so concurrent cameras are captured from the
  start.

## When processing stops

| `StreamReport.stop_reason` | When |
|---|---|
| `drill_end` | the "room … clear" call was confirmed and processing reached the drill end |
| `stream_ended` | no data for `stream_idle_timeout_sec` (the sender finished or the link died) |
| `max_after_entry` | no drill end within `stream_max_after_entry_sec` of the first entry |
| `max_duration` | `stream_max_duration_sec` of stream processed |
| `connect_timeout` | nothing arrived within `stream_connect_timeout_sec` (the run fails with a clear error) |

- **A stream that ends before "room clear" still writes every output.** The drill window
  ends at the last received frame and is marked `end_uncertain`.
- **Results don't depend on machine speed.** The drill end is known only once the speech is
  confirmed, so frames processed after it are discarded.

## Output files (in addition to the usual ones)

| File | Content |
|---|---|
| `{title}.mp4` | the analyzed stream as a normal video with audio |
| `{title}_StreamCapture.ts` | lossless capture of everything received (`stream_keep_capture`) |
| `{title}_StreamReport.json` | what arrived and how it was repaired: frame rate and its source, repeated/dropped frames, gaps, audio gaps, recorder issues, stop reason |
| `{title}_StreamRecorder.log` | the recorder's ffmpeg log |

`video.source` in `Analysis.json` summarizes the stream. Its `drill_window_repeated_frames`
says how much of the drill window rests on repeated rather than received video.

## Config keys (room JSON, all optional)

| Key | Default | Meaning |
|---|---|---|
| `stream_connect_timeout_sec` | 120 | how long to wait for the stream to start |
| `stream_idle_timeout_sec` | 15 | silence that ends the stream |
| `stream_max_duration_sec` | 3600 | cap on processed stream time |
| `stream_max_after_entry_sec` | 600 | cap after the first entry without a drill end |
| `stream_frame_rate` | — | force the frame rate instead of detecting it |
| `stream_fps_probe_sec` | 2.0 | video used to measure the frame rate |
| `stream_max_gap_fill_sec` | idle timeout + 1 s | longest gap filled with repeated frames |
| `stream_discontinuity_sec` | idle timeout + 1 s | forward timestamp jump treated as a sender restart |
| `calibration_frame_size` | — | `[W, H]` of the video the room was mapped on; other sizes are scaled to it |
| `stream_udp_fifo_size` / `stream_udp_buffer_size` | 50000 × 188 B / 4 MiB | receive buffering |
| `stream_input_options` | `[]` | extra ffmpeg input options, e.g. `["-rtsp_transport", "tcp"]` |
| `stream_keep_capture` | true | keep `{title}_StreamCapture.ts` |

UDP URLs get `fifo_size`, `overrun_nonfatal`, `buffer_size` and `timeout` added unless they
already set them. `streaming_transcription: false` cannot apply to a live stream and is
ignored, with a log line.

## Limits

- UDP sent before the EAE listens is lost, so start the stream after `mt_initialize`.
- A late join starts at the next keyframe (x264 places them up to ~4 s apart). If the team
  is already in view by then, their tracking history is missing.
- An outage longer than `stream_idle_timeout_sec` ends the session.
- Set `calibration_frame_size` when the stream's resolution differs from the video the room
  was mapped on (GoPro live previews usually do).
- Frames damaged by packet loss are analyzed as the decoder conceals them; only missing
  frames are repeated.
- Speech recognition reacts to tiny audio differences, so a stream's transcript can differ
  from a file's in unclear words. The drill end and the timing of the words both hear match.

## Validation

Three test clips (`run1`–`run3`), about 16 stream conditions each, were compared with
static runs of the same clips:

| Condition | Result vs the static run |
|---|---|
| Lossless stream of the same file | identical: scores, drill window, every timeline event and flag, tracker on every frame |
| GIFT-style re-encode; EAE waiting for the sender; sender restart; 29.97 fps; variable frame rate; half resolution; 40 Mbps link; outages before the drill (up to 12 s) | same drill window and events; scores within 0.05 (re-encoding noise) |
| Late join or freeze right before the first entry | same timeline; tracking can differ because the team's history is cut off |
| Outage during the entries; heavy packet loss | scores change where video was missing; the run reports how much |

The same stream also gives identical results whether or not processing runs past the
drill end, and with `stream_frame_rate`, `stream_keep_capture` or `streaming_transcription`
set.

## Testing tools (`tools/stream_sim/`)

- `run_scenario.py` streams a video to the full EAE the way GIFT does, optionally through
  the proxy below.
- `udp_impair_proxy.py` adds packet loss, loss bursts, outages, jitter, reordering,
  duplication and bandwidth limits.
- `compare_runs.py` compares two runs: scores, drill window, every timeline event and flag,
  transcript, tracker and videos (`--offset-sec` for a late join).

Example, a 1 s outage during the entries:

```bash
python tools/stream_sim/run_scenario.py --name outage --out output/stream_sim \
  --source input/Videos/Iguana/3-TrimmedV2.mp4 --vmeta input/run1.vmeta.xml \
  --encode "-map 0:v:0 -map 0:a:0 -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 128k" \
  --proxy "--outage 23.8:1"
```
