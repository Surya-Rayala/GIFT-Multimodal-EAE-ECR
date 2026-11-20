# Live stream input

The EAE processes a **live stream** exactly like a recorded video. GIFT's streaming
integration delivers H.264 + AAC in **MPEG-TS over UDP**
(`VideoProcessing.streamVideoFile` → `udp://127.0.0.1:1234?pkt_size=1316`; live GoPro
feeds use other addresses/ports). `VideoProcessing.setupVideoProcessing` copies the
session's vmeta to `stream-<name>.vmeta.xml` with the video `location` replaced by the
URL and hands it to `mt_initialize` — nothing else changes on the GIFT side.

## Detection

A vmeta `technical/location` is a stream when it is a URL (`scheme:` + anything) whose
scheme is not `file:` — `udp://`, `udp:` (no slashes), `rtp://`, `rtsp://`, `srt://`,
`tcp://`, `http(s)://`, or any other `scheme://` ffmpeg understands. Windows drive paths
(`C:\…`) and ordinary paths are files. A stream URL is used verbatim (XML entities such
as `&amp;` decoded) and **never** joined onto the course folder (the bug that produced
the file-not-found error). Detection does not depend on one fixed address, so changing
ports/hosts (several GoPros) needs no configuration.

Artifacts are named from the vmeta **title**, not the URL (see the README's
*Output Folder*). GIFT's `stream-*.vmeta.xml` copies keep the original title, so a
stream run's files carry the same names as a file run of the same session.

## How a stream becomes a "static" video

```
UDP / RTSP / SRT …  (video + audio multiplexed on one socket)
      │
StreamRecorder — ffmpeg subprocess, independent of the Python process
      │  receives in real time (9 MB UDP FIFO, 4 MB socket buffer, overruns
      │  non-fatal), re-joins timestamp wraps/discontinuities, stream-copies
      ▼  everything into a growing, lossless {title}_StreamCapture.ts
capture file on disk  ← the buffer between real-time reception and the slower
      │                 vision loop, and the raw archive of the session
      ├─► audio follower (PyAV thread) — decodes audio at the live edge, places
      │     it on the video clock (gap → silence, overlap → trimmed):
      │     16 kHz mono for the ASR + native-rate archive
      └─► video source (PyAV, vision-loop thread) — decodes at the loop's pace,
            measures the frame rate, maps timestamps onto a constant-frame-rate
            grid → frames for detection/pose/tracking, also written to {title}.mp4
```

* **One receiver, both tracks.** A unicast UDP port can be read by one socket only, so
  video and audio are split from the single recorded MPEG-TS — never from two readers
  racing for packets.
* **Reception never waits for processing.** The vision loop is usually slower than real
  time. A slow consumer of a socket loses packets; here the recorder writes
  compressed packets to disk at line rate. The frame loop catches up from the file at
  its own pace, waiting at the live edge.
* **The clock is the stream's timestamps, not arrival time.** Network jitter and bursts
  change *when* packets arrive, not the presentation timestamps (PTS) they carry. Frame 1
  is the first decodable frame (the *epoch*). Every later frame lands in slot
  `round((PTS − epoch) × fps)`:
  * a missing slot (lost packets, a VFR source, a stall) is filled by repeating the
    previous frame. Frame N stays at t = (N−1)/fps, so every frame→seconds conversion
    in the metrics, the drill window and the viewer stays correct. A repeated frame
    reuses the previous detection/pose result, since the image is identical and the
    model is stateless. The tracker still advances one step.
  * an extra frame for an already-filled slot (source faster than nominal) is dropped.
  * a frame that jumps more than 0.5 s is only held until the next frame(s) confirm it.
    A decoder can release a stale pre-outage frame after an outage (seen in testing),
    and a corrupt timestamp can point anywhere. Such a frame is dropped rather than
    allowed to shift everything after it.
  * a timestamp jump is either a real outage or a sender/encoder reset.
    * Backward jumps are always resets, and the recorder re-joins them.
    * A forward jump shorter than the idle timeout can be a real outage (packets stopped,
      then resumed), so it is kept as elapsed time and filled.
    * A longer forward jump cannot be an outage, because the recorder would have ended
      the stream. It is re-joined as a reset.
    * Both thresholds (`stream_discontinuity_sec`, `stream_max_gap_fill_sec`) therefore
      default to the idle timeout + 1 s.
* **Frame rate from the stream, not the config.** The rate comes from the stream's
  codec/container metadata when it agrees (±3 %) with the timestamps of the first
  `stream_fps_probe_sec` of video. Otherwise it is measured from those timestamps by a
  least-squares fit (exact enough to tell 29.97 from 30). It is snapped to a standard
  rate. `config.frame_rate` is only a fallback, and a mismatch is logged. A stream at a
  different rate from the recording the room was set up on (e.g. GoPro live at 30 vs
  recorded at 60) is handled. Every metric works in seconds via this fps.
* **Audio on the video clock.** Each audio frame is placed at `PTS − epoch` on the same
  timeline. Audio that starts before frame 1 is trimmed, audio that starts later is
  padded, and lost audio packets become silence of the right length, so drill-end
  speech times map to the right frames. A plain file is assumed to start audio and video
  together; for a stream this is measured.
  * The 16 kHz ASR view is positioned by the resampler's own output timestamps, because
    the resampler holds back ~16 samples. Frames that arrive without a PTS (several AAC
    frames in one PES packet) are stamped as contiguous first, and the first sample is
    anchored exactly, not within the 4 ms jitter tolerance.
  * Result: the ASR audio is bit-identical to ffmpeg's own decode of the received stream
    and sits within ⅓ sample (21 µs) of frame 1. After an outage it stays within 1 sample.
  * The static path cuts audio from the container start. In an MPEG-TS that is the
    (earlier) audio start, so a TS file processed statically hears speech 21–33 ms late.
    MP4 files are unaffected.
* **After processing**, `{title}.mp4` (the processed frames plus the archived audio,
  muxed on the same clock) becomes the run's source video. Camera overlays, audio
  muxing, the POD camera still and the Analysis Viewer's *Original* mode all use it like
  a recorded file. Frame N of every artifact is frame N of the analysis.

## When processing stops

| Reason (`StreamReport.stop_reason`) | Trigger |
|---|---|
| `drill_end` | "room … clear" confirmed and the frame loop reached the drill end (`stop_at_drill_end`, default) |
| `stream_ended` | no data for `stream_idle_timeout_sec` — the sender finished or the link died |
| `max_after_entry` | no drill end confirmed `stream_max_after_entry_sec` after the first entry |
| `max_duration` | `stream_max_duration_sec` of stream processed |
| `connect_timeout` | nothing received within `stream_connect_timeout_sec` (run fails with a clear error; the report and recorder log say why) |

A stream that ends before "room clear" is not an error. The drill window ends at the
last received frame (`end_uncertain`), and every artifact is still written.

The loop learns the drill end only when the ASR confirms it, so it may already have
processed frames past the end. Those frames are discarded before analysis, exactly as if
the loop had stopped on the end frame. The results therefore do not depend on how fast
transcription ran relative to the frame loop (machine load, GPU vs CPU, frame rate).

## Artifacts (in addition to the usual ones)

| File | Content |
|---|---|
| `{title}.mp4` | the analyzed stream as a normal CFR video with audio (`video.video_path`) |
| `{title}_StreamCapture.ts` | lossless capture of everything received (archive; `stream_keep_capture`) |
| `{title}_StreamReport.json` | what arrived and how it was repaired: fps source / metadata / measured, source vs frame size, frames decoded / delivered / repeated / dropped, every gap (frame, length), re-anchors, audio gaps and offsets, recorder issues (continuity errors, corrupt packets, discontinuities, FIFO overruns), stop reason |
| `{title}_StreamRecorder.log` | the recorder's ffmpeg log |

`Analysis.json` → `video.source` summarizes the stream: URL (credentials redacted), fps
source, repeated/dropped frames, gaps, timestamp outliers, stop reason. It also includes
`drill_window_repeated_frames` / `_sec` / `_gaps`: how much of the drill window rests on
repeated frames rather than received video (also logged as a warning). An outage during
the entries legitimately changes the scores, and this says by how much data.
`RunMetadata.json` → `source` holds the same summary.

## Config keys (room JSON, all optional)

| Key | Default | Meaning |
|---|---|---|
| `stream_connect_timeout_sec` | 120 | wait for the stream to start |
| `stream_idle_timeout_sec` | 15 | silence that ends the stream |
| `stream_max_duration_sec` | 3600 | cap on processed stream time |
| `stream_max_after_entry_sec` | 600 | cap after first entry without a drill end |
| `stream_frame_rate` | — | force the frame rate (skip detection) |
| `stream_fps_probe_sec` | 2.0 | video used to measure the rate |
| `stream_max_gap_fill_sec` | idle timeout + 1 s (16) | longest gap filled with repeated frames |
| `stream_discontinuity_sec` | idle timeout + 1 s (16) | forward timestamp jump treated as a reset (ffmpeg `dts_delta_threshold`) |
| `calibration_frame_size` | — | `[W, H]` of the video the room was mapped on; stream frames of another size are scaled to it so the pixel→map homography stays valid |
| `stream_udp_fifo_size` / `stream_udp_buffer_size` | 50000 (×188 B) / 4 MiB | receive buffering |
| `stream_input_options` | `[]` | extra ffmpeg input options (e.g. `["-rtsp_transport", "tcp"]`) |
| `stream_keep_capture` | true | keep `{title}_StreamCapture.ts` |

UDP URLs get `fifo_size`, `overrun_nonfatal`, `buffer_size` and `timeout` appended unless
the URL already sets them. `streaming_transcription: false` (the static-video audio
prepass) cannot apply to a live stream and is overridden with a log line.

## Multiple streams in one request

`mt_initialize` starts a recorder for **every** stream vmeta of the request before it
processes the first one. Streams that send concurrently (several GoPros) are captured
from the start, even though they are analyzed one after another.

## Limits

* UDP sent before the EAE listens is gone. Start the stream after (or right as)
  `mt_initialize` is called. A stream joined mid-way starts at the next keyframe (x264
  defaults allow ~4 s between keyframes). If that is after the team is in view, their
  tracking history is missing and entries can be tracked differently.
* An outage longer than `stream_idle_timeout_sec` ends the session.
* The homography is only valid at the resolution the room was mapped on. Set
  `calibration_frame_size` when a stream's resolution differs (GoPro live previews
  usually do).
* A pixel-damaged frame (packet loss inside a frame) is decoded with the decoder's error
  concealment, as any player would show it. Only frames that are missing entirely are
  repeated.

## Validation (2026-11-20, three clips, MPS)

Every control was run on the three test clips, each compared with a static run of the
same clip (`compare_runs.py`):
- scores;
- the drill window and the call that ended it;
- every timeline item and flag: present, time, and content (kind, label, entry/track
  numbers, direction, durations, flag type and message);
- the transcript word by word;
- tracker IDs and keypoints;
- artifact videos.

All 49 stream runs exited cleanly and wrote every artifact.

| Clip | vmeta | Length | First entry | Drill end | Static scores (vec / hes / tot / wall / POD cov / POD facing) |
|---|---|---|---|---|---|
| Iguana `3-TrimmedV2` | `run1` | 45 s | 22.1 s | "clear," at 29.3 s → frame 1788 | 1.0 / 1.0 / 1.0 / 0.91 / 0.39 / 1.0 |
| Crested Gecko `3-Trimmed` | `run2` | 26 s | 4.7 s | no call → last frame | 0.33 / 0.79 / 1.0 / 0.84 / 0.39 / 1.0 |
| Crested Gecko `14-Trimmed` | `run3` | 13 s | 7.7 s | no call → last frame | 1.0 / 0.87 / 1.0 / 0.59 / – / – |

Differences from the static run ("=" identical; "fr" = frames; late joins and restarts
compared after their offset):

| Control | Iguana | Crested Gecko 3 | Crested Gecko 14 |
|---|---|---|---|
| Lossless (`-c copy` stream) | identical¹ | identical¹ | identical¹ |
| GIFT-style re-encode (libx264 + AAC) | window =; wall −0.01, POD cov +0.02 | entry +3 fr; POD hold detected 2.7 s later² (cov −0.02) | entry +1 fr; hes +0.01 |
| EAE waits 20 s for the sender | = re-encode | = re-encode | = re-encode |
| Sender restart (0.5 s gap, PTS reset) | = re-encode | POD cov −0.03 | all scores = |
| Late join (EAE starts after the sender) | = re-encode | = re-encode | joined at the next keyframe, 0.2 s before the first entry³: hes −0.10, wall +0.09 |
| 29.97 fps | all scores = | hes −0.01 | hes −0.02, wall +0.02 |
| VFR, 12 % of frames dropped | wall −0.01, POD cov +0.01 | POD cov −0.03 | wall +0.01 |
| Half resolution (scaled back) | entry −6 fr; POD cov +0.05 | hes +0.01 | hes +0.01, wall +0.01 |
| 40 Mbps link | = re-encode | tail drops in the last second; POD cov −0.02 | tail drops; hes +0.01, wall +0.03 |
| No audio | no drill end → window to stream end | = re-encode | = re-encode |
| Outage before the drill (3 s / 1 s / 3 s) | = re-encode, gap filled | = re-encode, gap filled | freeze ended 2 s before the entry; the tracker lost people near the door⁴: entry 0.55 s early, hes −0.12 |
| 12 s outage | = re-encode, 12 s filled | window =; wall +0.15 (frozen frames in the window) | — (clip too short) |
| 0.3 % loss + bursts, 40 ms jitter | wall −0.08 | vec +0.34, hes −0.05, wall −0.03 | hes +0.11, wall −0.13 |
| 1 s outage during the entries | hes −0.26, vec −0.33, wall −0.20 | vec +0.67, wall −0.03 | hes −0.16, vec −0.33, wall −0.22 |
| Sender cut mid-drill | ends at the last frame (`end_uncertain`) | ends at the last frame | entry 4 never seen |
| 1 % loss, 80 ms jitter | first "clear" misheard → end +2.0 s; degraded | entry 4 lost; degraded | most entries lost |

¹ Identical to a static run of the same TS file in every respect: scores, window, every
event and flag, and tracker IDs and keypoints on every frame. Iguana is also identical to
its MP4 run after the 12-frame offset. On the Crested Gecko clips a few tracker IDs differ
from the MP4 run, because the TS starts 56 / 42 frames later; scores, events and flags
still match.
² The automatic POD hold detector is threshold-based. Re-encoding noise in the keypoints
moved its first "collective hold". The lossless run finds it on the same frame.
³ The sender's x264 places keyframes up to ~4 s apart, and a late joiner waits for one.
With no tracker history of the people already in the room, the entries are tracked
differently. Start the stream before the drill.
⁴ A 3 s freeze followed by everyone having moved breaks the tracks. A track re-born in
the entry region counts as an entry.

Up to the impairment, every row stays on the static timeline. Score changes come from
what the impairment did to the video or audio, not from the stream handling:
re-encoding noise, frames that never arrived during the entries, a late join, or heavy
loss. When frames are missing inside the drill, `video.source.drill_window_repeated_frames`
says how many.

More checks:
* **Speed independence.** The Iguana 29.97 fps stream was run with the loop overshooting
  the drill end by 110 frames and again with no overshoot. The results were
  byte-identical.
* **Config controls**, each identical to the plain stream:
  * `streaming_transcription: false` on a stream (overridden and logged);
  * `stream_frame_rate` forced (reported as `config_override`);
  * `stream_keep_capture: false` (no capture kept).
* **Transcription modes.** Static Iguana in prepass mode ends 2 frames from streaming
  mode, with all six scores identical. With segment ends, the two modes gave wall
  0.91 / 0.93.
* **Old outage thresholds.** With 10 s thresholds, a 12 s outage put every event 12 s
  early. With the new defaults it is filled.
* **Confirmation speed.** Every run that finds the call confirms it on the 3rd ASR pass
  (36 s of audio), where the segment-based rule needed up to 5 (42 s).

**Speech.** The stream's ASR audio is bit-identical to ffmpeg's decode of what was
received, aligned to frame 1. Its transcript still differs from the static run's in
unclear words ("keep it" / "keep going"). The static path cuts every 8 s / 3 s chunk with
a fresh ffmpeg seek, while a live stream is decoded continuously. Parakeet changes unclear
words when the audio moves by a single 16 kHz sample (62.5 µs). Run offline on the same
MP4, the static reader reproduces the static transcript exactly, and the stream reader
gives exactly the two-word difference seen in Iguana's lossless stream. The drill-ending
phrase and its time were identical in every variant. Compare speech by the ending phrase
and the times of matched words, not by transcript identity.

**Drill end and ASR segmentation.** The drill used to end 0.5 s after the *segment* that
contained "room" and "clear", and Parakeet draws segments by punctuation. In Iguana's
29.97 fps and VFR streams it heard the same words at the same times as the static run but
wrote "room," instead of "room.", so the call merged with 6 s of later chatter and the
drill end moved with it. Static processing has the same sensitivity. The prepass and
streaming transcription modes ended this clip 1.1 s apart, and an earlier Whisper build
moved its window from 1885 to 1725 purely by re-segmenting.

The drill now ends 0.5 s after the **word that completes the call**, taken from the ASR's
word timestamps, which stayed put in every variant. The same word times decide which
words count as after the entry, and streaming confirmation matches the call's time
across passes rather than a segment's start time. No threshold was fitted to a clip.
Effects on the 22 static transcripts available (Iguana, Crested Gecko, SamT Team 1):
* 20 are unchanged (no call).
* Iguana (`run1` / `test.vmeta.xml`) now ends at frame 1788 instead of 1855, on "clear,".
* SamT Team 1 `13_40_45` moves 4.3 s earlier. Its "call" is instructor talk ("…clear the
  room…"), a false match under either rule.

**30 fps note.** Timing at 29.97 fps is exact. On Iguana all six scores match the 60 fps
static run, and on the Crested Gecko clips they are within ±0.02. An earlier 30 fps run
scored hesitation 0.74. The cause was not the tracker but frames processed *after* the
drill end.
* The frame loop stops only once "room clear" is confirmed, so it can run past the end.
* At 30 fps it ran 81 frames past. A fifth person (track 5) was tracked through them and
  out-lasted entrant 4 in the "longest-tracked" entrant selection.
* Those frames are now discarded before analysis, so results no longer depend on how fast
  transcription ran relative to the frame loop.

The tracker's frame-count parameters (`delta_t=5`, `max_age=1000`, 3-frame in-room
confirmation) were tuned at 60 fps. 30 fps has been checked on these three clips only.

## Testing

* `pytest tests/test_stream_ingest.py`: URL detection, naming, frame-rate decision,
  CFR mapping (jitter, drops, extra frames, gaps, discontinuities), audio timeline
  (epoch alignment, gaps, overlaps, blocking reads). It also runs live UDP round trips
  of a marker clip: a white flash and a 1 kHz beep at t = 1.0 s must land on frame
  31 / t = 1.0 s (±25 ms), including with 40 % of frames dropped by the sender (VFR)
  and at 25 fps.
* Two tests check the ASR audio to the sample:
  * an AAC-in-TS file must decode bit-identically to ffmpeg at the exact frame-1 offset;
  * PTS-less frames must be placed exactly like timestamped ones.
* `pytest tests/test_drill_window.py` locates the call by word timestamps however the ASR
  punctuates. The test uses Parakeet's real word timings for Iguana, in both punctuations
  it produced, and includes streaming confirmation across re-segmented passes.
* `tools/stream_sim/`: end-to-end harness.
  * `run_scenario.py` streams an existing video to the full EAE, GIFT-style.
  * `udp_impair_proxy.py` injects loss, bursts, outages, jitter/reordering,
    duplication and bandwidth limits.
  * `compare_runs.py` compares a stream run with the static run of the same video:
    scores, drill window and its ending phrase, every timeline item/flag (time and
    content), the transcript word by word, tracker ids/keypoints, and the artifact
    videos.
