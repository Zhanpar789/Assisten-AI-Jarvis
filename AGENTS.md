# JARVIS Operational Guide

This document describes the implementation currently present in this workspace. It is an operational guide for future coding agents, not a proposed redesign.

## Project Overview

JARVIS is a manually started, local-only macOS voice helper for Apple Silicon. The application entrypoint is `jarvis.py`. It uses one `sounddevice` microphone stream, sherpa-onnx for keyword spotting and speech processing, an in-process whisper.cpp bridge for command transcription, and fixed allowlisted command handlers. It has no cloud speech, telemetry, saved audio, automatic startup, or runtime downloader.

There is no Git metadata in the project root (`git status` cannot be used here). Do not infer repository history or ownership from generated/vendor directories. `whisper.cpp` has its own nested project guidance in `whisper.cpp/AGENTS.md`; respect it when touching that vendored tree.

## Current Architecture

The active configuration is `COMMAND_ASR = "whisper"` in `jarvis.py`.

```text
sounddevice InputStream, 44.1 kHz mono
    |
    +--> DoubleClapDetector
    +--> bounded audio_queue --> speech_worker --> resampler --> Silero VAD
    |                                            |
    |                                            +--> VAD speech segment --> Whisper C API --> strict intent parser
    +--> bounded kws_audio_queue --> kws_worker --> sherpa-onnx KWS
                                                       |
                                                       +--> wake-pending event / standalone direct_call
main thread <-- bounded intent_queue <-- speech/KWS workers
    |
    +--> fixed macOS command handler and optional TTS
```

KWS and VAD are parallel consumers of the continuously forwarded microphone stream. KWS does not directly execute arbitrary commands and does not gate VAD. In active Whisper mode, VAD-confirmed speech is transcribed even when no KWS event is pending; the strict parser is the execution gate.

`COMMAND_ASR = "nemotron"` is an existing rollback mode. In that mode the speech worker uses Nemotron endpoint detection and its result directly instead of Silero VAD plus Whisper. Do not change this switch casually.

## Audio Pipeline

- `sounddevice.InputStream` is created in `main()` at 44,100 Hz, block size 1,024, one `float32` channel.
- The callback is `AudioRouter.audio_callback()`.
- The callback first returns when shutdown or `speech_suppressed` is set.
- Otherwise it runs the unchanged double-clap detector, copies the mono block, and tries to enqueue it into both bounded queues.
- Each audio queue holds at most 32 blocks, about 0.74 seconds at 44.1 kHz. On overflow, the oldest queued block is discarded.
- Audio is not written to disk or sent over the network.

The clap detector uses peak, RMS, crest factor, debounce, interval, cooldown, and rearm thresholds. A valid double clap sets `work_action_requested`; it does not wake the voice command path. The first work action launches Mattermost and the configured Chrome URLs; later claps do nothing because `actions_launched` is set.

## KWS Behavior

`create_keyword_spotter()` loads the int8 Zipformer KWS files and `keywords_jarvis.txt`, whose configured keyword is JARVIS. It uses 16 kHz input, one CPU thread, keyword score 1.0, and threshold 0.25.

`kws_worker()` owns the KWS stream. It continuously consumes `kws_audio_queue`, decodes ready frames, resets its stream after a result, drains its queue, and sets `kws_wake_requested`. It never dispatches an application command.

After KWS detects JARVIS, the speech worker waits up to `KWS_STANDALONE_TIMEOUT_SECONDS` (1.5 seconds). If no VAD speech is active, it enqueues `direct_call` with source `kws`. If a speech segment completes while KWS is pending, the transcription is interpreted with KWS context: a valid command following the wake word is accepted; otherwise it becomes a direct call. The main thread clears the KWS event after handling a KWS direct call.

`hotwords.txt` exists at the project root but is not referenced by `jarvis.py`. The active KWS keyword file is the model-directory `keywords_jarvis.txt`.

## VAD Behavior

`create_voice_activity_detector()` verifies and loads `models/silero_vad.int8.onnx` through sherpa-onnx. It uses threshold 0.5, minimum speech 0.15 seconds, trailing silence 0.45 seconds, a 0.25-second pre-roll, maximum speech 15 seconds, 512-sample windows, 16 kHz, and one CPU thread.

The speech worker continuously resamples 44.1 kHz blocks to 16 kHz. While VAD is inactive, it retains only the bounded pre-roll. When VAD starts, it begins `speech_audio` with that pre-roll and the current VAD output. It appends audio while speech is active. When VAD ends, it drains VAD output, concatenates the segment, logs its duration, and sends the complete segment to Whisper. VAD-confirmed audio is the current speech-centered gate, but it is not a semantic or wake-word gate.

## Nemotron Behavior

The configured Nemotron model is `models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-1120ms-int8-2026-04-25`. `create_recognizer()` loads its tokens, encoder, decoder, and joiner with sherpa-onnx's online transducer recognizer, greedy search, endpoint detection enabled, 16 kHz, feature dimension 128, two CPU threads, and CPU provider.

Nemotron is inactive in the current default configuration. If selected, its stream receives the original 44.1 kHz blocks as configured by the existing code, decodes while ready, and on endpoint obtains a result, resets the stream, and passes the text to the same intent processing path. A second 560 ms Nemotron model directory and older Zipformer model directories are present but are not used by the active code.

## Whisper Behavior

`WhisperCommandRecognizer` loads the verified local model `whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin` through `native/build/libjarvis_whisper.dylib`. It verifies the exact file size and SHA-1 before loading. Audio is resampled to 16 kHz if needed and sent to the bridge as contiguous `float32` samples.

The bridge is `native/whisper_bridge.cpp`. It enables GPU use and flash attention in whisper.cpp, sets English, no context, no timestamps, one segment, greedy sampling at temperature 0, blank/non-speech suppression, and four threads. It returns only the concatenated transcription through a fixed 4,096-byte output buffer. Observed command inference has been about 1.7–1.8 seconds on the M1; reliability is more important than reducing this latency.

## Intent Parsing And Execution

`normalize_voice_transcription()` applies a small set of known speech substitutions, then `command_intent()` lowercases, removes punctuation, splits words, and applies exact allowlist rules. Negations reject the utterance. Unsupported, incomplete, or ambiguous text returns `None` and is not executed.

The intent constants currently defined are:

- `dbeaver`, `close_dbeaver`
- `mongodb`, `close_mongodb`
- `postman`, `close_postman`
- `docker`, `close_docker`
- `terminal`, `close_terminal`
- `stickies`, `close_stickies`
- `vpn_on`, `vpn_off`
- `stay_awake`, `sleep_normally`, `special_sleep`
- `direct_call`, `work_mode`, `gratitude`, `break_time`

`process_transcription()` is the common safety boundary. It normalizes text, applies KWS context when pending, and enqueues only a recognized internal intent. The main thread consumes the single-slot `intent_queue` and dispatches fixed handlers. Speech text is never used as a shell command, executable, path, application name, or subprocess argument.

Handlers use fixed `open`, `osascript`, `pgrep`, `scutil`, and `caffeinate` argument lists with the default `shell=False`. Supported application names and process names are source constants. VPN state changes require an expected current state and are verified by polling for the target state.

`break_time` accepts clear break/rest requests and its main-thread handler first speaks a randomized concise response using either `boss` or `sir`. After the blocking TTS path completes, it opens `https://www.youtube.com/` in the default browser using macOS `open`; it performs no other system action.

## TTS Suppression And Reset

All response speech goes through `speak_while_suppressed()` when suppression/reset arguments are available. It sets `speech_suppressed`, drains the speech queue, runs macOS `say` using voice `Daniel`, waits 0.35 seconds for speaker settling, drains again, requests a worker reset, waits up to two seconds for reset completion, drains once more, and finally clears suppression.

While suppression is set, the audio callback forwards no new blocks. The KWS worker resets and discards input during suppression. The speech worker resets its Nemotron stream or VAD state, resampler, pre-roll, speech buffer, and KWS timing state. This is intended to prevent JARVIS TTS from becoming a command. The startup greeting is spoken before the microphone stream is opened.

## Queue And Threading Architecture

- Main thread: constructs models, speaks, launches applications, handles intents, owns the microphone stream context, and handles shutdown.
- Audio callback thread: performs clap metrics and non-blocking queue insertion only; callback failures request shutdown and abort the stream.
- `jarvis-speech-worker`: the sole owner of the active speech recognizer/VAD state and Whisper calls.
- `jarvis-kws-worker`: owns the KWS stream.
- `audio_queue` and `kws_audio_queue`: 32-block bounded queues with newest-data retention.
- `intent_queue`: one-slot bounded queue; newest intent replaces an older queued intent on overflow.
- `runtime_errors`: one-slot error queue used to report worker/callback failures to the main thread.
- `STOP_AUDIO` is used to unblock workers during shutdown.

Ctrl+C exits the stream context, suppresses and stops workers, drains queues, sends stop markers, joins workers with a three-second bound, stops `caffeinate` if active, and closes the Whisper context after the speech worker has stopped.

## Models And Exact Roles

- KWS: `models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`, active keyword file `keywords_jarvis.txt`, detects JARVIS only.
- Active endpoint gate: `models/silero_vad.int8.onnx`, official sherpa-onnx Silero VAD; size 212,860 bytes; SHA-256 `c36d490aff5ab924ca6c7aeec4d8f6bd3d22db6fa17611b9c5b17eae58ac3a20`.
- Active command ASR: whisper.cpp model `ggml-large-v3-turbo-q5_0.bin`, size 574,041,195 bytes; SHA-1 `e050f7970618a659205450ad97eb95a18d69c9ee`.
- Rollback streaming ASR: Nemotron 0.6b 1120 ms int8 model directory named above.

The required Python packages are pinned/minimally constrained in `requirements.txt`: NumPy, `sherpa-onnx==1.13.6`, and sounddevice. Do not download or replace any model as part of routine work.

## Important Files And Directories

- `jarvis.py`: all application configuration, audio routing, workers, recognizers, parser, handlers, and lifecycle code.
- `README.md`: user-facing setup, safety, and manual test notes; verify it against source before relying on it.
- `requirements.txt`: Python runtime dependencies.
- `native/whisper_bridge.cpp`: C API bridge and Whisper decoding parameters.
- `native/CMakeLists.txt`: bridge build; links to the prebuilt official whisper.cpp library.
- `native/build/`: generated Release build currently containing `libjarvis_whisper.dylib`.
- `models/`: local sherpa-onnx model directories, archives, and verified Silero VAD file.
- `whisper.cpp/`: vendored official whisper.cpp source, build tree, scripts, and local model.

## Current Known Problems

The primary unresolved issue is false or hallucinated command transcription after real commands and TTS. Reported outputs include `Thank you.`, `I'm going to go.`, `Okay.`, and `I don't know.` despite duration/RMS gates appearing non-trivial. The strict parser prevents most arbitrary execution, but phrases such as `Thank you.` are valid `gratitude`, so a false segment can still cause an unwanted response.

The current implementation has no explicit requirement that a VAD segment contain a KWS wake before sending it to Whisper. It also has no acoustic distinction between user speech and room/TTS audio beyond callback suppression and VAD. KWS and VAD process parallel copies of the stream, so timing around saying only `Jarvis`, saying a command immediately afterward, and TTS/reset transitions needs measured testing. Whisper latency can also make premature or stale-state behavior visible.

## Debugging Priorities

1. Instrument state transitions and timestamps before changing recognition architecture.
2. Measure VAD start/end, segment duration, queue age/overflow, KWS wake timing, suppression intervals, reset completion, Whisper duration, transcription, and parser result.
3. Determine whether false segments originate from VAD activation, pre-roll/trailing silence, queue timing, TTS leakage, or Whisper behavior.
4. Preserve strict intent validation and all suppression/reset gates.
5. Prefer reliability and intentional delay over reaction speed; keep CPU and memory bounded on the M1.
6. Change one audio behavior at a time and test silence/noise/TTS as well as successful commands.

## Known Successful Fixes

- The active command path was changed to speech-centered Silero VAD buffering with pre-roll and trailing-silence endpointing.
- Whisper command transcription is performed through the local whisper.cpp bridge because it gives better observed command transcription than the previous command-recognition behavior.
- Whisper and Silero model integrity is checked by exact size and checksum before use.
- TTS microphone suppression, settling delay, queue draining, and recognizer/VAD reset are implemented.
- KWS is kept separate from command execution and only contributes wake context/direct-call behavior.
- Bounded queues and explicit worker shutdown prevent unbounded audio retention.

## Failed Or Insufficient Approaches

Earlier total-duration, overall-RMS, minimum-active-audio, queue-draining, recognizer-reset, and TTS-suppression gates did not eliminate hallucinated Whisper results. Long segments with non-trivial RMS still produced outputs such as `Thank you.` and `Okay.`. RMS alone is therefore not evidence of genuine user speech and must not be treated as a complete solution.

## Testing Procedures

Do not launch the microphone automatically from validation. When explicitly authorized for a manual microphone test, use the existing `python3 jarvis.py` entrypoint and record timestamps and debug output. Test each category separately:

- silence, room/fan noise, keyboard noise, and mouse clicks
- JARVIS TTS responses and the post-TTS settling interval
- `Jarvis` alone
- `Jarvis, open terminal`, `Jarvis, close terminal`, and `Jarvis, open MongoDB`
- `Jarvis, turn on VPN` and `Jarvis, turn off VPN`
- `Let's get to work`, `Stay awake`, and `You can sleep now`
- gratitude phrases
- wake word plus an immediately following command
- unsupported, ambiguous, negated, and repeated phrases

Verify separately that wake-only produces only `direct_call`, wake-plus-command produces one intended command, noise/TTS produces no intent, repeated app-open commands do not relaunch an already-running app, and Ctrl+C leaves no worker/listener running. Do not treat one successful command as proof of reliability.

## Build And Validation Commands

Safe offline checks from the project root:

```bash
python3 -m py_compile jarvis.py
cmake --build native/build --config Release --parallel
```

The bridge requires the official whisper.cpp library at `whisper.cpp/build/bin/libwhisper.dylib`. Reconfigure only when needed with the existing documented CMake commands. Check model integrity without downloading:

```bash
stat -f '%z bytes' models/silero_vad.int8.onnx whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin
shasum -a 256 models/silero_vad.int8.onnx
shasum -a 1 whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin
```

Do not run `python3 jarvis.py` as a syntax/build check: it opens the microphone and speaks. No automated test suite is present in the project root.

## Safety Rules

- KWS must never directly execute arbitrary commands.
- Every command must pass the existing strict intent parser.
- Do not weaken validation to improve recall.
- Do not use transcribed text as shell input or executable input.
- Keep TTS suppression, queue draining, and recognizer reset behavior intact unless an approved change explicitly addresses them.
- Keep local-only behavior and bounded queues.
- Never launch the microphone during static validation.
- Never claim an audio fix without an actual authorized test and evidence.

## DO NOT DO THIS WITHOUT EXPLICIT USER APPROVAL

- Change, replace, or download speech recognition models.
- Replace Whisper, Nemotron, KWS, or Silero VAD.
- Change the single-stream audio architecture or introduce another recognizer/framework/dependency.
- Download random, unofficial, or unverified models.
- Weaken or bypass intent validation, wake-state handling, or safety gates.
- Remove TTS suppression, settling, queue draining, or recognizer reset.
- Change macOS security, privacy, accessibility, automation, or administrator settings.
- Use `sudo`.
- Add auto-start, services, telemetry, cloud APIs, or persistent audio storage.
- Make destructive repository or model changes.
- Implement future V4 automation features.

## Future / Planned V4 Ideas

These are not implemented and must remain out of scope until explicitly requested: opening a selected VS Code repository, choosing the correct folder/workspace, and running the project automatically as part of a more advanced developer workflow automation layer.

## Rules For Future AI Agents

Read this file, `README.md`, and the relevant source before editing. Treat `jarvis.py` as the behavioral source of truth when documentation conflicts with code, and document discrepancies rather than silently changing architecture. Inspect audio ownership, buffering, VAD/KWS timing, endpointing, Whisper input, parser gates, TTS suppression, resets, and shutdown before proposing a fix. Make minimal isolated changes, validate syntax/builds first, ask before any operation listed above, and report what was actually verified.
