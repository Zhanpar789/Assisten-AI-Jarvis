# Jarvis Core v2

A manually started, local-only macOS helper. It speaks one time-based greeting, recognizes the existing double clap, and supports allowlisted offline voice commands. It has no auto-start behavior, services, cloud speech, telemetry, or saved audio.

## Behavior

- Startup: Daniel speaks a dynamic greeting such as `Good evening, Sir. It's Saturday, August 29, 2026. All systems are ready.` The greeting finishes before microphone capture starts.
- First double clap: Mattermost is opened if needed, then the existing ClickUp URLs are sent to Chrome profile `Zhanpar` using verified profile directory `Default`.
- Later double claps: no action. The microphone remains active for voice commands.
- `open` or `launch` plus `DBeaver`, or `SQL` and `database`: opens DBeaver if it is not running.
- `open` or `launch` plus `MongoDB`, `mongo db`, `mongo database`, or `mongo compass`: opens MongoDB Compass if it is not running.
- Unsupported, incomplete, or ambiguous speech does nothing.
- Ctrl+C closes the stream, stops and joins the recognition worker, and exits.

## Architecture

There is one `sounddevice.InputStream` using the existing 44.1 kHz, 1,024-sample configuration. Its callback passes every block through the unchanged clap detector. When speech is not suppressed, the same callback copies the current mono block into a bounded 32-block queue.

The callback queue retains at most about 0.74 seconds, or roughly 128 KiB, of float audio. The worker keeps a 0.25-second pre-roll plus at most 15 seconds of a VAD-confirmed utterance in memory. If recognition falls behind, the oldest queued block is discarded. Audio is never saved, logged, uploaded, or retained after processing.

One worker thread continuously resamples the microphone blocks to 16 kHz and passes them to sherpa-onnx Silero VAD. It starts a speech-centered buffer only after VAD confirms speech, includes the bounded pre-roll, and ends the utterance after 0.45 seconds of trailing silence. Only that VAD-confirmed audio reaches the in-process whisper.cpp command recognizer and fixed intent matcher. The main thread alone speaks and launches applications. It waits on bounded queues rather than using a busy loop.

The VAD requires at least 0.15 seconds of confirmed speech. Room audio that never enters the speech state is continuously discarded except for the 0.25-second pre-roll, preventing arbitrary noise windows from reaching Whisper while retaining short commands.

## Offline recognizers

Command transcription uses:

- Official source: `https://github.com/ggml-org/whisper.cpp`
- Model: `ggml-large-v3-turbo-q5_0.bin`
- Official model repository: `https://huggingface.co/ggerganov/whisper.cpp`
- Expected size: `574041195` bytes
- Expected SHA-1: `e050f7970618a659205450ad97eb95a18d69c9ee`
- Acceleration: Apple Metal and Accelerate, enabled by the official build

Speech detection uses:

- Official source: `https://github.com/k2-fsa/sherpa-onnx`
- Model: `silero_vad.int8.onnx`
- Official release URL: `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.int8.onnx`
- Expected size: `212860` bytes
- Expected SHA-256: `c36d490aff5ab924ca6c7aeec4d8f6bd3d22db6fa17611b9c5b17eae58ac3a20`

The existing sherpa-onnx Nemotron model and code remain intact. Set `COMMAND_ASR = "nemotron"` in `jarvis.py` to roll endpoint detection and command transcription back without changing model files. The dedicated Jarvis KWS remains unchanged and can confirm a standalone wake even if VAD does not produce an utterance.

The package and model are needed only from local files at runtime. `jarvis.py` contains no downloader and makes no network requests.

## Manual installation

Nothing in this project installs or downloads automatically. From the project directory, create or activate a virtual environment and install the dependencies manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Clone and build whisper.cpp from the official source repository:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON -DGGML_ACCELERATE=ON
cmake --build whisper.cpp/build --config Release --parallel
sh whisper.cpp/models/download-ggml-model.sh large-v3-turbo-q5_0 whisper.cpp/models
```

Verify the model before loading it. Stop if either value differs:

```bash
stat -f '%z bytes' whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin
shasum -a 1 whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin
```

Download Silero VAD only from its official sherpa-onnx release and verify it:

```bash
curl --fail --location --output models/silero_vad.int8.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.int8.onnx
stat -f '%z bytes' models/silero_vad.int8.onnx
shasum -a 256 models/silero_vad.int8.onnx
```

Build the local bridge against the official library:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release
```

Jarvis verifies the Whisper size and SHA-1 and the Silero VAD size and SHA-256 again before loading either model or opening the microphone.

## Manual start

With the virtual environment active:

```bash
python3 jarvis.py
```

The Terminal application should request Microphone permission on first use. No Speech Recognition, Accessibility, Automation, administrator, Camera, Contacts, Photos, or Keychain permission is required. If macOS unexpectedly requests another permission for Jarvis, deny it and stop the program.

## Voice-command safety

Recognized text is lowercased, stripped of punctuation, and split into words. Apostrophes are removed first so a phrase such as `don't open` remains recognizable as a negation. `Jarvis`, `please`, and `the` are removed as optional filler words.

After optional filler is removed, every supported command must start with the exact word `open` or `launch` and contain only words relevant to one allowlisted target. Negations such as `not`, `never`, `don't`, `can't`, and `no` reject the entire utterance. `Opening`, `run`, `execute`, `install`, unrelated context, and other verbs do not qualify. A lone `compass` does not qualify. If one utterance matches both database applications, it is rejected.

Recognized text never becomes a shell command, subprocess argument, path, application name, or executable input. The only returned intents are internal constants for DBeaver and MongoDB Compass. All application and process names are fixed source-code constants.

Verified installed names:

```python
DBEAVER_APP_NAME = "DBeaver"
DBEAVER_PROCESS_NAME = "dbeaver"
MONGODB_APP_NAME = "MongoDB Compass"
MONGODB_PROCESS_NAME = "MongoDB Compass"
```

`pgrep -x` checks the appropriate fixed process name only after a supported intent is recognized. If the application is running, Jarvis neither speaks nor launches it.

## Acknowledgements and self-trigger prevention

`VOICE_CONFIRMATIONS` controls acknowledgements. When enabled and an application needs to open, Python's standard `random` module chooses either a short acknowledgement or an acknowledgement containing the fixed application name. The editable pools are `SHORT_ACKNOWLEDGEMENTS` and `ACTION_ACKNOWLEDGEMENTS` in `jarvis.py`.

Before Daniel speaks, Jarvis suppresses clap evaluation and forwarding microphone blocks to the recognizer, then clears queued audio. After `say` finishes, it waits `SPEAKER_SETTLING_SECONDS`, clears the queue again, requests a recognizer reset, and then resumes processing. The microphone stream remains open, but Daniel's response is not transcribed or interpreted as a clap.

## Existing work-app behavior

The existing Chrome mapping remains unchanged:

```python
CHROME_PROFILE_NAME = "Zhanpar"
CHROME_PROFILE_DIRECTORY = "Default"
```

Chrome receives one first-clap launch request equivalent to:

```bash
open -a "Google Chrome" --args --profile-directory=Default URL1 URL2 URL3
```

Version 2 does not inspect Chrome tabs or use AppleScript. `actions_launched` prevents another work-app request in the same Jarvis process. Mattermost is checked once at that action with `pgrep -x Mattermost` and opened only if absent.

## Safe manual test plan

1. Start with `python3 jarvis.py`. Confirm Daniel speaks exactly once with the correct local salutation and date, then the Terminal microphone indicator appears.
2. Double clap once. Confirm Mattermost opens only if closed and the three existing ClickUp tabs open under Chrome profile `Zhanpar`.
3. Double clap again. Confirm no work applications or tabs reopen and microphone listening continues.
4. Close DBeaver, say `Jarvis, open the SQL database`, and confirm one randomly selected Daniel acknowledgement followed by one DBeaver launch.
5. With DBeaver open, repeat the command. Confirm there is no acknowledgement and no relaunch.
6. Close MongoDB Compass, say `Please open MongoDB`, and confirm one acknowledgement followed by one MongoDB Compass launch.
7. With MongoDB Compass open, repeat the command. Confirm there is no acknowledgement and no relaunch.
8. Close and reopen either app between tests, then repeat commands to hear different random acknowledgement styles. Random selection does not guarantee a different response every time.
9. Say unsupported phrases such as `run DBeaver`, `opening MongoDB`, `open Compass`, or `delete this folder`. Confirm nothing happens.
10. Press Ctrl+C. Confirm the process exits and the Terminal microphone indicator disappears. No Jarvis worker or listener should remain.

## Security and resource review

- All subprocesses use fixed argument lists with `shell=False`, which is the default. Speech is never inserted into a subprocess call. Interrupted or timed-out helper commands are terminated and reaped.
- There is no sudo, root access, arbitrary execution, LaunchAgent, LaunchDaemon, login item, auto-start, AppleScript, browser automation, API, telemetry, analytics, or runtime network request.
- Audio exists only in Core Audio callback buffers and the bounded in-memory queue. It is never written to disk or transmitted.
- One microphone stream serves clap and speech processing. One worker owns the recognizer, avoiding concurrent recognizer access.
- Existing clap thresholds and timing remain unchanged. After the first clap action, the detector returns immediately while voice forwarding continues.
- No process is polled continuously. Application state is checked only after the first clap action or a supported voice intent.
- Ctrl+C first leaves the stream context, then suppresses new speech data, signals the worker, discards queued audio, adds a stop marker, and joins the worker. A bounded join reports a stuck native decoder rather than silently continuing; the daemon fallback ensures top-level process exit releases native resources.
- The worker and audio callback report unrecoverable errors to the main thread and request shutdown. Native recognizer resources are released when the worker and process exit.
