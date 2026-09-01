#!/usr/bin/env python3
"""Manual, local-only clap and voice launcher for macOS."""

from __future__ import annotations

import ctypes
import hashlib
import math
import queue
import random
import re
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd


# Application and URL configuration.
MATTERMOST_APP_NAME = "Mattermost"
CHROME_APP_NAME = "Google Chrome"
CHROME_PROFILE_NAME = "Zhanpar"
CHROME_PROFILE_DIRECTORY = "Default"
DBEAVER_APP_NAME = "DBeaver"
DBEAVER_PROCESS_NAME = "dbeaver"
MONGODB_APP_NAME = "MongoDB Compass"
MONGODB_PROCESS_NAME = "MongoDB Compass"
POSTMAN_APP_NAME = "Postman"
POSTMAN_PROCESS_NAME = "Postman"
DOCKER_APP_NAME = "Docker"
DOCKER_PROCESS_NAME = "Docker Desktop"
TERMINAL_APP_NAME = "Terminal"
TERMINAL_PROCESS_NAME = "Terminal"
STICKIES_APP_NAME = "Stickies"
STICKIES_PROCESS_NAME = "Stickies"
WIREGUARD_TUNNEL_NAME = "zhanfarisman"
CLICKUP_URLS = [
    "https://app.clickup.com/25639638/inbox?tab=primary",
    "https://app.clickup.com/25639638/v/l/reepp-18916",
    "https://app.clickup.com/25639638/hubs/dashboards",
]
YOUTUBE_URL = "https://www.youtube.com/"

# Local speech configuration.
DEBUG = True
VOICE_NAME = "Daniel"
VOICE_CONFIRMATIONS = True
SPEAKER_SETTLING_SECONDS = 0.35
SAY_TIMEOUT_SECONDS = 30.0
APP_COMMAND_TIMEOUT_SECONDS = 10.0
COMMAND_ASR = "whisper"
MODEL_DIRECTORY = (
    Path(__file__).resolve().parent
    / "models"
    / "sherpa-onnx-nemotron-speech-streaming-en-0.6b-1120ms-int8-2026-04-25"
)
MODEL_TOKENS = MODEL_DIRECTORY / "tokens.txt"
MODEL_ENCODER = MODEL_DIRECTORY / "encoder.int8.onnx"
MODEL_DECODER = MODEL_DIRECTORY / "decoder.int8.onnx"
MODEL_JOINER = MODEL_DIRECTORY / "joiner.int8.onnx"
WHISPER_DIRECTORY = Path(__file__).resolve().parent / "whisper.cpp"
WHISPER_MODEL = WHISPER_DIRECTORY / "models" / "ggml-large-v3-turbo-q5_0.bin"
WHISPER_MODEL_SIZE = 574_041_195
WHISPER_MODEL_SHA1 = "e050f7970618a659205450ad97eb95a18d69c9ee"
WHISPER_BRIDGE = (
    Path(__file__).resolve().parent
    / "native"
    / "build"
    / "libjarvis_whisper.dylib"
)
WHISPER_SAMPLE_RATE = 16_000
WHISPER_NUM_THREADS = 4
WHISPER_OUTPUT_BYTES = 4_096
VAD_MODEL = Path(__file__).resolve().parent / "models" / "silero_vad.int8.onnx"
VAD_MODEL_SIZE = 212_860
VAD_MODEL_SHA256 = "c36d490aff5ab924ca6c7aeec4d8f6bd3d22db6fa17611b9c5b17eae58ac3a20"
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_SECONDS = 0.15
VAD_MIN_SILENCE_SECONDS = 0.45
VAD_PRE_ROLL_SECONDS = 0.25
VAD_MAX_SPEECH_SECONDS = 15.0
KWS_STANDALONE_TIMEOUT_SECONDS = 1.5
KWS_DIRECTORY = (
    Path(__file__).resolve().parent
    / "models"
    / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
)
KWS_TOKENS = KWS_DIRECTORY / "tokens.txt"
KWS_ENCODER = (
    KWS_DIRECTORY / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
)
KWS_DECODER = (
    KWS_DIRECTORY / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
)
KWS_JOINER = (
    KWS_DIRECTORY / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
)
KWS_KEYWORDS = KWS_DIRECTORY / "keywords_jarvis.txt"
KWS_KEYWORDS_SCORE = 1.0
KWS_KEYWORDS_THRESHOLD = 0.25
KWS_NUM_THREADS = 1

SHORT_ACKNOWLEDGEMENTS = [
    "Okay, Sir.",
    "Roger, Sir.",
    "On your way, Sir.",
    "Right away, Sir.",
    "Understood, Sir.",
    "On it, Sir.",
    "At once, Sir.",
    "Got it, Sir.",
]
ACTION_ACKNOWLEDGEMENTS = [
    "Certainly, Sir.",
    "Of course, Sir.",
    "Very well, Sir.",
    "Consider it done, Sir.",
    "Right away, Sir.",
]
DBEAVER_CLOSE_CONFIRMATIONS = [
    "Database closed, Sir.",
    "Database is offline, Sir.",
    "Database session closed, Sir.",
    "Closed and secured, Sir.",
]
MONGODB_CLOSE_CONFIRMATIONS = [
    "MongoDB closed, Sir.",
    "MongoDB is offline, Sir.",
    "Compass has been closed, Sir.",
    "Closed and secured, Sir.",
]
POSTMAN_CLOSE_CONFIRMATIONS = [
    "Postman closed, Sir.",
    "API workspace closed, Sir.",
    "Postman is offline, Sir.",
]
DOCKER_CLOSE_CONFIRMATIONS = [
    "Docker has been shut down, Sir.",
    "Container environment is offline, Sir.",
    "Docker is offline, Sir.",
]
TERMINAL_CLOSE_CONFIRMATIONS = [
    "Terminal closed, Sir.",
    "Command line session closed, Sir.",
    "Terminal is offline, Sir.",
]
STICKIES_CLOSE_CONFIRMATIONS = [
    "Notes closed, Sir.",
    "Notepad closed, Sir.",
    "Your notes have been put away, Sir.",
]
POSTMAN_OPEN_CONFIRMATIONS = [
    "Postman is ready, Sir.",
    "API workspace is ready, Sir.",
    "Your API tools are ready, Sir.",
    "Postman is standing by, Sir.",
]
DOCKER_OPEN_CONFIRMATIONS = [
    "Docker is starting up, Sir.",
    "Container environment is coming online, Sir.",
    "Docker environment is ready, Sir.",
    "Containers are at your disposal, Sir.",
]
TERMINAL_OPEN_CONFIRMATIONS = [
    "Terminal is ready, Sir.",
    "Command line is ready, Sir.",
    "Terminal is at your disposal, Sir.",
    "Command interface ready, Sir.",
]
STICKIES_OPEN_CONFIRMATIONS = [
    "Your notes are ready, Sir.",
    "Notepad is ready, Sir.",
    "Your workspace for notes is ready, Sir.",
    "Ready when you are, Sir. Your notes are open.",
]
GRATITUDE_RESPONSES = [
    "At your service, Sir.",
    "Always a pleasure, Sir.",
    "You're most welcome, Sir.",
    "Anytime, Sir.",
    "My pleasure, Sir.",
    "Always, Sir.",
]
VPN_ON_RESPONSES = [
    "Secure tunnel established, Sir.",
    "VPN is now active, Sir.",
    "Secure connection established, Sir.",
]
VPN_OFF_RESPONSES = [
    "VPN connection terminated, Sir.",
    "Secure tunnel closed, Sir.",
    "VPN is now offline, Sir.",
]
STAY_AWAKE_RESPONSES = [
    "I'll keep the system awake, Sir.",
    "Sleep mode suspended, Sir.",
    "The system will remain awake, Sir.",
    "I'll make sure the Mac stays awake, Sir.",
    "No sleeping on my watch, Sir.",
]
SLEEP_NORMALLY_RESPONSES = [
    "Normal sleep behavior restored, Sir.",
    "Sleep mode restored, Sir.",
    "The system may sleep normally now, Sir.",
    "Normal power management restored, Sir.",
]
SPECIAL_SLEEP_RESPONSE = "Thank you, Sir. I'll get some rest for a while."
DIRECT_CALL_RESPONSES = [
    "Yes, Sir?",
    "I'm here, Sir.",
    "I'm listening, Sir.",
    "Yes, boss?",
]
STARTUP_PROMPTS = [
    "How can I help you, Sir?",
    "What can I do for you today, Sir?",
    "I'm ready when you are, Sir.",
    "Standing by for your instructions, Sir.",
    "What shall we work on today, Sir?",
    "I'm at your service, Sir.",
    "Ready when you are, Sir.",
]
WORK_MODE_RESPONSES = [
    "Work mode initiated, Sir.",
    "Your workspace is ready, Sir.",
    "Everything is ready for work, Sir.",
    "Work environment is ready, Sir.",
    "We're ready to begin, Sir.",
]
BREAK_TIME_RESPONSES = [
    "Alright, {address}. Break time.",
    "Good idea, {address}. Time to take a break.",
    "Of course, {address}. Take a well-deserved break.",
    "Alright, {address}. Let's take a little break.",
    "You got it, {address}. Time to rest.",
    "Yes, {address}. Break time. You've earned it.",
]

# Audio and clap-detector configuration. Existing clap values are unchanged.
SAMPLE_RATE = 44_100
BLOCK_SIZE = 1_024
CLAP_THRESHOLD = 0.35
MAX_CLAP_RMS = 0.20
MIN_CREST_FACTOR = 2.5
MIN_CLAP_INTERVAL = 0.20
MAX_CLAP_INTERVAL = 0.80
DEBOUNCE_SECONDS = 0.12
COOLDOWN_SECONDS = 2.0
REARM_BLOCKS = 3
AUDIO_QUEUE_BLOCKS = 32
KWS_AUDIO_QUEUE_BLOCKS = 32

INTENT_DBEAVER = "dbeaver"
INTENT_MONGODB = "mongodb"
INTENT_CLOSE_DBEAVER = "close_dbeaver"
INTENT_CLOSE_MONGODB = "close_mongodb"
INTENT_POSTMAN = "postman"
INTENT_CLOSE_POSTMAN = "close_postman"
INTENT_DOCKER = "docker"
INTENT_CLOSE_DOCKER = "close_docker"
INTENT_TERMINAL = "terminal"
INTENT_CLOSE_TERMINAL = "close_terminal"
INTENT_STICKIES = "stickies"
INTENT_CLOSE_STICKIES = "close_stickies"
INTENT_GRATITUDE = "gratitude"
INTENT_VPN_ON = "vpn_on"
INTENT_VPN_OFF = "vpn_off"
INTENT_STAY_AWAKE = "stay_awake"
INTENT_SLEEP_NORMALLY = "sleep_normally"
INTENT_SPECIAL_SLEEP = "special_sleep"
INTENT_DIRECT_CALL = "direct_call"
INTENT_WORK_MODE = "work_mode"
INTENT_BREAK_TIME = "break_time"
DIRECT_CALL_WAKE_NAMES = {
    "jarvis",
    "jarvish",
    "javis",
    "javish",
    "jarfish",
    "garfish",
    "tarfish",
    "darvis",
}
STOP_AUDIO = object()


def debug(message: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {message}", flush=True)


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WhisperCommandRecognizer:
    """In-process command transcription through the official whisper.cpp C API."""

    def __init__(self) -> None:
        missing_files = [
            str(path) for path in (WHISPER_MODEL, WHISPER_BRIDGE) if not path.is_file()
        ]
        if missing_files:
            missing = "\n".join(f"  {path}" for path in missing_files)
            raise FileNotFoundError(f"Missing local whisper.cpp files:\n{missing}")
        if WHISPER_MODEL.stat().st_size != WHISPER_MODEL_SIZE:
            raise RuntimeError("The whisper.cpp model size does not match the official model")
        if sha1_file(WHISPER_MODEL) != WHISPER_MODEL_SHA1:
            raise RuntimeError("The whisper.cpp model checksum does not match; refusing to load it")

        self.library = ctypes.CDLL(str(WHISPER_BRIDGE))
        self.library.jarvis_whisper_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.library.jarvis_whisper_create.restype = ctypes.c_void_p
        self.library.jarvis_whisper_destroy.argtypes = [ctypes.c_void_p]
        self.library.jarvis_whisper_destroy.restype = None
        self.library.jarvis_whisper_transcribe.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.library.jarvis_whisper_transcribe.restype = ctypes.c_int

        self.context = self.library.jarvis_whisper_create(
            str(WHISPER_MODEL).encode(), WHISPER_NUM_THREADS
        )
        if not self.context:
            raise RuntimeError("Unable to initialize the verified whisper.cpp model")

    def close(self) -> None:
        if self.context:
            self.library.jarvis_whisper_destroy(self.context)
            self.context = None

    def transcribe(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        if not self.context:
            raise RuntimeError("The whisper.cpp recognizer is closed")
        if samples.size == 0:
            return ""

        if sample_rate == WHISPER_SAMPLE_RATE:
            resampled = np.ascontiguousarray(samples, dtype=np.float32)
        else:
            output_count = max(
                1, round(samples.size * WHISPER_SAMPLE_RATE / sample_rate)
            )
            source_positions = np.arange(samples.size, dtype=np.float64)
            target_positions = np.arange(output_count, dtype=np.float64) * (
                sample_rate / WHISPER_SAMPLE_RATE
            )
            resampled = np.ascontiguousarray(
                np.interp(target_positions, source_positions, samples), dtype=np.float32
            )
        output = ctypes.create_string_buffer(WHISPER_OUTPUT_BYTES)
        result = self.library.jarvis_whisper_transcribe(
            self.context,
            resampled.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            resampled.size,
            output,
            len(output),
        )
        if result != 0:
            raise RuntimeError(f"whisper.cpp transcription failed with code {result}")
        return output.value.decode("utf-8").strip()


class StreamingLinearResampler:
    """Continuously resample blocks without duplicating their boundaries."""

    def __init__(self, input_rate: int, output_rate: int) -> None:
        self.step = input_rate / output_rate
        self.buffer = np.empty(0, dtype=np.float32)
        self.next_position = 0.0

    def reset(self) -> None:
        self.buffer = np.empty(0, dtype=np.float32)
        self.next_position = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        self.buffer = np.concatenate((self.buffer, samples))
        if self.buffer.size < 2 or self.next_position >= self.buffer.size - 1:
            return np.empty(0, dtype=np.float32)

        positions = np.arange(
            self.next_position, self.buffer.size - 1, self.step, dtype=np.float64
        )
        output = np.ascontiguousarray(
            np.interp(
                positions,
                np.arange(self.buffer.size, dtype=np.float64),
                self.buffer,
            ),
            dtype=np.float32,
        )
        next_position = positions[-1] + self.step
        consumed = min(int(next_position), self.buffer.size - 1)
        self.buffer = self.buffer[consumed:]
        self.next_position = next_position - consumed
        return output


def greeting_for_current_time() -> str:
    now = datetime.now()
    if 5 <= now.hour < 12:
        salutation = "Good morning, Sir."
    elif 12 <= now.hour < 18:
        salutation = "Good afternoon, Sir."
    else:
        salutation = "Good evening, Sir."

    date_text = f"{now:%A, %B} {now.day}, {now.year}"
    return (
        f"{salutation} It's {date_text}. All systems are ready. "
        f"{random.choice(STARTUP_PROMPTS)}"
    )


def run_command(arguments: list[str], timeout: float, *, quiet: bool = False) -> int:
    output = subprocess.DEVNULL if quiet else None
    process = subprocess.Popen(
        arguments,
        stdout=output,
        stderr=output,
    )
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def vpn_status() -> str | None:
    try:
        result = subprocess.run(
            ["/usr/sbin/scutil", "--nc", "status", WIREGUARD_TUNNEL_NAME],
            capture_output=True,
            text=True,
            timeout=APP_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.partition("\n")[0].strip()


def change_vpn_state(*, enabled: bool) -> bool:
    target_status = "Connected" if enabled else "Disconnected"
    current_status = "Disconnected" if enabled else "Connected"
    status = vpn_status()
    if status == target_status:
        return False
    if status != current_status:
        return False

    action = "start" if enabled else "stop"
    if (
        run_command(
            ["/usr/sbin/scutil", "--nc", action, WIREGUARD_TUNNEL_NAME],
            APP_COMMAND_TIMEOUT_SECONDS,
        )
        != 0
    ):
        return False

    deadline = time.monotonic() + APP_COMMAND_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if vpn_status() == target_status:
            return True
        time.sleep(0.2)
    return False


def start_stay_awake(
    process: subprocess.Popen | None,
) -> tuple[subprocess.Popen | None, bool]:
    if process is not None and process.poll() is None:
        return process, False
    try:
        process = subprocess.Popen(
            ["/usr/bin/caffeinate", "-i"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None, False
    time.sleep(0.1)
    if process.poll() is not None:
        return None, False
    return process, True


def stop_stay_awake(
    process: subprocess.Popen | None,
) -> tuple[subprocess.Popen | None, bool]:
    if process is None or process.poll() is not None:
        return None, False
    try:
        process.terminate()
        process.wait(timeout=1.0)
    except ProcessLookupError:
        return None, False
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    return None, process.poll() is not None


def say(text: str) -> None:
    run_command(
        ["say", "-v", VOICE_NAME, text],
        SAY_TIMEOUT_SECONDS,
    )


def is_app_running(process_name: str) -> bool:
    return_code = run_command(
        ["pgrep", "-x", process_name],
        APP_COMMAND_TIMEOUT_SECONDS,
        quiet=True,
    )
    return return_code == 0


def launch_work_apps() -> bool:
    debug("launch_work_apps() called")
    mattermost_ready = True
    if not is_app_running(MATTERMOST_APP_NAME):
        mattermost_ready = (
            run_command(
                ["open", "-a", MATTERMOST_APP_NAME],
                APP_COMMAND_TIMEOUT_SECONDS,
            )
            == 0
        )

    chrome_arguments = [
        "open",
        "-a",
        CHROME_APP_NAME,
        "--args",
        f"--profile-directory={CHROME_PROFILE_DIRECTORY}",
        *CLICKUP_URLS,
    ]
    chrome_ready = run_command(chrome_arguments, APP_COMMAND_TIMEOUT_SECONDS) == 0
    return mattermost_ready and chrome_ready


def normalize_command(text: str) -> list[str]:
    normalized = text.lower().replace("'", "").replace("\u2019", "")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return normalized.split()


def normalize_voice_transcription(text: str) -> str:
    substitutions = (
        (r"\bsequel\b", "sql"),
        (r"\bdatabas\b", "database"),
        (r"\bdatabys\b", "database"),
        (r"\bdatabish\b", "database"),
        (r"\bletterbase\b", "database"),
        (r"\bat\s+abyss\b", "database"),
        (r"\bd\s+beaver\b", "dbeaver"),
        (r"\bmongo\s+d\s+b\b", "mongodb"),
        (r"\bmongo\s+dbi\b", "mongodb"),
        (r"\bmongo\s+di\s+b\b", "mongodb"),
        (r"\bmongodibi\b", "mongodb"),
        (r"\bmongodbi\b", "mongodb"),
        (r"\bmodibi\b", "mongodb"),
        (r"\bmongod\s+db\b", "mongodb"),
        (r"\bmungo\s+db\b", "mongodb"),
        (r"\bmomo\s+db\b", "mongodb"),
    )
    normalized = text
    for pattern, replacement in substitutions:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def command_intent(text: str) -> str | None:
    words = normalize_command(text)
    is_direct_call = (
        (len(words) == 1 and words[0] in DIRECT_CALL_WAKE_NAMES)
        or (
            len(words) == 2
            and words[0] in {"hey", "hello"}
            and words[1] in DIRECT_CALL_WAKE_NAMES
        )
        or tuple(words) == ("jarvis", "please")
    )
    if is_direct_call:
        return INTENT_DIRECT_CALL

    work_mode_words = tuple(
        word for word in words if word not in {"jarvis", "please"}
    )
    if work_mode_words in {
        ("ready", "to", "work"),
        ("lets", "get", "to", "work"),
        ("start", "work", "mode"),
        ("lets", "start", "work"),
        ("lets", "go", "to", "work"),
    }:
        return INTENT_WORK_MODE

    break_time_words = tuple(
        word for word in words if word not in {"jarvis", "please", "the"}
    )
    if break_time_words in {
        ("break", "time"),
        ("its", "break", "time"),
        ("its", "time", "for", "a", "break"),
        ("lets", "get", "a", "break"),
        ("lets", "take", "a", "break"),
        ("take", "a", "break"),
        ("take", "a", "rest"),
        ("rest", "time"),
        ("its", "rest", "time"),
        ("time", "to", "rest"),
        ("lets", "rest"),
        ("i", "need", "a", "break"),
        ("i", "need", "to", "take", "a", "break"),
        ("i", "need", "a", "rest"),
        ("lets", "have", "a", "break"),
        ("we", "need", "a", "break"),
        ("lets", "take", "some", "rest"),
        ("i", "think", "its", "time", "to", "rest"),
        ("its", "time", "to", "take", "a", "break"),
    }:
        return INTENT_BREAK_TIME

    gratitude_words = words
    if gratitude_words and gratitude_words[0] == "jarvis":
        gratitude_words = gratitude_words[1:]
    elif gratitude_words and gratitude_words[-1] == "jarvis":
        gratitude_words = gratitude_words[:-1]
    if tuple(gratitude_words) in {
        ("thanks",),
        ("thank", "you"),
        ("thank", "you", "very", "much"),
    }:
        return INTENT_GRATITUDE

    vpn_words = [word for word in words if word not in {"jarvis", "please", "the"}]
    if vpn_words == ["turn", "on", "vpn"]:
        return INTENT_VPN_ON
    if vpn_words == ["turn", "off", "vpn"]:
        return INTENT_VPN_OFF

    stay_awake_words = tuple(
        word for word in words if word not in {"jarvis", "please", "the"}
    )
    if stay_awake_words in {
        ("stay", "awake"),
        ("keep", "mac", "awake"),
        ("keep", "my", "mac", "awake"),
        ("keep", "system", "awake"),
        ("dont", "let", "mac", "sleep"),
    }:
        return INTENT_STAY_AWAKE
    if stay_awake_words == ("sleep", "normally"):
        return INTENT_SLEEP_NORMALLY
    if stay_awake_words == ("you", "can", "sleep", "now"):
        return INTENT_SPECIAL_SLEEP

    if {"not", "never", "dont", "cannot", "cant", "no"} & set(words):
        return None

    command_words = [word for word in words if word not in {"jarvis", "please", "the"}]
    if not command_words or command_words[0] not in {"open", "launch", "close"}:
        return None

    action = command_words[0]
    app_commands = {
        ("postman",): (INTENT_POSTMAN, INTENT_CLOSE_POSTMAN, True),
        ("docker",): (INTENT_DOCKER, INTENT_CLOSE_DOCKER, True),
        ("terminal",): (INTENT_TERMINAL, INTENT_CLOSE_TERMINAL, True),
        ("stickies",): (INTENT_STICKIES, INTENT_CLOSE_STICKIES, False),
        ("sticky", "notes"): (INTENT_STICKIES, INTENT_CLOSE_STICKIES, False),
        ("sticky", "note"): (INTENT_STICKIES, INTENT_CLOSE_STICKIES, False),
        ("notepad",): (INTENT_STICKIES, INTENT_CLOSE_STICKIES, False),
    }
    app_command = app_commands.get(tuple(command_words[1:]))
    if app_command is not None:
        open_intent, close_intent, allow_launch = app_command
        if action == "close":
            return close_intent
        if action == "open" or allow_launch:
            return open_intent
        return None

    word_set = set(command_words)
    dbeaver_words = {"open", "launch", "close", "dbeaver", "sql", "database"}
    mongo_words = {
        "open",
        "launch",
        "close",
        "mongodb",
        "mongo",
        "db",
        "database",
        "compass",
    }

    dbeaver_match = word_set <= dbeaver_words and bool(
        {"dbeaver", "database"} & word_set
    )
    mongo_match = (
        word_set <= mongo_words
        and (
            "mongodb" in word_set
            or "mongo" in word_set
        )
    )

    if dbeaver_match == mongo_match:
        return None
    if dbeaver_match:
        return INTENT_CLOSE_DBEAVER if action == "close" else INTENT_DBEAVER
    return INTENT_CLOSE_MONGODB if action == "close" else INTENT_MONGODB


def command_intent_with_kws_context(text: str) -> str | None:
    intent = command_intent(text)
    if intent is not None:
        return intent

    words = normalize_command(text)
    if words and words[0] in DIRECT_CALL_WAKE_NAMES:
        words = words[1:]
    elif (
        len(words) >= 2
        and words[0] in {"hey", "hello"}
        and words[1] in DIRECT_CALL_WAKE_NAMES
    ):
        words = words[2:]
    else:
        return None

    return command_intent(" ".join(words)) if words else INTENT_DIRECT_CALL


def acknowledgement_for(app_name: str) -> str:
    if random.choice((True, False)):
        return random.choice(SHORT_ACKNOWLEDGEMENTS)
    acknowledgement = random.choice(ACTION_ACKNOWLEDGEMENTS)
    return f"{acknowledgement} Opening {app_name}."


def drain_audio_queue(audio_queue: queue.Queue[object]) -> None:
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            return


def enqueue_latest_audio(
    audio_queue: queue.Queue[object], samples: np.ndarray
) -> None:
    try:
        audio_queue.put_nowait(samples)
    except queue.Full:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            audio_queue.put_nowait(samples)
        except queue.Full:
            pass


def enqueue_intent(
    intent_queue: queue.Queue[tuple[str, str]], intent: str, source: str
) -> None:
    try:
        intent_queue.put_nowait((intent, source))
    except queue.Full:
        try:
            intent_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            intent_queue.put_nowait((intent, source))
        except queue.Full:
            pass


def append_pre_roll(
    chunks: deque[np.ndarray], samples: np.ndarray, max_samples: int
) -> None:
    chunks.append(samples)
    total = sum(chunk.size for chunk in chunks)
    while chunks and total - chunks[0].size >= max_samples:
        total -= chunks.popleft().size
    if chunks and total > max_samples:
        trim = total - max_samples
        chunks[0] = chunks[0][trim:]


def process_transcription(
    text: str,
    intent_queue: queue.Queue[tuple[str, str]],
    kws_wake_pending: threading.Event,
) -> None:
    normalized_text = normalize_voice_transcription(text)
    if normalized_text != text:
        debug(f'normalized transcription: "{normalized_text}"')
    source = "asr"
    if kws_wake_pending.is_set():
        debug("ASR utterance completed while KWS pending")
        intent = command_intent_with_kws_context(normalized_text)
        if intent is not None and intent != INTENT_DIRECT_CALL:
            debug("KWS pending wake converted to command")
            kws_wake_pending.clear()
            debug("KWS pending wake cancelled")
        else:
            intent = INTENT_DIRECT_CALL
            source = "kws"
            debug("KWS standalone wake confirmed")
    else:
        intent = command_intent(normalized_text)
    if intent is not None:
        debug(f"supported intent detected: {intent}")
        enqueue_intent(intent_queue, intent, source)
    else:
        debug("voice command rejected")


def speak_while_suppressed(
    text: str,
    audio_queue: queue.Queue[object],
    speech_suppressed: threading.Event,
    recognizer_reset_requested: threading.Event,
    recognizer_reset_complete: threading.Event,
) -> None:
    speech_suppressed.set()
    drain_audio_queue(audio_queue)
    try:
        say(text)
        time.sleep(SPEAKER_SETTLING_SECONDS)
        drain_audio_queue(audio_queue)
        recognizer_reset_complete.clear()
        recognizer_reset_requested.set()
        if not recognizer_reset_complete.wait(timeout=2.0):
            raise RuntimeError("The local speech recognizer did not reset after speaking")
        drain_audio_queue(audio_queue)
    finally:
        speech_suppressed.clear()


def respond_to_direct_call(
    source: str,
    audio_queue: queue.Queue[object],
    speech_suppressed: threading.Event,
    recognizer_reset_requested: threading.Event,
    recognizer_reset_complete: threading.Event,
) -> None:
    debug(f"direct_call source: {source}")
    speak_while_suppressed(
        random.choice(DIRECT_CALL_RESPONSES),
        audio_queue,
        speech_suppressed,
        recognizer_reset_requested,
        recognizer_reset_complete,
    )
    debug("direct_call response spoken")


def launch_voice_app(
    intent: str,
    audio_queue: queue.Queue[object],
    speech_suppressed: threading.Event,
    recognizer_reset_requested: threading.Event,
    recognizer_reset_complete: threading.Event,
) -> None:
    if intent in {INTENT_DBEAVER, INTENT_CLOSE_DBEAVER}:
        app_name = DBEAVER_APP_NAME
        process_name = DBEAVER_PROCESS_NAME
        open_confirmations = None
        close_confirmations = DBEAVER_CLOSE_CONFIRMATIONS
    elif intent in {INTENT_MONGODB, INTENT_CLOSE_MONGODB}:
        app_name = MONGODB_APP_NAME
        process_name = MONGODB_PROCESS_NAME
        open_confirmations = None
        close_confirmations = MONGODB_CLOSE_CONFIRMATIONS
    elif intent in {INTENT_POSTMAN, INTENT_CLOSE_POSTMAN}:
        app_name = POSTMAN_APP_NAME
        process_name = POSTMAN_PROCESS_NAME
        open_confirmations = POSTMAN_OPEN_CONFIRMATIONS
        close_confirmations = POSTMAN_CLOSE_CONFIRMATIONS
    elif intent in {INTENT_DOCKER, INTENT_CLOSE_DOCKER}:
        app_name = DOCKER_APP_NAME
        process_name = DOCKER_PROCESS_NAME
        open_confirmations = DOCKER_OPEN_CONFIRMATIONS
        close_confirmations = DOCKER_CLOSE_CONFIRMATIONS
    elif intent in {INTENT_TERMINAL, INTENT_CLOSE_TERMINAL}:
        app_name = TERMINAL_APP_NAME
        process_name = TERMINAL_PROCESS_NAME
        open_confirmations = TERMINAL_OPEN_CONFIRMATIONS
        close_confirmations = TERMINAL_CLOSE_CONFIRMATIONS
    elif intent in {INTENT_STICKIES, INTENT_CLOSE_STICKIES}:
        app_name = STICKIES_APP_NAME
        process_name = STICKIES_PROCESS_NAME
        open_confirmations = STICKIES_OPEN_CONFIRMATIONS
        close_confirmations = STICKIES_CLOSE_CONFIRMATIONS
    else:
        return

    if intent in {
        INTENT_CLOSE_DBEAVER,
        INTENT_CLOSE_MONGODB,
        INTENT_CLOSE_POSTMAN,
        INTENT_CLOSE_DOCKER,
        INTENT_CLOSE_TERMINAL,
        INTENT_CLOSE_STICKIES,
    }:
        if is_app_running(process_name):
            return_code = run_command(
                ["osascript", "-e", f'tell application "{app_name}" to quit'],
                APP_COMMAND_TIMEOUT_SECONDS,
            )
            if return_code == 0 and VOICE_CONFIRMATIONS:
                speak_while_suppressed(
                    random.choice(close_confirmations),
                    audio_queue,
                    speech_suppressed,
                    recognizer_reset_requested,
                    recognizer_reset_complete,
                )
        return

    if is_app_running(process_name):
        return

    if open_confirmations is not None:
        return_code = run_command(
            ["open", "-a", app_name],
            APP_COMMAND_TIMEOUT_SECONDS,
        )
        if return_code == 0 and VOICE_CONFIRMATIONS:
            speak_while_suppressed(
                random.choice(open_confirmations),
                audio_queue,
                speech_suppressed,
                recognizer_reset_requested,
                recognizer_reset_complete,
            )
        return

    if VOICE_CONFIRMATIONS:
        speak_while_suppressed(
            acknowledgement_for(app_name),
            audio_queue,
            speech_suppressed,
            recognizer_reset_requested,
            recognizer_reset_complete,
        )
    run_command(
        ["open", "-a", app_name],
        APP_COMMAND_TIMEOUT_SECONDS,
    )


class DoubleClapDetector:
    """Recognize two separated, short transients from an audio callback."""

    def __init__(self, action_requested: threading.Event) -> None:
        self.action_requested = action_requested
        self.first_clap_time: float | None = None
        self.last_candidate_time = -math.inf
        self.cooldown_until = -math.inf
        self.was_transient = False
        self.quiet_blocks = REARM_BLOCKS

    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        callback_time: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, callback_time, status

        if self.action_requested.is_set():
            return

        samples = indata[:, 0]
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        crest_factor = peak / max(rms, 1e-9)
        is_transient = (
            peak >= CLAP_THRESHOLD
            and rms <= MAX_CLAP_RMS
            and crest_factor >= MIN_CREST_FACTOR
        )

        if DEBUG and peak >= CLAP_THRESHOLD:
            debug(
                f"clap block: peak={peak:.4f}, rms={rms:.4f}, "
                f"crest_factor={crest_factor:.2f}, candidate={is_transient}"
            )

        if not is_transient:
            self.quiet_blocks = min(self.quiet_blocks + 1, REARM_BLOCKS)
            self.was_transient = False
            return

        if self.was_transient or self.quiet_blocks < REARM_BLOCKS:
            return

        self.was_transient = True
        self.quiet_blocks = 0
        now = time.monotonic()

        if now < self.cooldown_until or now - self.last_candidate_time < DEBOUNCE_SECONDS:
            return

        self.last_candidate_time = now
        if self.first_clap_time is None or now - self.first_clap_time > MAX_CLAP_INTERVAL:
            self.first_clap_time = now
            return

        interval = now - self.first_clap_time
        if interval < MIN_CLAP_INTERVAL:
            return

        self.first_clap_time = None
        self.cooldown_until = now + COOLDOWN_SECONDS
        debug("valid double clap detected")
        self.action_requested.set()
        debug("work_action_requested triggered")


class AudioRouter:
    """Send one input stream to clap detection and bounded recognizer queues."""

    def __init__(
        self,
        detector: DoubleClapDetector,
        audio_queue: queue.Queue[object],
        kws_audio_queue: queue.Queue[object],
        speech_suppressed: threading.Event,
        stop_requested: threading.Event,
        runtime_errors: queue.Queue[BaseException],
    ) -> None:
        self.detector = detector
        self.audio_queue = audio_queue
        self.kws_audio_queue = kws_audio_queue
        self.speech_suppressed = speech_suppressed
        self.stop_requested = stop_requested
        self.runtime_errors = runtime_errors

    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        callback_time: object,
        status: sd.CallbackFlags,
    ) -> None:
        try:
            if self.stop_requested.is_set() or self.speech_suppressed.is_set():
                return

            self.detector.audio_callback(indata, frames, callback_time, status)

            samples = indata[:, 0].copy()
            samples.setflags(write=False)
            enqueue_latest_audio(self.audio_queue, samples)
            enqueue_latest_audio(self.kws_audio_queue, samples)
        except BaseException as error:
            try:
                self.runtime_errors.put_nowait(error)
            except queue.Full:
                pass
            self.stop_requested.set()
            raise sd.CallbackAbort


def create_recognizer() -> sherpa_onnx.OnlineRecognizer:
    model_files = (MODEL_TOKENS, MODEL_ENCODER, MODEL_DECODER, MODEL_JOINER)
    missing_files = [str(path) for path in model_files if not path.is_file()]
    if missing_files:
        missing = "\n".join(f"  {path}" for path in missing_files)
        raise FileNotFoundError(f"Missing local speech model files:\n{missing}")

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(MODEL_TOKENS),
        encoder=str(MODEL_ENCODER),
        decoder=str(MODEL_DECODER),
        joiner=str(MODEL_JOINER),
        num_threads=2,
        sample_rate=16_000,
        feature_dim=128,
        decoding_method="greedy_search",
        enable_endpoint_detection=True,
        model_type="",
        provider="cpu",
    )


def create_voice_activity_detector() -> sherpa_onnx.VoiceActivityDetector:
    if not VAD_MODEL.is_file():
        raise FileNotFoundError(f"Missing local Silero VAD model: {VAD_MODEL}")
    if VAD_MODEL.stat().st_size != VAD_MODEL_SIZE:
        raise RuntimeError("The Silero VAD model size does not match the official model")
    if sha256_file(VAD_MODEL) != VAD_MODEL_SHA256:
        raise RuntimeError("The Silero VAD model checksum does not match; refusing to load it")

    silero_config = sherpa_onnx.SileroVadModelConfig(
        model=str(VAD_MODEL),
        threshold=VAD_THRESHOLD,
        min_silence_duration=VAD_MIN_SILENCE_SECONDS,
        min_speech_duration=VAD_MIN_SPEECH_SECONDS,
        window_size=512,
        max_speech_duration=VAD_MAX_SPEECH_SECONDS,
    )
    config = sherpa_onnx.VadModelConfig(
        silero_vad=silero_config,
        sample_rate=WHISPER_SAMPLE_RATE,
        num_threads=1,
        provider="cpu",
    )
    return sherpa_onnx.VoiceActivityDetector(
        config,
        buffer_size_in_seconds=VAD_MAX_SPEECH_SECONDS + VAD_PRE_ROLL_SECONDS,
    )


def create_keyword_spotter() -> sherpa_onnx.KeywordSpotter:
    model_files = (
        KWS_TOKENS,
        KWS_ENCODER,
        KWS_DECODER,
        KWS_JOINER,
        KWS_KEYWORDS,
    )
    missing_files = [str(path) for path in model_files if not path.is_file()]
    if missing_files:
        missing = "\n".join(f"  {path}" for path in missing_files)
        raise FileNotFoundError(f"Missing local KWS model files:\n{missing}")

    return sherpa_onnx.KeywordSpotter(
        tokens=str(KWS_TOKENS),
        encoder=str(KWS_ENCODER),
        decoder=str(KWS_DECODER),
        joiner=str(KWS_JOINER),
        keywords_file=str(KWS_KEYWORDS),
        num_threads=KWS_NUM_THREADS,
        sample_rate=16_000,
        feature_dim=80,
        max_active_paths=4,
        keywords_score=KWS_KEYWORDS_SCORE,
        keywords_threshold=KWS_KEYWORDS_THRESHOLD,
        num_trailing_blanks=1,
        provider="cpu",
    )


def kws_worker(
    keyword_spotter: sherpa_onnx.KeywordSpotter,
    audio_queue: queue.Queue[object],
    wake_requested: threading.Event,
    speech_suppressed: threading.Event,
    stop_requested: threading.Event,
    worker_errors: queue.Queue[BaseException],
) -> None:
    try:
        stream = keyword_spotter.create_stream()
        reset_for_suppression = False
        while not stop_requested.is_set():
            if speech_suppressed.is_set():
                if not reset_for_suppression:
                    keyword_spotter.reset_stream(stream)
                    reset_for_suppression = True
                try:
                    item = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is STOP_AUDIO:
                    break
                continue

            if reset_for_suppression:
                keyword_spotter.reset_stream(stream)
                reset_for_suppression = False

            try:
                item = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is STOP_AUDIO:
                break
            if (
                wake_requested.is_set()
                or speech_suppressed.is_set()
                or not isinstance(item, np.ndarray)
            ):
                continue

            stream.accept_waveform(SAMPLE_RATE, item)
            while keyword_spotter.is_ready(stream):
                keyword_spotter.decode_stream(stream)
                result = keyword_spotter.get_result(stream)
                if result and not speech_suppressed.is_set():
                    keyword_spotter.reset_stream(stream)
                    drain_audio_queue(audio_queue)
                    debug("KWS wake word detected: JARVIS")
                    wake_requested.set()
                    debug("KWS wake pending")
                    break
    except BaseException as error:
        try:
            worker_errors.put_nowait(error)
        except queue.Full:
            pass
        stop_requested.set()


def speech_worker(
    recognizer: sherpa_onnx.OnlineRecognizer | None,
    vad: sherpa_onnx.VoiceActivityDetector | None,
    command_recognizer: WhisperCommandRecognizer | None,
    audio_queue: queue.Queue[object],
    intent_queue: queue.Queue[tuple[str, str]],
    kws_wake_pending: threading.Event,
    speech_suppressed: threading.Event,
    recognizer_reset_requested: threading.Event,
    recognizer_reset_complete: threading.Event,
    stop_requested: threading.Event,
    worker_errors: queue.Queue[BaseException],
) -> None:
    try:
        if (recognizer is None) == (vad is None):
            raise RuntimeError("Exactly one speech endpoint detector must be active")

        stream = recognizer.create_stream() if recognizer is not None else None
        resampler = StreamingLinearResampler(SAMPLE_RATE, WHISPER_SAMPLE_RATE)
        pre_roll: deque[np.ndarray] = deque()
        pre_roll_samples = round(VAD_PRE_ROLL_SECONDS * WHISPER_SAMPLE_RATE)
        speech_audio: list[np.ndarray] = []
        vad_speech_active = False
        kws_pending_since: float | None = None
        kws_standalone_sent = False
        while not stop_requested.is_set():
            if recognizer_reset_requested.is_set():
                if recognizer is not None and stream is not None:
                    recognizer.reset(stream)
                if vad is not None:
                    vad.reset()
                resampler.reset()
                pre_roll.clear()
                speech_audio.clear()
                vad_speech_active = False
                kws_pending_since = None
                kws_standalone_sent = False
                recognizer_reset_requested.clear()
                recognizer_reset_complete.set()

            if kws_wake_pending.is_set():
                if kws_pending_since is None:
                    kws_pending_since = time.monotonic()
                if (
                    not kws_standalone_sent
                    and not vad_speech_active
                    and time.monotonic() - kws_pending_since
                    >= KWS_STANDALONE_TIMEOUT_SECONDS
                ):
                    debug("KWS standalone wake confirmed")
                    enqueue_intent(intent_queue, INTENT_DIRECT_CALL, "kws")
                    kws_standalone_sent = True
            else:
                kws_pending_since = None
                kws_standalone_sent = False

            try:
                item = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is STOP_AUDIO:
                break
            if speech_suppressed.is_set() or not isinstance(item, np.ndarray):
                continue

            if vad is not None:
                vad_audio = resampler.process(item)
                if vad_audio.size == 0:
                    continue

                was_speech = vad_speech_active
                vad.accept_waveform(vad_audio)
                vad_speech_active = vad.is_speech_detected()
                if not was_speech and vad_speech_active:
                    debug("VAD speech started")
                    speech_audio = [*pre_roll, vad_audio]
                    pre_roll.clear()
                elif was_speech:
                    speech_audio.append(vad_audio)
                else:
                    append_pre_roll(pre_roll, vad_audio, pre_roll_samples)

                if was_speech and not vad_speech_active:
                    debug("VAD speech ended")
                    while not vad.empty():
                        vad.pop()
                    audio = np.concatenate(speech_audio)
                    speech_audio.clear()
                    debug(
                        f"speech duration: {audio.size / WHISPER_SAMPLE_RATE:.2f} s"
                    )
                    inference_started = time.monotonic()
                    try:
                        if command_recognizer is None:
                            raise RuntimeError("Whisper is unavailable in Silero VAD mode")
                        text = command_recognizer.transcribe(
                            audio, sample_rate=WHISPER_SAMPLE_RATE
                        )
                    finally:
                        inference_ms = round(
                            (time.monotonic() - inference_started) * 1000
                        )
                        debug(f"whisper inference: {inference_ms} ms")
                    debug(f'whisper transcription: "{text}"')
                    process_transcription(text, intent_queue, kws_wake_pending)
                continue

            if recognizer is None or stream is None:
                raise RuntimeError("Nemotron rollback recognizer is unavailable")
            stream.accept_waveform(SAMPLE_RATE, item)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            if recognizer.is_endpoint(stream):
                text = recognizer.get_result(stream)
                recognizer.reset(stream)
                debug(f'nemotron transcription: "{text}"')
                process_transcription(text, intent_queue, kws_wake_pending)
    except BaseException as error:
        try:
            worker_errors.put_nowait(error)
        except queue.Full:
            pass
        stop_requested.set()


def stop_speech_worker(
    worker: threading.Thread,
    audio_queue: queue.Queue[object],
    speech_suppressed: threading.Event,
    stop_requested: threading.Event,
) -> None:
    if worker.ident is None:
        return

    speech_suppressed.set()
    stop_requested.set()
    drain_audio_queue(audio_queue)
    try:
        audio_queue.put_nowait(STOP_AUDIO)
    except queue.Full:
        pass
    worker.join(timeout=3.0)
    if worker.is_alive():
        raise RuntimeError("The local speech-recognition worker did not stop cleanly")


def stop_kws_worker(
    worker: threading.Thread,
    audio_queue: queue.Queue[object],
) -> None:
    if worker.ident is None:
        return

    drain_audio_queue(audio_queue)
    try:
        audio_queue.put_nowait(STOP_AUDIO)
    except queue.Full:
        pass
    worker.join(timeout=3.0)
    if worker.is_alive():
        raise RuntimeError("The local keyword-spotting worker did not stop cleanly")


def main() -> None:
    if COMMAND_ASR not in {"whisper", "nemotron"}:
        raise ValueError('COMMAND_ASR must be either "whisper" or "nemotron"')

    actions_launched = False
    caffeinate_process: subprocess.Popen | None = None
    work_action_requested = threading.Event()
    speech_suppressed = threading.Event()
    recognizer_reset_requested = threading.Event()
    recognizer_reset_complete = threading.Event()
    stop_requested = threading.Event()
    kws_wake_requested = threading.Event()
    audio_queue: queue.Queue[object] = queue.Queue(maxsize=AUDIO_QUEUE_BLOCKS)
    kws_audio_queue: queue.Queue[object] = queue.Queue(
        maxsize=KWS_AUDIO_QUEUE_BLOCKS
    )
    intent_queue: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)
    runtime_errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)

    recognizer = create_recognizer() if COMMAND_ASR == "nemotron" else None
    vad = create_voice_activity_detector() if COMMAND_ASR == "whisper" else None
    command_recognizer = (
        WhisperCommandRecognizer() if COMMAND_ASR == "whisper" else None
    )
    keyword_spotter = create_keyword_spotter()
    say(greeting_for_current_time())

    detector = DoubleClapDetector(work_action_requested)
    audio_router = AudioRouter(
        detector,
        audio_queue,
        kws_audio_queue,
        speech_suppressed,
        stop_requested,
        runtime_errors,
    )
    worker = threading.Thread(
        target=speech_worker,
        name="jarvis-speech-worker",
        args=(
            recognizer,
            vad,
            command_recognizer,
            audio_queue,
            intent_queue,
            kws_wake_requested,
            speech_suppressed,
            recognizer_reset_requested,
            recognizer_reset_complete,
            stop_requested,
            runtime_errors,
        ),
        daemon=True,
    )
    kws_thread = threading.Thread(
        target=kws_worker,
        name="jarvis-kws-worker",
        args=(
            keyword_spotter,
            kws_audio_queue,
            kws_wake_requested,
            speech_suppressed,
            stop_requested,
            runtime_errors,
        ),
        daemon=True,
    )

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        worker.start()
        kws_thread.start()
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=audio_router.audio_callback,
        ):
            while not stop_requested.is_set():
                if work_action_requested.is_set() and not actions_launched:
                    launch_work_apps()
                    actions_launched = True

                try:
                    intent, source = intent_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if intent == INTENT_DIRECT_CALL:
                    try:
                        respond_to_direct_call(
                            source,
                            audio_queue,
                            speech_suppressed,
                            recognizer_reset_requested,
                            recognizer_reset_complete,
                        )
                    finally:
                        if source == "kws":
                            kws_wake_requested.clear()
                    continue
                if intent == INTENT_WORK_MODE:
                    if not actions_launched and launch_work_apps():
                        actions_launched = True
                        speak_while_suppressed(
                            random.choice(WORK_MODE_RESPONSES),
                            audio_queue,
                            speech_suppressed,
                            recognizer_reset_requested,
                            recognizer_reset_complete,
                        )
                    continue
                if intent == INTENT_BREAK_TIME:
                    response = random.choice(BREAK_TIME_RESPONSES).format(
                        address=random.choice(("boss", "sir"))
                    )
                    speak_while_suppressed(
                        response,
                        audio_queue,
                        speech_suppressed,
                        recognizer_reset_requested,
                        recognizer_reset_complete,
                    )
                    run_command(["open", YOUTUBE_URL], APP_COMMAND_TIMEOUT_SECONDS)
                    continue
                if intent == INTENT_GRATITUDE:
                    speak_while_suppressed(
                        random.choice(GRATITUDE_RESPONSES),
                        audio_queue,
                        speech_suppressed,
                        recognizer_reset_requested,
                        recognizer_reset_complete,
                    )
                    continue
                if intent in {INTENT_VPN_ON, INTENT_VPN_OFF}:
                    enabled = intent == INTENT_VPN_ON
                    if change_vpn_state(enabled=enabled):
                        responses = VPN_ON_RESPONSES if enabled else VPN_OFF_RESPONSES
                        speak_while_suppressed(
                            random.choice(responses),
                            audio_queue,
                            speech_suppressed,
                            recognizer_reset_requested,
                            recognizer_reset_complete,
                        )
                    continue
                if intent in {
                    INTENT_STAY_AWAKE,
                    INTENT_SLEEP_NORMALLY,
                    INTENT_SPECIAL_SLEEP,
                }:
                    if intent == INTENT_STAY_AWAKE:
                        caffeinate_process, changed = start_stay_awake(
                            caffeinate_process
                        )
                        response = random.choice(STAY_AWAKE_RESPONSES)
                    else:
                        caffeinate_process, changed = stop_stay_awake(
                            caffeinate_process
                        )
                        response = (
                            SPECIAL_SLEEP_RESPONSE
                            if intent == INTENT_SPECIAL_SLEEP
                            else random.choice(SLEEP_NORMALLY_RESPONSES)
                        )
                    if changed:
                        speak_while_suppressed(
                            response,
                            audio_queue,
                            speech_suppressed,
                            recognizer_reset_requested,
                            recognizer_reset_complete,
                        )
                    continue
                launch_voice_app(
                    intent,
                    audio_queue,
                    speech_suppressed,
                    recognizer_reset_requested,
                    recognizer_reset_complete,
                )
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop_speech_worker(worker, audio_queue, speech_suppressed, stop_requested)
        finally:
            try:
                stop_kws_worker(kws_thread, kws_audio_queue)
            finally:
                try:
                    stop_stay_awake(caffeinate_process)
                finally:
                    if command_recognizer is not None and not worker.is_alive():
                        command_recognizer.close()

    if not runtime_errors.empty():
        raise RuntimeError("A local audio component stopped unexpectedly") from (
            runtime_errors.get_nowait()
        )


if __name__ == "__main__":
    main()
