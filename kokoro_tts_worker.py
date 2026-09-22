#!/usr/bin/env python3
"""Persistent local Kokoro-MLX TTS helper for Jarvis."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import wave
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from kokoro_mlx import KokoroTTS
from kokoro_mlx.phonemize import Phonemizer, normalize_language
from misaki import espeak


_original_build_g2p = Phonemizer._build_g2p


def _build_offline_g2p(language: str):
    normalized = normalize_language(language)
    if normalized in {"en-us", "en-gb"}:
        return espeak.EspeakG2P(language=normalized)
    return _original_build_g2p(language)


Phonemizer._build_g2p = staticmethod(_build_offline_g2p)


def send(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def play_audio(audio: np.ndarray, sample_rate: int, timeout: float) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = np.ascontiguousarray(pcm * 32767.0, dtype=np.int16)
    audio_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="jarvis-kokoro-", suffix=".wav", delete=False
        ) as audio_file:
            audio_path = audio_file.name
        with wave.open(audio_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())

        process = subprocess.Popen(
            ["/usr/bin/afplay", audio_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise TimeoutError("Kokoro audio playback timed out")
        if process.returncode != 0:
            raise RuntimeError("Kokoro audio playback failed")
    finally:
        if audio_path is not None:
            try:
                os.unlink(audio_path)
            except OSError:
                pass


def main() -> int:
    model_directory = Path(sys.argv[1])
    voice = sys.argv[2]
    timeout = float(sys.argv[3])
    tts: KokoroTTS | None = None
    try:
        if not model_directory.is_dir():
            send({"ready": False, "error": "model directory unavailable"})
            return 1

        with redirect_stdout(sys.stderr):
            tts = KokoroTTS.from_pretrained(str(model_directory))
            voices = tts.list_voices()
        if voice not in voices:
            send({"ready": False, "error": "configured voice unavailable"})
            return 1

        send({"ready": True})
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("shutdown") is True:
                send({"closed": True})
                return 0

            text = request.get("text")
            if not isinstance(text, str):
                send({"ok": False, "error": "invalid text payload"})
                continue

            with redirect_stdout(sys.stderr):
                result = tts.generate(
                    text,
                    voice=voice,
                    sample_rate=24_000,
                )
            play_audio(result.audio, result.sample_rate, timeout)
            send({"ok": True})
        return 0
    except Exception:
        send({"ready": False, "ok": False, "error": "Kokoro TTS worker failed"})
        return 1
    finally:
        if tts is not None:
            tts.close()


if __name__ == "__main__":
    raise SystemExit(main())
