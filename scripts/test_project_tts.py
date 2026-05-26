#!/usr/bin/env python3
"""Small smoke-test runner for the project's TTS worker pipeline.

The script mirrors the app's queue contract:
  request_queue.put((speech_id, text))
  request_queue.put((None, None))

Most workers return 48 kHz mono int16 PCM bytes, which are written as WAV.
DashScope custom voice returns OGG/Opus frames through ("__audio__", sid, bytes);
those frames are written as an OGG file instead.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PCM_SAMPLE_RATE = 48_000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
DEFAULT_TEXT = "\u4f60\u597d\uff0c\u8fd9\u662f\u4e00\u6b21 TTS \u7ba1\u7ebf\u6d4b\u8bd5\u3002"


@dataclass
class RuntimeConfig:
    provider: str
    worker_name: str
    tts_config_name: str
    api_key: str
    voice_id: str
    has_custom_tts: bool
    notes: list[str]


@dataclass
class AudioCapture:
    pcm: bytearray
    encoded: bytearray
    pcm_chunks: int = 0
    encoded_chunks: int = 0
    control_messages: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def _is_free_preset_voice(voice_id: str) -> bool:
    if not voice_id:
        return False
    from utils.api_config_loader import get_free_voices  # noqa: WPS433

    return voice_id in set(get_free_voices().values())


def resolve_runtime_config(args: argparse.Namespace) -> tuple[RuntimeConfig, Any]:
    from main_logic.tts_client import get_tts_worker  # noqa: WPS433
    from utils.config_manager import get_config_manager  # noqa: WPS433

    cm = get_config_manager()
    core_config = cm.get_core_config()
    realtime_config = cm.get_model_api_config("realtime")

    provider = args.provider
    if provider == "auto" or provider == "custom":
        provider = realtime_config.get("api_type", "") or core_config.get("CORE_API_TYPE", "") or "qwen"

    voice_id = args.voice_id
    if voice_id is None:
        voice_id = core_config.get("TTS_VOICE_ID", "") or ""
    if voice_id.startswith("__gptsovits_disabled__"):
        voice_id = ""

    is_free_preset = _is_free_preset_voice(voice_id)
    if is_free_preset and provider != "free":
        voice_id = ""
        is_free_preset = False

    has_custom_tts = (
        args.provider == "custom"
        or ((bool(voice_id) and not is_free_preset))
        or bool(core_config.get("ENABLE_CUSTOM_API") and core_config.get("TTS_MODEL_URL"))
    )
    if args.force_default:
        has_custom_tts = False

    notes: list[str] = []
    default_providers = {"qwen", "free", "step", "glm", "gemini", "openai"}
    if args.provider == "auto" and not args.strict_auto and not has_custom_tts and provider not in default_providers:
        notes.append(
            f"auto resolved unsupported provider '{provider}', using 'free' for a TTS smoke test"
        )
        provider = "free"

    tts_config_name = "tts_custom" if has_custom_tts else "tts_default"
    tts_config = cm.get_model_api_config(tts_config_name)
    api_key = args.api_key if args.api_key is not None else (tts_config.get("api_key", "") or "")

    if args.provider == "free" and args.api_key is None and not api_key:
        api_key = "free-access"
    if provider == "free" and args.api_key is None and not api_key:
        api_key = "free-access"

    worker = get_tts_worker(core_api_type=provider, has_custom_voice=has_custom_tts)
    runtime = RuntimeConfig(
        provider=provider,
        worker_name=getattr(worker, "__name__", repr(worker)),
        tts_config_name=tts_config_name,
        api_key=api_key,
        voice_id=voice_id,
        has_custom_tts=has_custom_tts,
        notes=notes,
    )
    return runtime, worker


def wait_for_ready(
    response_queue: queue.Queue[Any],
    timeout_s: float,
) -> tuple[bool, list[Any]]:
    deadline = time.monotonic() + timeout_s
    buffered: list[Any] = []

    while time.monotonic() < deadline:
        try:
            item = response_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if isinstance(item, tuple) and item:
            tag = item[0]
            if tag == "__ready__":
                return bool(item[1]), buffered
            if tag == "__error__":
                print(f"[tts] error before ready: {item[1]}", file=sys.stderr)
            buffered.append(item)
        else:
            buffered.append(item)

    return False, buffered


def _record_item(item: Any, capture: AudioCapture) -> bool:
    if isinstance(item, bytes):
        if item:
            capture.pcm.extend(item)
            capture.pcm_chunks += 1
            return True
        return False

    if isinstance(item, tuple) and item:
        tag = item[0]
        if tag == "__audio__" and len(item) >= 3 and isinstance(item[2], bytes):
            if item[2]:
                capture.encoded.extend(item[2])
                capture.encoded_chunks += 1
                return True
        elif tag == "__error__":
            capture.errors.append(str(item[1] if len(item) > 1 else "unknown TTS error"))
        else:
            capture.control_messages += 1
    else:
        capture.control_messages += 1

    return False


def collect_audio(
    response_queue: queue.Queue[Any],
    prebuffered: list[Any],
    timeout_s: float,
    silence_timeout_s: float,
) -> AudioCapture:
    capture = AudioCapture(pcm=bytearray(), encoded=bytearray())
    start = time.monotonic()
    last_audio_at: float | None = None

    for item in prebuffered:
        if _record_item(item, capture):
            last_audio_at = time.monotonic()

    while time.monotonic() - start < timeout_s:
        if last_audio_at is not None and time.monotonic() - last_audio_at >= silence_timeout_s:
            break

        try:
            item = response_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if _record_item(item, capture):
            last_audio_at = time.monotonic()

    return capture


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH)
        wav_file.setframerate(PCM_SAMPLE_RATE)
        wav_file.writeframes(pcm)


def write_encoded(path: Path, encoded: bytes) -> Path:
    if path.suffix.lower() not in {".ogg", ".opus"}:
        path = path.with_suffix(".ogg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return path


def print_runtime(runtime: RuntimeConfig) -> None:
    masked_key = "(empty)"
    if runtime.api_key:
        masked_key = f"{runtime.api_key[:4]}...{runtime.api_key[-4:]}" if len(runtime.api_key) > 8 else "***"

    print("[tts] runtime")
    print(f"  provider: {runtime.provider}")
    print(f"  worker: {runtime.worker_name}")
    print(f"  config: {runtime.tts_config_name}")
    print(f"  custom: {runtime.has_custom_tts}")
    print(f"  voice_id: {runtime.voice_id or '(default)'}")
    print(f"  api_key: {masked_key}")
    for note in runtime.notes:
        print(f"  note: {note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small TTS smoke test through the project worker pipeline.")
    parser.add_argument(
        "--provider",
        choices=["auto", "custom", "qwen", "free", "step", "glm", "gemini", "openai"],
        default="auto",
        help="TTS provider to use. auto mirrors the project config.",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize.")
    parser.add_argument("--voice-id", default=None, help="Override voice_id. Empty string forces provider default.")
    parser.add_argument("--api-key", default=None, help="Override API key passed into the worker.")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "output" / "tts_smoke.wav"),
        help="Output path. PCM workers write WAV; OGG workers switch the suffix to .ogg.",
    )
    parser.add_argument("--ready-timeout", type=float, default=12.0, help="Seconds to wait for __ready__.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Max seconds to wait for audio after sending text.")
    parser.add_argument("--silence-timeout", type=float, default=2.5, help="Stop after this many seconds without new audio.")
    parser.add_argument("--force-default", action="store_true", help="Ignore custom TTS config and use provider default TTS.")
    parser.add_argument(
        "--strict-auto",
        action="store_true",
        help="With --provider auto, keep the exact configured provider even if it resolves to dummy_tts_worker.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and print it without starting TTS.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime, worker = resolve_runtime_config(args)
    print_runtime(runtime)

    if runtime.worker_name == "dummy_tts_worker":
        print(
            "[tts] resolved to dummy_tts_worker; this config cannot synthesize audio. "
            "Try --provider free/qwen/step/glm/gemini/openai, or configure custom TTS.",
            file=sys.stderr,
        )
        if not args.dry_run:
            return 3

    if args.dry_run:
        print("[tts] dry run complete; no synthesis requested.")
        return 0

    request_queue: queue.Queue[Any] = queue.Queue()
    response_queue: queue.Queue[Any] = queue.Queue()
    thread = threading.Thread(
        target=worker,
        args=(request_queue, response_queue, runtime.api_key, runtime.voice_id),
        daemon=True,
        name=f"tts-smoke-{runtime.provider}",
    )
    thread.start()

    ready, prebuffered = wait_for_ready(response_queue, args.ready_timeout)
    if not ready:
        print(f"[tts] worker was not ready within {args.ready_timeout:.1f}s", file=sys.stderr)
        request_queue.put(("__interrupt__", None))
        return 2

    speech_id = f"tts-smoke-{uuid.uuid4().hex[:8]}"
    print(f"[tts] sending {len(args.text)} chars, speech_id={speech_id}")
    request_queue.put((speech_id, args.text))
    request_queue.put((None, None))

    capture = collect_audio(response_queue, prebuffered, args.timeout, args.silence_timeout)
    request_queue.put(("__interrupt__", None))

    if capture.errors:
        for error in capture.errors:
            print(f"[tts] worker error: {error}", file=sys.stderr)

    output = Path(args.output)
    if capture.pcm:
        write_wav(output, bytes(capture.pcm))
        duration = len(capture.pcm) / (PCM_SAMPLE_RATE * PCM_SAMPLE_WIDTH * PCM_CHANNELS)
        print(f"[tts] wrote WAV: {output.resolve()}")
        print(f"[tts] pcm_chunks={capture.pcm_chunks}, bytes={len(capture.pcm)}, duration~{duration:.2f}s")
        return 0

    if capture.encoded:
        encoded_path = write_encoded(output, bytes(capture.encoded))
        print(f"[tts] wrote encoded audio: {encoded_path.resolve()}")
        print(f"[tts] encoded_chunks={capture.encoded_chunks}, bytes={len(capture.encoded)}")
        return 0

    print("[tts] no audio captured", file=sys.stderr)
    print(f"[tts] control_messages={capture.control_messages}, errors={len(capture.errors)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
