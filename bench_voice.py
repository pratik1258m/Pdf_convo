"""Measure the local speech engines side by side.

Run: python3 bench_voice.py [--engines piper,supertonic] [--voices M1]

Everything here is a measurement of this machine, not an assertion: the table
prints time-to-first-audio-chunk, total wall time, seconds of audio produced
and the real-time factor, and the WAVs land in tts_models/samples/ so the
voice can be judged by ear rather than by a number.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MODEL_DIR = Path(__file__).resolve().parent / "tts_models"
SAMPLE_DIR = MODEL_DIR / "samples"

MIX = "Pehle pip install selenium pytest chalao, phir login form par username standard_user daalo."
HI_DEV = "सबसे पहले pip install selenium pytest चलाएँ, फिर लॉगिन फ़ॉर्म पर username standard_user डालें।"
EN = "First run pip install selenium pytest, then open the login form and assert the page title."

SENTENCES: List[Tuple[str, str, Optional[str]]] = [
    # (label, text, forced language for Supertonic; None = engine default)
    ("mixed", MIX, "na"),
    ("mixed_en", MIX, "en"),
    ("devanagari", HI_DEV, "hi"),
    ("english", EN, "en"),
]

PIPER_VOICES = [
    "en_US-ryan-medium",
    "en_US-lessac-medium",
    "hi_IN-pratham-medium",
    "hi_IN-priyamvada-medium",
    "hi_IN-rohan-medium",
]
# A shortlist to judge by ear: two male and two female styles, the four the
# voice sheet recommends first. Pass --voices to compare all ten.
SUPERTONIC_VOICES = ["M1", "M2", "F1", "F2"]


def write_wav(path: Path, pcm: bytes, sample_rate: int, width: int, channels: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return path.stat().st_size


def bench_piper(voice_names: List[str]) -> List[Dict[str, Any]]:
    from piper import PiperVoice, SynthesisConfig

    rows: List[Dict[str, Any]] = []
    for name in voice_names:
        model = MODEL_DIR / f"{name}.onnx"
        if not model.exists():
            print(f"  !! missing {model} — run: python3 -m piper.download_voices {name} --download-dir tts_models")
            continue
        # Warm the ONNX session twice; the first load pays for the disk read.
        load_ms: List[int] = []
        for _ in range(2):
            t0 = time.perf_counter()
            voice = PiperVoice.load(model)
            load_ms.append(round((time.perf_counter() - t0) * 1000))
        for label, text, _lang in SENTENCES:
            pcm = bytearray()
            rate, width, channels = 22050, 2, 1
            first_chunk_ms = 0.0
            t1 = time.perf_counter()
            for chunk in voice.synthesize(text, SynthesisConfig()):
                if not pcm:
                    first_chunk_ms = (time.perf_counter() - t1) * 1000
                rate = chunk.sample_rate
                width = chunk.sample_width
                channels = chunk.sample_channels
                pcm += chunk.audio_int16_bytes
            total_ms = (time.perf_counter() - t1) * 1000
            if not pcm:
                print(f"  !! unexpected chunk payload for {name}/{label}")
                continue
            audio_s = len(pcm) / (rate * width * channels)
            out = SAMPLE_DIR / f"piper_{name}_{label}.wav"
            size = write_wav(out, bytes(pcm), rate, width, channels)
            rows.append(
                {
                    "engine": "piper",
                    "voice": name,
                    "case": label,
                    "ttfa_ms": round(first_chunk_ms),
                    "total_ms": round(total_ms),
                    "audio_s": round(audio_s, 2),
                    "rtf": round(total_ms / 1000 / audio_s, 3),
                    "bytes": size,
                    "rate": rate,
                    "load_ms": load_ms,
                }
            )
    return rows


def bench_supertonic(voice_names: List[str], steps: List[int]) -> List[Dict[str, Any]]:
    import supertonic

    t0 = time.perf_counter()
    tts = supertonic.TTS(model_dir=str(MODEL_DIR / "supertonic3"))
    load_ms = round((time.perf_counter() - t0) * 1000)
    rate = tts.sample_rate
    rows: List[Dict[str, Any]] = []
    for name in voice_names:
        style = tts.get_voice_style(name)
        for label, text, lang in SENTENCES:
            for n_steps in steps:
                t1 = time.perf_counter()
                wav, dur = tts.synthesize(text, voice_style=style, lang=lang, total_steps=n_steps)
                total_ms = (time.perf_counter() - t1) * 1000
                audio_s = float(dur[0])
                out = SAMPLE_DIR / f"supertonic_{name}_s{n_steps}_{label}.wav"
                tts.save_audio(wav, str(out))
                size = out.stat().st_size
                rows.append(
                    {
                        "engine": "supertonic",
                        "voice": f"{name} steps={n_steps}",
                        "case": label,
                        "ttfa_ms": total_ms,  # not streamable: the whole clip is the first chunk
                        "total_ms": round(total_ms),
                        "audio_s": round(audio_s, 2),
                        "rtf": round(total_ms / 1000 / audio_s, 3),
                        "bytes": size,
                        "rate": rate,
                        "load_ms": [load_ms],
                    }
                )
    return rows


def show(rows: List[Dict[str, Any]]) -> None:
    head = f"{'engine':<11} {'voice':<28} {'case':<12} {'ttfa':>6} {'total':>7} {'audio':>7} {'rtf':>6} {'bytes':>9} {'rate':>6}"
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['engine']:<11} {r['voice']:<28} {r['case']:<12} "
            f"{r['ttfa_ms']:>5}ms {r['total_ms']:>6}ms {r['audio_s']:>6}s "
            f"{r['rtf']:>6} {r['bytes']:>9} {r['rate']:>6}"
        )
    loads: Dict[str, List[int]] = {}
    for r in rows:
        loads.setdefault(f"{r['engine']} {r['voice'].split(' steps')[0]}", r["load_ms"])
    print("\nmodel load (cold, warm):")
    for k, v in loads.items():
        print(f"  {k:<28} {' / '.join(str(x) + 'ms' for x in v)}")


def phoneme_evidence() -> None:
    """Show what a monolingual phonemizer does with romanised Hindi."""
    from piper import PiperVoice

    for name in ["hi_IN-pratham-medium", "en_US-ryan-medium"]:
        model = MODEL_DIR / f"{name}.onnx"
        if not model.exists():
            continue
        voice = PiperVoice.load(model)
        print(f"\n{name}")
        print(f"  mixed      -> {voice.phonemize(MIX)[:14]}")
        print(f"  devanagari -> {voice.phonemize(HI_DEV)[:14]}")


def main() -> int:
    from voice_engine import SupertonicProvider

    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="piper,supertonic")
    ap.add_argument("--voices", default="")
    # The number the application actually speaks with, so the clip you listen to
    # is the clip the phone will play.
    ap.add_argument("--steps", default=str(SupertonicProvider.TOTAL_STEPS))
    ap.add_argument("--no-phonemes", action="store_true")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    rows: List[Dict[str, Any]] = []

    if "piper" in engines:
        print(f"piper: {len(voices or PIPER_VOICES)} voices x {len(SENTENCES)} cases")
        rows += bench_piper(voices or PIPER_VOICES)
    if "supertonic" in engines:
        print("supertonic-3: loading model ...")
        rows += bench_supertonic([v for v in voices if v in ("M1", "M2", "M3", "F1", "F2")] or SUPERTONIC_VOICES, steps)

    show(rows)
    if not args.no_phonemes and "piper" in engines:
        phoneme_evidence()
    print(f"\nsamples -> {SAMPLE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
