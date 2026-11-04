"""Speech synthesis behind one contract: cache-first, atomic on disk, honest about failure.

Three providers answer to the same interface, and the voice id decides which one
is used:

  * `edge`       — Microsoft Edge-TTS over the network. Most natural of the three
                   at no cost, but it is a round trip (measured ~1.8 s for the
                   first sentence) and it throttles bursts with HTTP 429.
  * `piper`      — local ONNX voices. About 0.15 s for a whole sentence, no
                   network at all, 22.05 kHz mono. This is the real-time path.
  * `supertonic` — local Supertonic 3. One speaker reads Hindi and English
                   (and 29 other languages) at 44.1 kHz; ~0.4-0.8 s per sentence
                   at 4 steps, so it is both local and noticeably softer-edged.

What the engine guarantees is the same whichever provider is behind it:

  * clips are content-addressed, so asking the same sentence twice costs nothing;
  * a clip is only ever visible once every byte of it is on disk (`.part` then
    `os.replace`), because a truncated WAV or MP3 plays as a click or not at all;
  * a voice whose model is missing fails with the reason, and never silently
    changes the speaker mid-answer;
  * `status` is `partial` when some but not all segments were made.
"""

import asyncio
import base64
import hashlib
import importlib.util
import logging
import os
import time
import uuid
import weakref
import wave
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import edge_tts

logger = logging.getLogger(__name__)

# Voices are data, not code: one entry per speaker, and `engine` picks the
# provider that can actually make audio for it.
VOICE_CATALOG: List[Dict[str, Any]] = [
    # ---- 1. Primary Natural Human Voice (Edge Neural - exact match to user voice sample)
    {
        "id": "hi-IN-MadhurNeural",
        "name": "Madhur (Natural Human Hindi & English - Male)",
        "locale": "hi-IN",
        "gender": "Male",
        "recommended": True,
        "engine": "edge",
        "format": "mp3",
        "local": False,
        "model": "",
    },
    {
        "id": "hi-IN-SwaraNeural",
        "name": "Swara (Natural Human Hindi & English - Female)",
        "locale": "hi-IN",
        "gender": "Female",
        "recommended": True,
        "engine": "edge",
        "format": "mp3",
        "local": False,
        "model": "",
    },
    # ---- 2. VoiceStudio Supertonic 3 (Local-first from debpalash/voicestudio)
    {
        "id": "supertonic-M1",
        "name": "Supertonic M1 (VoiceStudio Local - Hindi & English - Male)",
        "locale": "en-IN",
        "gender": "Male",
        "recommended": True,
        "engine": "supertonic",
        "format": "wav",
        "local": True,
        "model": "M1",
        "languages": "Hindi and English bilingual",
    },
    {
        "id": "supertonic-F1",
        "name": "Supertonic F1 (VoiceStudio Local - Hindi & English - Female)",
        "locale": "en-IN",
        "gender": "Female",
        "recommended": False,
        "engine": "supertonic",
        "format": "wav",
        "local": True,
        "model": "F1",
        "languages": "Hindi and English bilingual",
    },
    # ---- 3. Indian English & Test Baseline
    {
        "id": "en-IN-PrabhatNeural",
        "name": "Prabhat (Edge - Indian English - Male)",
        "locale": "en-IN",
        "gender": "Male",
        "recommended": False,
        "engine": "edge",
        "format": "mp3",
        "local": False,
        "model": "",
    },
    {
        "id": "en_US-ryan-medium",
        "name": "Ryan (Local Piper - English - Male)",
        "locale": "en-US",
        "gender": "Male",
        "recommended": False,
        "engine": "piper",
        "format": "wav",
        "local": True,
        "model": "en_US-ryan-medium.onnx",
    },
]

VOICE_BY_ID: Dict[str, Dict[str, Any]] = {voice["id"]: voice for voice in VOICE_CATALOG}
VOICE_IDS = set(VOICE_BY_ID)

# Preference order: Madhur (matches test_voice.mp3) first, then Swara, then VoiceStudio M1
AUTO_DEFAULT_ORDER = [
    "hi-IN-MadhurNeural",
    "hi-IN-SwaraNeural",
    "supertonic-M1",
    "en-IN-PrabhatNeural",
    "en_US-ryan-medium",
]

FALLBACK_VOICE = "hi-IN-MadhurNeural"

DEFAULT_RATE = "+8%"

# Edge-TTS streams into the file it is given, so an interrupted save leaves a
# truncated MP3. Clips are therefore written beside their final name and only
# renamed into place once every byte has arrived.
MIN_CLIP_BYTES = 200

# The endpoint stalls now and then; without a bound the request hangs until the
# phone gives up and the coroutine stays alive holding the slot.
SYNTHESIS_TIMEOUT_S = 20.0

# Local models are CPU work, not network work. A synthesis that has not landed
# after this long is a hung process, not a slow one, and the caller would rather
# report a failure than leave the phone waiting.
LOCAL_TIMEOUT_S = 20.0

# A clip is handed to the phone the moment it is made, and a replay from stored
# history can fetch it much later, so eviction only reaps settled files.
MIN_REAP_AGE_S = 600.0

# Inline audio is a legacy field; a 600 kB WAV does not belong in a JSON body.
MAX_INLINE_BYTES = 256 * 1024

MODEL_DIR = os.getenv("CONVO_TTS_MODELS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_models")

DEVANAGARI = ("\u0900", "\u097f")


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _rate_multiplier(rate: str) -> float:
    """`+8%` -> 1.08, so one client-side knob drives all three engines."""
    try:
        percent = float(rate.replace("%", "").replace("+", ""))
    except (TypeError, ValueError):
        return 1.0
    return max(0.5, min(2.0, 1.0 + percent / 100.0))


def _has_devanagari(text: str) -> bool:
    return any(DEVANAGARI[0] <= ch <= DEVANAGARI[1] for ch in text)


class EdgeProvider:
    """Microsoft Edge-TTS: free, natural, and one network round trip per clip."""

    engine = "edge"
    audio_format = "mp3"
    parallel = True

    def status(self) -> Tuple[bool, str]:
        return True, ""

    def ready(self, voice: Dict[str, Any]) -> Tuple[bool, str]:
        return True, ""

    async def synthesize(self, text: str, voice: Dict[str, Any], rate: str, part_path: str) -> None:
        communicate = edge_tts.Communicate(text, voice["id"], rate=rate)
        try:
            await asyncio.wait_for(communicate.save(part_path), timeout=SYNTHESIS_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Edge-TTS did not answer within {SYNTHESIS_TIMEOUT_S:g}s")


class LocalProvider:
    """Shared plumbing for the engines that run on this machine.

    Two things every local provider has to get right:

    * the ONNX work is blocking, so it goes to a worker thread and a lock keeps
      one sentence at a time — an ONNX model already uses every core for one
      sentence, so a second sentence in parallel delays the first one the phone
      is waiting for;
    * the lock belongs to the event loop that created it. Servers reload and tests
      run several loops, so the lock is made per loop instead of being shared.
    """

    parallel = False

    def __init__(self) -> None:
        self._locks: "weakref.WeakKeyDictionary[Any, asyncio.Lock]" = weakref.WeakKeyDictionary()

    @property
    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[loop] = lock
        return lock

    def _run(self, voice: Dict[str, Any], text: str, rate: str, part_path: str) -> None:
        raise NotImplementedError

    async def warm(self, voice: Dict[str, Any]) -> None:
        """Load the model now, without speaking.

        A warm-up sentence is not proof of anything: it is a cache hit as soon as
        the same line has been spoken once, and then the ONNX session is still
        missing and an answer pays for the load. This is the honest version.
        """
        async with self._lock:
            await asyncio.to_thread(self._warm, voice)

    def _warm(self, voice: Dict[str, Any]) -> None:
        raise NotImplementedError

    async def synthesize(self, text: str, voice: Dict[str, Any], rate: str, part_path: str) -> None:
        async with self._lock:
            # A worker thread cannot be killed, so on a timeout it is left to
            # finish into its own `.part` file, which the reaper then removes.
            done = asyncio.ensure_future(asyncio.to_thread(self._run, voice, text, rate, part_path))
            try:
                await asyncio.wait_for(asyncio.shield(done), timeout=LOCAL_TIMEOUT_S)
            except asyncio.TimeoutError:
                raise RuntimeError(f"{self.engine} did not finish within {LOCAL_TIMEOUT_S:g}s")


class PiperProvider(LocalProvider):
    """Piper voices: a whole sentence in about 0.15 s, entirely offline.

    Sessions are loaded once and reused (a load costs 0.6-0.9 s on this machine).
    """

    engine = "piper"
    audio_format = "wav"

    def __init__(self, model_dir: str = MODEL_DIR):
        super().__init__()
        self.model_dir = model_dir
        self._voices: Dict[str, Any] = {}

    def _model_path(self, voice: Dict[str, Any]) -> str:
        return os.path.join(self.model_dir, voice.get("model") or f"{voice['id']}.onnx")

    def ready(self, voice: Dict[str, Any]) -> Tuple[bool, str]:
        if importlib.util.find_spec("piper") is None:
            return False, "Piper is not installed. Run: python3 -m pip install piper-tts"
        path = self._model_path(voice)
        if not os.path.exists(path):
            return False, (
                f"{os.path.basename(path)} is missing. "
                f"Run: python3 -m piper.download_voices {voice['id']} --download-dir {self.model_dir}"
            )
        return True, ""

    def _load(self, voice: Dict[str, Any]):
        from piper import PiperVoice

        # Only ever called under the lock, so one model can not be loaded twice.
        return PiperVoice.load(self._model_path(voice))

    def _warm(self, voice: Dict[str, Any]) -> None:
        if voice["id"] not in self._voices:
            self._voices[voice["id"]] = self._load(voice)

    def _run(self, voice: Dict[str, Any], text: str, rate: str, part_path: str) -> None:
        from piper import SynthesisConfig

        speaker = self._voices.get(voice["id"])
        if speaker is None:
            speaker = self._load(voice)
            self._voices[voice["id"]] = speaker
        # length_scale is spoken duration: 1.08 means 8% slower, so the client's
        # "+8%" rate has to invert to keep meaning ("faster" stays faster).
        config = SynthesisConfig(length_scale=round(1.0 / _rate_multiplier(rate), 4))
        pending = None
        with wave.open(part_path, "wb") as out:
            for chunk in speaker.synthesize(text, config):
                if pending is None:
                    # The header has to be written before the first frame, and
                    # the frame rate is only known once a chunk exists.
                    out.setnchannels(max(1, chunk.sample_channels))
                    out.setsampwidth(chunk.sample_width)
                    out.setframerate(chunk.sample_rate)
                    pending = chunk.audio_int16_bytes
                    continue
                out.writeframes(pending)
                pending = chunk.audio_int16_bytes
            if pending is not None:
                out.writeframes(pending)


class SupertonicProvider(LocalProvider):
    """Supertonic 3: one speaker for Hindi and English, 44.1 kHz, still local.

    Four diffusion steps rather than the library default of eight: measured on
    this machine 4 steps costs ~0.75 s per sentence against ~1.4 s at 8, which
    is the difference between keeping up with a conversation and not.
    """

    engine = "supertonic"
    audio_format = "wav"

    TOTAL_STEPS = 4

    def __init__(self, model_dir: str = MODEL_DIR):
        super().__init__()
        self.root = os.path.join(model_dir, "supertonic3")
        self._tts = None

    def ready(self, voice: Dict[str, Any]) -> Tuple[bool, str]:
        if importlib.util.find_spec("supertonic") is None:
            return False, "Supertonic is not installed. Run: python3 -m pip install supertonic"
        if not os.path.exists(os.path.join(self.root, "onnx", "vocoder.onnx")):
            return False, (
                f"Supertonic 3 weights are not in {self.root}. Download them once with: "
                f'python3 -c "import supertonic; supertonic.TTS(model_dir=\'{self.root}\')"'
            )
        style = self._tts_style(voice)
        if not os.path.exists(os.path.join(self.root, "voice_styles", f"{style}.json")):
            return False, (
                f"This Supertonic model has no voice style {style}. "
                f"Pick one of M1-M5 or F1-F5, or refetch the weights into {self.root}."
            )
        return True, ""

    @staticmethod
    def _tts_style(voice: Dict[str, Any]) -> str:
        return voice.get("model") or "M1"

    def _warm(self, voice: Dict[str, Any]) -> None:
        self._ensure_tts()

    def _ensure_tts(self):
        import supertonic

        if self._tts is None:
            self._tts = supertonic.TTS(model_dir=self.root)
        return self._tts

    def _run(self, voice: Dict[str, Any], text: str, rate: str, part_path: str) -> None:
        import soundfile as sf

        tts = self._ensure_tts()
        style = self._tts_style(voice)
        wav, _duration = tts.synthesize(
            text,
            voice_style=tts.get_voice_style(style),
            lang="hi" if _has_devanagari(text) else "en",
            total_steps=self.TOTAL_STEPS,
            speed=_rate_multiplier(rate),
        )
        # The clip is still a `.part` here, so the container has to be named:
        # soundfile cannot guess WAV from a suffix that is not one.
        sf.write(part_path, wav[0], tts.sample_rate, format="WAV", subtype="PCM_16")


PROVIDERS: Dict[str, Any] = {
    "edge": EdgeProvider(),
    "piper": PiperProvider(),
    "supertonic": SupertonicProvider(),
}


def voice_is_ready(voice_id: str) -> Tuple[bool, str]:
    entry = VOICE_BY_ID.get(voice_id)
    if entry is None:
        return False, f"Unknown voice: {voice_id}"
    return PROVIDERS[entry["engine"]].ready(entry)


def pick_default_voice(preferred: Optional[str] = None) -> str:
    """The first voice in `preferred` that can actually speak, else the Edge voice.

    Reported through `/api/health` rather than assumed, so the phone can show
    which speaker it chose and why the faster one was or was not available.
    """
    for voice_id in ([preferred] if preferred else []) + AUTO_DEFAULT_ORDER:
        if voice_id in VOICE_IDS and voice_is_ready(voice_id)[0]:
            return voice_id
    return FALLBACK_VOICE


def available_voices() -> List[Dict[str, Any]]:
    """The catalog annotated with whether this machine can make audio for each voice."""
    out: List[Dict[str, Any]] = []
    for entry in VOICE_CATALOG:
        ready, reason = voice_is_ready(entry["id"])
        out.append({**entry, "available": ready, "reason": reason})
    return out


def provider_status() -> Dict[str, Any]:
    """One line per engine for `/api/health`: present or not, and what is missing."""
    summary: Dict[str, Any] = {}
    for name, provider in PROVIDERS.items():
        voices = [entry for entry in VOICE_CATALOG if entry["engine"] == name]
        usable = [entry["id"] for entry in voices if provider.ready(entry)[0]]
        first_missing = next((provider.ready(entry)[1] for entry in voices if not provider.ready(entry)[0]), "")
        summary[name] = {
            "local": name != "edge",
            "format": provider.audio_format,
            "voices": len(voices),
            "usable": len(usable),
            "available": bool(usable),
            "why_not": first_missing if not usable else "",
        }
    return summary


class VoiceEngine:
    """Disk-cached speech in parallel segments, for whichever engine a voice picks.

    Caching is keyed on (voice, rate, exact text), so a repeated question or a
    replayed answer costs no synthesis at all. Long answers are split into
    sentences: Edge synthesises them concurrently, and the local engines queue
    them, because an ONNX model already holds every core for one sentence.
    """

    def __init__(self, default_voice: Optional[str] = None, output_dir: str = "/tmp/realtime_convo_audio",
                 max_cache_entries: int = 240):
        self.default_voice = default_voice or pick_default_voice()
        self.output_dir = output_dir
        self.max_cache_entries = max_cache_entries
        # key -> (filename, stamp of the last time this clip was made or served)
        self._cache: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()
        self._last_reap = 0.0
        os.makedirs(self.output_dir, exist_ok=True)

    def list_voices(self) -> List[Dict[str, Any]]:
        return available_voices()

    async def warm_voice(self, voice_id: str) -> Tuple[bool, str]:
        """Load a local voice's model now, so no answer has to pay for the load.

        Returns whether the voice can speak and why not if it cannot. Network
        voices have nothing to load and report that plainly rather than pretending
        to be warm.
        """
        entry = self._entry(voice_id)
        provider = PROVIDERS[entry["engine"]]
        ready, reason = provider.ready(entry)
        if not ready:
            return False, reason
        warm = getattr(provider, "warm", None)
        if warm is None:
            return True, ""
        await warm(entry)
        return True, ""

    @staticmethod
    def _key(text: str, voice: str, rate: str) -> str:
        return hashlib.md5(f"{voice}|{rate}|{text}".encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _entry(voice: str) -> Dict[str, Any]:
        return VOICE_BY_ID.get(voice) or VOICE_BY_ID[FALLBACK_VOICE]

    def _path_for(self, key: str, audio_format: str = "mp3") -> str:
        return os.path.join(self.output_dir, f"tts_{key}.{audio_format}")

    def _serve_cached(self, key: str, audio_format: str = "mp3") -> Optional[Tuple[str, int]]:
        """The clip for this key if a complete one is already on disk, marked recent."""
        path = self._path_for(key, audio_format)
        try:
            size = os.path.getsize(path)
        except OSError:
            self._cache.pop(key, None)
            return None
        if size < MIN_CLIP_BYTES:
            # A stub is not a clip: drop it and let the caller re-synthesise.
            self._cache.pop(key, None)
            _remove_quietly(path)
            return None
        filename = os.path.basename(path)
        self._remember(key, filename)
        return filename, size

    def _remember(self, key: str, filename: str) -> None:
        self._cache[key] = (filename, time.time())
        self._cache.move_to_end(key)
        self._reap_parts()
        overflow = len(self._cache) - self.max_cache_entries
        if overflow <= 0:
            return
        now = self._cache[key][1]
        for stale in list(self._cache)[:overflow]:
            if now - self._cache[stale][1] < MIN_REAP_AGE_S:
                break
            stale_name, _ = self._cache.pop(stale)
            _remove_quietly(os.path.join(self.output_dir, stale_name))

    def _reap_parts(self) -> None:
        """Delete abandoned `.part` files, which only a timed-out worker leaves.

        The worker cannot be killed, so its output outlives the request that
        gave up on it; without this sweep the audio directory fills with clips
        nobody renamed.
        """
        now = time.time()
        if now - self._last_reap < MIN_REAP_AGE_S:
            return
        self._last_reap = now
        try:
            names = os.listdir(self.output_dir)
        except OSError:
            return
        for name in names:
            if not name.endswith(".part"):
                continue
            path = os.path.join(self.output_dir, name)
            try:
                if now - os.path.getmtime(path) >= MIN_REAP_AGE_S:
                    _remove_quietly(path)
            except OSError:
                continue

    async def _synthesize(self, text: str, voice: str, rate: str, key: str) -> Dict[str, Any]:
        entry = self._entry(voice)
        provider = PROVIDERS[entry["engine"]]
        audio_format = entry.get("format", provider.audio_format)
        cached = self._serve_cached(key, audio_format)
        if cached:
            filename, size = cached
            return {"status": "success", "filename": filename, "voice": voice, "engine": entry["engine"],
                    "byte_size": size, "cached": True, "audio_base64": None}

        # Content-addressed and unique per attempt: two identical requests may
        # race harmlessly, and a worker that outlives its own timeout cannot
        # write into the file another request is still streaming.
        final_path = self._path_for(key, audio_format)
        part_path = f"{final_path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part"
        ready, reason = provider.ready(entry)
        if not ready:
            return {"status": "fallback", "error": reason, "filename": "", "voice": voice,
                    "engine": entry["engine"], "byte_size": 0, "cached": False, "audio_base64": None}
        try:
            await provider.synthesize(text, entry, rate, part_path)
            size = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            if size < MIN_CLIP_BYTES:
                raise RuntimeError(f"{entry['engine']} returned only {size} bytes of audio")
            os.replace(part_path, final_path)
            self._remember(key, os.path.basename(final_path))
            inline = None
            if size <= MAX_INLINE_BYTES:
                with open(final_path, "rb") as handle:
                    inline = base64.b64encode(handle.read()).decode("utf-8")
            return {
                "status": "success",
                "filename": os.path.basename(final_path),
                "voice": voice,
                "engine": entry["engine"],
                "byte_size": size,
                "cached": False,
                "audio_base64": inline,
            }
        except Exception as error:
            logger.warning("Speech synthesis failed on %s: %s", entry["engine"], error)
            return {"status": "fallback", "error": str(error) or type(error).__name__, "filename": "",
                    "voice": voice, "engine": entry["engine"], "byte_size": 0, "cached": False,
                    "audio_base64": None}
        finally:
            _remove_quietly(part_path)

    def _resolve_voice(self, voice: Optional[str]) -> str:
        return voice if voice in VOICE_IDS else self.default_voice

    async def text_to_speech(self, text: str, voice: Optional[str] = None, file_prefix: str = "reply",
                             rate: str = DEFAULT_RATE) -> Dict[str, Any]:
        """Single clip. `file_prefix` is accepted for API compatibility; files are content-addressed."""
        selected_voice = self._resolve_voice(voice)
        clean_text = " ".join((text or "").split())
        if not clean_text:
            clean_text = "I am listening."
        key = self._key(clean_text, selected_voice, rate)
        return await self._synthesize(clean_text, selected_voice, rate, key)

    async def text_to_speech_segments(self, segments: List[str], voice: Optional[str] = None,
                                      rate: str = DEFAULT_RATE) -> Dict[str, Any]:
        """Synthesise every spoken segment and return them in reading order."""
        selected_voice = self._resolve_voice(voice)
        entry = self._entry(selected_voice)
        cleaned = [" ".join(segment.split()) for segment in segments if segment.strip()]
        if not cleaned:
            cleaned = ["I am listening."]

        results = await asyncio.gather(
            *(self._synthesize(text, selected_voice, rate, self._key(text, selected_voice, rate)) for text in cleaned)
        )
        ok = [item for item in results if item.get("status") == "success"]
        # `partial` matters: speaking three of five sentences is not a success,
        # and the caller needs to say so rather than cut the answer short silently.
        status = "success" if len(ok) == len(results) else ("partial" if ok else "fallback")
        return {
            "status": status,
            "voice": selected_voice,
            "engine": entry["engine"],
            "segments": [
                {"text": cleaned[i], "filename": item.get("filename", ""), "cached": bool(item.get("cached")),
                 "byte_size": item.get("byte_size", 0)}
                for i, item in enumerate(results) if item.get("status") == "success"
            ],
            "errors": [item.get("error") for item in results if item.get("status") != "success"],
            "requested": len(results),
            "cached_any": any(bool(item.get("cached")) for item in results),
        }
