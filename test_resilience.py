"""Failure-mode harness: every way this application could fall over, driven on purpose.

Self-contained and offline. The PDFs are synthesised here, speech is faked at the
engine boundary and the model is replaced by a scripted one, so each check proves
what the *server and agent do when a part of the system fails* rather than what
the network happens to allow today.

    python3 test_resilience.py
"""

import asyncio
import os
import re
import sys
import time
import types
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import voice_engine as voice_engine_module  # noqa: E402
from agent import GroundedLlm, LlmAnswerError  # noqa: E402

failures: List[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> bool:
    global checks
    checks += 1
    print(f"{'PASS' if ok else 'FAIL'} {label}" + (f"\n     {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(label)
    return ok


# --------------------------------------------------------------------- fixtures

def make_pdf(lines: List[str]) -> bytes:
    """One page of selectable text, assembled by hand so no sample file is needed."""
    content = "BT /F1 12 Tf 72 720 Td 16 TL " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> "
        "/Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets: List[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return out


DOC_LINES = [
    "The login form takes a username and a password for the saucedemo site.",
    "Use the username tomsmith and the secret password to sign in today.",
    "Run pip install selenium pytest to prepare the laboratory environment.",
]


def http_detail(response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:120]
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("error") or payload)[:200]
    return str(payload)[:200]


# ------------------------------------------------------------------- A uploads

def section_uploads(client) -> None:
    print("\nA. Uploads that must be refused, and must not destroy a working index")
    print("-" * 78)

    import tempfile
    import server

    # Isolate test uploads from live user documents in UPLOAD_DIR
    original_upload_dir = server.UPLOAD_DIR
    original_rag = server.agent.rag
    original_upload = dict(server.current_upload)
    test_temp_dir = tempfile.TemporaryDirectory()
    server.UPLOAD_DIR = test_temp_dir.name

    try:
        r = client.post("/api/upload_pdf", files={"file": ("notes.pdf", b"just text, not a pdf", "application/pdf")})
        check(r.status_code == 415, f"non-PDF bytes named .pdf are refused (got {r.status_code})", http_detail(r))

        r = client.post("/api/upload_pdf", files={"file": ("empty.pdf", b"", "application/pdf")})
        check(r.status_code == 400, f"empty upload is refused (got {r.status_code})", http_detail(r))

        r = client.post("/api/upload_pdf", files={"file": ("report.txt", b"hello", "text/plain")})
        check(r.status_code == 400, f"non-PDF filename is refused (got {r.status_code})", http_detail(r))

        original_cap = server.MAX_UPLOAD_BYTES
        server.MAX_UPLOAD_BYTES = 4096
        big = b"%PDF-1.4\n" + b"x" * 200000
        r = client.post("/api/upload_pdf", files={"file": ("big.pdf", big, "application/pdf")})
        server.MAX_UPLOAD_BYTES = original_cap
        check(r.status_code == 413, f"upload over the size cap is refused (got {r.status_code})", http_detail(r))

        r = client.post("/api/upload_pdf", files={"file": ("broken.pdf", b"%PDF-1.4\nnot a real body\n", "application/pdf")})
        check(r.status_code in (415, 422), f"unreadable PDF is refused with a reason (got {r.status_code})", http_detail(r))

        # A text-less PDF is a scanned page, and an empty index is useless: say so now.
        r = client.post("/api/upload_pdf", files={"file": ("scan.pdf", make_pdf([]), "application/pdf")})
        check(r.status_code == 422, f"PDF with no extractable text is refused (got {r.status_code})", http_detail(r))

        # A good upload, then two bad ones: the live index must still answer.
        r = client.post("/api/upload_pdf", files={"file": ("lab.pdf", make_pdf(DOC_LINES), "application/pdf")})
        ok_upload = r.status_code == 200 and r.json()["stats"]["total_chunks"] > 0
        detail = http_detail(r)
        if ok_upload:
            stats = r.json()["stats"]
            detail = (f"{stats['total_chunks']} chunks in {stats['index_ms']}ms, "
                      f"{stats['chunking']['text_preserved_pct']}% of the words indexed")
        check(ok_upload, "synthetic PDF indexes with chunks to search", detail)

        client.post("/api/upload_pdf", files={"file": ("scan.pdf", make_pdf([]), "application/pdf")})
        client.post("/api/upload_pdf", files={"file": ("broken.pdf", b"%PDF-1.4\njunk\n", "application/pdf")})
        health = client.get("/api/health").json()
        doc = client.get("/api/document").json()
        check(
            health["total_chunks"] > 0 and doc["loaded"] and doc["filename"] == "lab.pdf",
            "two failed uploads left the good index serving questions",
            f"health={health['total_chunks']} chunks, document={doc.get('filename')}",
        )

        r = client.post("/api/ask_text", json={"query": "which username does the login form take"})
        payload = r.json()
        check(
            r.status_code == 200 and "login" in payload["answer"] and payload["sources"] == [1]
            and not payload["is_out_of_context"],
            "the surviving index still answers from the page it retrieved",
            f"{r.status_code} answer={payload.get('answer')!r} sources={payload.get('sources')}",
        )
        r = client.post("/api/ask_text", json={"query": "which pip packages do I install"})
        pip_answer = r.json()["answer"]
        check(
            "selenium" in pip_answer and "pytest" in pip_answer
            and not any(word in pip_answer for word in ("prepare", "laboratory", " today")),
            "the pip answer lists packages, not the prose that followed them",
            repr(pip_answer),
        )
    finally:
        server.UPLOAD_DIR = original_upload_dir
        server.agent.rag = original_rag
        server.current_upload.clear()
        server.current_upload.update(original_upload)
        test_temp_dir.cleanup()

    # A list longer than what the answer is willing to read out must report the
    # remainder: a dropped package is an install that fails later.
    import tempfile

    from agent import ConversationalAgent
    from rag_engine import PDFRAGEngine

    many = [
        "Run pip install selenium pytest openpyxl lxml pillow requests urllib3 html5lib "
        "beautifulsoup4 lxml_html_clean filelock to prepare the laboratory environment."
    ]
    long_rag = PDFRAGEngine()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(make_pdf(many))
        many_path = handle.name
    try:
        long_rag.load_pdf(many_path)
    finally:
        os.remove(many_path)
    listed = many[0].split("pip install", 1)[1].split(" to prepare")[0].split()
    pip_agent = ConversationalAgent(rag_engine=long_rag, llm=BoomLlm("m", "http://x/y", {}, "m", "openai"))
    long_answer = pip_agent.process_query("which pip packages do I install")["answer"]
    check(
        all(name in long_answer for name in listed[:8])
        and f"plus {len(listed) - 8} more on that line" in long_answer
        and "laboratory" not in long_answer,
        f"a {len(listed)}-package line names {min(8, len(listed))} and counts the rest",
        repr(long_answer),
    )


# -------------------------------------------------------------- B speech files

class FakeEngine:
    """Speech that reports whatever the scenario needs, without touching the network."""

    def __init__(self, status: str, segments: List[Dict[str, Any]], errors: Optional[List[str]] = None):
        self.status_value = status
        self.segments = segments
        self.errors = errors or []
        self.requested = len(segments) + len(self.errors)

    async def text_to_speech_segments(self, texts, voice=None, rate=None):
        return {
            "status": self.status_value,
            "voice": voice or "en-IN-PrabhatNeural",
            "engine": "edge",
            "segments": self.segments,
            "errors": list(self.errors),
            "requested": self.requested,
            "cached_any": False,
        }


def section_speech_http(client) -> None:
    print("\nB. Speech that only partly works must be reported, not hidden")
    print("-" * 78)
    import server

    original = server.voice_engine
    try:
        server.voice_engine = FakeEngine("fallback", [], errors=["Edge-TTS did not answer within 20s"])
        r = client.post("/api/speak", json={"texts": ["Open the login form."], "voice": "en-IN-PrabhatNeural"})
        check(r.status_code == 502, "every segment failing is a 502, not an empty success", http_detail(r))

        server.voice_engine = FakeEngine(
            "partial",
            [{"text": "Open the login form.", "filename": "tts_head.mp3", "cached": False, "byte_size": 4096}],
            errors=["Edge-TTS returned only 0 bytes of audio"],
        )
        r = client.post("/api/speak", json={"texts": ["Open the login form.", "Then assert the title."],
                                           "voice": "en-IN-PrabhatNeural"})
        payload = r.json() if r.status_code < 500 else {}
        check(
            r.status_code == 200 and payload.get("status") == "partial" and payload.get("errors"),
            "one clip of two comes back as 200 + status partial + the reason",
            f"{r.status_code} {str(payload)[:180]}",
        )
        check(
            payload.get("requested") == 2 and len(payload.get("segments", [])) == 1,
            "partial says how much was asked for and how much arrived",
            str(payload),
        )
        check(
            payload.get("speech_engine") == "edge",
            "the phone is told which engine made the audio, so 0.2s and 2s are explainable",
            str(payload)[:120],
        )

        server.voice_engine = FakeEngine(
            "partial",
            [{"text": "Open the login form.", "filename": "tts_head.mp3", "cached": False, "byte_size": 4096}],
            errors=["no audio"],
        )
        r = client.post("/api/ask_voice", json={"query": "which username does the login form take"})
        check(
            r.status_code == 200 and r.json()["status"] == "partial" and r.json()["answer"],
            "ask_voice carries partial through with the text answer intact",
            f"{r.status_code} {str(r.json())[:180]}" if r.status_code == 200 else http_detail(r),
        )
    finally:
        server.voice_engine = original


def section_clip_files() -> None:
    print("\nC. Clip integrity on disk")
    print("-" * 78)
    import shutil
    import tempfile

    from voice_engine import VoiceEngine

    audio_dir = tempfile.mkdtemp(prefix="convo_clips_")
    # Pinned to the network voice on purpose: this section is about how a clip
    # lands on disk, and the stub below replaces Edge.
    engine = VoiceEngine(default_voice="en-IN-PrabhatNeural", output_dir=audio_dir, max_cache_entries=3)
    original_tts = voice_engine_module.edge_tts
    original_timeout = voice_engine_module.SYNTHESIS_TIMEOUT_S
    try:
        key = engine._key("anything", engine.default_voice, "+8%")
        path = engine._path_for(key)
        with open(path, "wb") as handle:
            handle.write(b"stub")
        engine._remember(key, os.path.basename(path))
        check(engine._serve_cached(key) is None, "a 4-byte stub is never served as a clip")
        check(not os.path.exists(path), "the stub file is removed so the clip gets re-made")

        class Good:
            def __init__(self, text, voice, rate=None):
                pass

            async def save(self, target):
                with open(target, "wb") as handle:
                    handle.write(b"ok" * 300)

        voice_engine_module.edge_tts = types.SimpleNamespace(Communicate=Good)
        result = asyncio.run(engine.text_to_speech("Say this in full voice please.", rate="+8%"))
        check(result["status"] == "success" and result["byte_size"] == 600, "a complete clip is stored and returned",
              str(result)[:160])
        again = asyncio.run(engine.text_to_speech("Say this in full voice please.", rate="+8%"))
        check(again["cached"] and again["byte_size"] == 600, "the same sentence is served from disk, no network")

        class Empty:
            def __init__(self, text, voice, rate=None):
                pass

            async def save(self, target):
                open(target, "wb").close()

        voice_engine_module.edge_tts = types.SimpleNamespace(Communicate=Empty)
        result = asyncio.run(engine.text_to_speech("Nothing at all comes back here.", rate="+8%"))
        empty_key = engine._key("Nothing at all comes back here.", engine.default_voice, "+8%")
        check(result["status"] == "fallback", "a zero-byte save is a failure, not an empty success", str(result)[:160])
        check(not os.path.exists(engine._path_for(empty_key)), "an empty clip is never committed to the cache path")

        class Stalled:
            def __init__(self, text, voice, rate=None):
                pass

            async def save(self, target):
                with open(target, "wb") as handle:
                    handle.write(b"half-written" * 40)
                await asyncio.sleep(30)

        voice_engine_module.edge_tts = types.SimpleNamespace(Communicate=Stalled)
        voice_engine_module.SYNTHESIS_TIMEOUT_S = 0.2
        started = time.perf_counter()
        result = asyncio.run(engine.text_to_speech("This synthesis will be cut short deliberately.", rate="+8%"))
        waited = time.perf_counter() - started
        stalled_key = engine._key("This synthesis will be cut short deliberately.", engine.default_voice, "+8%")
        check(result["status"] == "fallback" and "did not answer within" in result["error"],
              "a stalled endpoint is abandoned instead of hanging the request", str(result)[:160])
        check(waited < 5.0, f"the request returns in {waited:.1f}s instead of hanging the phone")
        check(not os.path.exists(engine._path_for(stalled_key)), "the half clip is not committed")
        check(not [name for name in os.listdir(audio_dir) if name.endswith(".part")], "no .part file is left behind")

        # Eviction must not reap a clip that is still being downloaded or replayed.
        voices = VoiceEngine(default_voice="en-IN-PrabhatNeural", output_dir=audio_dir, max_cache_entries=3)
        names = {}
        for index in range(3):
            text = f"Eviction probe number {index} spoken aloud"
            entry_key = voices._key(text, voices.default_voice, "+8%")
            clip = voices._path_for(entry_key)
            with open(clip, "wb") as handle:
                handle.write(b"z" * 900)
            voices._remember(entry_key, os.path.basename(clip))
            names[index] = (entry_key, clip)
        # Backdate only the oldest one so it becomes the sole reap candidate.
        old_key, old_path = names[0]
        voices._cache[old_key] = (os.path.basename(old_path), time.time() - voice_engine_module.MIN_REAP_AGE_S - 5)
        extra = voices._key("Eviction probe pushes the cache over its limit", voices.default_voice, "+8%")
        with open(voices._path_for(extra), "wb") as handle:
            handle.write(b"z" * 900)
        voices._remember(extra, os.path.basename(voices._path_for(extra)))
        check(not os.path.exists(old_path), "the stale clip beyond the limit is reaped")
        check(os.path.exists(names[1][1]) and os.path.exists(names[2][1]),
              "young clips are kept on disk even when the cache is over its limit")
    finally:
        voice_engine_module.edge_tts = original_tts
        voice_engine_module.SYNTHESIS_TIMEOUT_S = original_timeout
        shutil.rmtree(audio_dir, ignore_errors=True)


# ----------------------------------------------------------- F local speech engines

class StubLocal:
    """A local engine that can be told to work, to be missing, or to fail."""

    engine = "piper"
    audio_format = "wav"
    parallel = False

    def __init__(self, ready: bool = True, failure: str = ""):
        self.ready_value = ready
        self.failure = failure
        self.calls: List[str] = []

    def ready(self, voice):
        if self.ready_value:
            return True, ""
        return False, f"{voice['id']} is not on this machine"

    async def synthesize(self, text, voice, rate, part_path):
        self.calls.append(text)
        if self.failure:
            raise RuntimeError(self.failure)
        with open(part_path, "wb") as handle:
            handle.write(b"RIFF" + b"\x00" * 900)


def section_local_speech(client) -> None:
    print("\nF. Local speech engines: the real-time path keeps the same guarantees")
    print("-" * 78)
    import shutil
    import tempfile

    import voice_engine as ve
    from voice_engine import (VoiceEngine, _has_devanagari, _rate_multiplier, provider_status, voice_is_ready)

    audio_dir = tempfile.mkdtemp(prefix="convo_local_")
    original_piper = ve.PROVIDERS["piper"]
    try:
        engine = VoiceEngine(default_voice="en_US-ryan-medium", output_dir=audio_dir)

        stub = StubLocal(ready=False)
        ve.PROVIDERS["piper"] = stub
        result = asyncio.run(engine.text_to_speech("The login form takes username tomsmith.", rate="+8%"))
        check(
            result["status"] == "fallback" and "not on this machine" in result["error"] and not result["filename"],
            "a voice whose model is missing fails with the reason instead of a silent swap",
            str(result)[:170],
        )
        check(not os.listdir(audio_dir), "nothing is committed to the cache when the engine cannot speak",
              str(os.listdir(audio_dir)))

        stub = StubLocal()
        ve.PROVIDERS["piper"] = stub
        result = asyncio.run(engine.text_to_speech("The login form takes username tomsmith.", rate="+8%"))
        local_name = result.get("filename", "")
        check(
            result["status"] == "success" and local_name.endswith(".wav") and result["engine"] == "piper",
            "a local clip is a WAV and says which engine made it",
            str(result)[:170],
        )
        again = asyncio.run(engine.text_to_speech("The login form takes username tomsmith.", rate="+8%"))
        check(again["cached"] and again["filename"] == local_name and len(stub.calls) == 1,
              "the second ask is served from disk and never re-synthesised", str(again)[:150])

        # The same words through another engine must not collide in the cache.
        edge_engine = VoiceEngine(default_voice="en-IN-PrabhatNeural", output_dir=audio_dir)
        edge_key = edge_engine._key("The login form takes username tomsmith.", "en-IN-PrabhatNeural", "+8%")
        local_key = engine._key("The login form takes username tomsmith.", "en_US-ryan-medium", "+8%")
        check(
            edge_key != local_key
            and edge_engine._path_for(edge_key, "mp3") != engine._path_for(local_key, "wav"),
            "two engines cannot serve each other's clips out of one cache",
        )

        stub = StubLocal(failure="Piper did not finish within 20s")
        ve.PROVIDERS["piper"] = stub
        parts = asyncio.run(engine.text_to_speech_segments(
            ["First sentence to speak.", "Second sentence to speak."], rate="+8%"))
        check(parts["status"] == "fallback" and len(parts["errors"]) == 2,
              "every segment failing is a fallback with one reason each", str(parts)[:170])

        # One of two: partial, not a shorter answer that sounds complete.
        class Flaky:
            engine = "piper"
            audio_format = "wav"
            parallel = False
            calls = 0

            def ready(self, voice):
                return True, ""

            async def synthesize(self, text, voice, rate, part_path):
                Flaky.calls += 1
                if Flaky.calls % 2:
                    with open(part_path, "wb") as handle:
                        handle.write(b"RIFF" + b"\x00" * 900)
                    return
                raise RuntimeError("the model refused the second sentence")

        ve.PROVIDERS["piper"] = Flaky()
        mixed = asyncio.run(engine.text_to_speech_segments(
            ["Sentence one speaks fine.", "Sentence two does not."], rate="+8%"))
        check(
            mixed["status"] == "partial" and len(mixed["segments"]) == 1 and mixed["errors"],
            "half a local answer is reported as partial with the reason",
            str(mixed)[:200],
        )

        # A timed-out worker leaves a .part behind; the reaper must take it.
        # A fresh engine is used because an instance sweeps once per reap age,
        # and the one above has already swept.
        zombie = os.path.join(audio_dir, "tts_zombieclip.wav.999.abc123.part")
        with open(zombie, "wb") as handle:
            handle.write(b"RIFF" + b"\x00" * 500)
        fresh = os.path.join(audio_dir, "tts_freshclip.wav.999.def456.part")
        with open(fresh, "wb") as handle:
            handle.write(b"RIFF" + b"\x00" * 500)
        old = time.time() - ve.MIN_REAP_AGE_S - 10
        os.utime(zombie, (old, old))
        ve.PROVIDERS["piper"] = StubLocal()
        sweeper = VoiceEngine(default_voice="en_US-ryan-medium", output_dir=audio_dir)
        asyncio.run(sweeper.text_to_speech("A sentence that reaps the abandoned part file.", rate="+8%"))
        check(not os.path.exists(zombie), "an abandoned part file older than the reap age is deleted")
        check(os.path.exists(fresh), "a part file still being written is left alone",
              "the sweep only takes settled files")

        # Rate and script detection, the two knobs the client cannot see.
        check(
            _rate_multiplier("+8%") == 1.08 and _rate_multiplier("-20%") == 0.8
            and _rate_multiplier("nonsense") == 1.0 and _rate_multiplier("+900%") == 2.0,
            "one rate string maps onto both engines, and a bad one cannot make a mute clip",
            f"{_rate_multiplier('+8%')} {_rate_multiplier('-20%')} {_rate_multiplier('nonsense')}",
        )
        check(
            _has_devanagari("पहले pip install selenium चलाएँ") and not _has_devanagari("pip install selenium"),
            "Devanagari is spoken as Hindi and Latin script as English by the bilingual voice",
        )

        # Availability is reported, not assumed.
        ve.PROVIDERS["piper"] = StubLocal(ready=False)
        status = provider_status()
        check(
            status["piper"]["available"] is False and status["piper"]["why_not"]
            and status["edge"]["available"] is True and status["edge"]["local"] is False,
            "health can say which engine is missing and why",
            str(status)[:200],
        )
        voices = engine.list_voices()
        flagged = [v for v in voices if v["id"] == "en_US-ryan-medium"][0]
        check(
            flagged["available"] is False and "not on this machine" in flagged["reason"],
            "an unusable voice is still listed but marked with the reason",
            str(flagged)[:170],
        )
        check(voice_is_ready("no.such-voice")[0] is False, "an unknown voice id is refused, not guessed")
    finally:
        ve.PROVIDERS["piper"] = original_piper
        shutil.rmtree(audio_dir, ignore_errors=True)

    # The real thing, if this machine has it: the numbers are only worth quoting
    # when they came out of the model the app will actually use.
    import server

    usable = [v for v in voice_engine_module.available_voices() if v["available"] and v["local"]]
    if not usable:
        print("   (no local model present - real local synthesis skipped)")
        return
    wav_dir = tempfile.mkdtemp(prefix="convo_real_")
    probe_clip = os.path.join(server.AUDIO_DIR, "probe_local_clip.wav")
    try:
        real = VoiceEngine(default_voice=usable[0]["id"], output_dir=wav_dir)
        segments = ["Open the login form and type the username tomsmith.", "Then assert that the page title is correct."]
        asyncio.run(real.text_to_speech_segments(segments))  # load the model and warm the cache
        # Timed on a sentence that has never been heard: a cache hit would be a
        # flattering number rather than the one the phone actually waits for.
        fresh_sentence = "Practical four needs webdriver manager before pytest can run."
        started = time.perf_counter()
        head = asyncio.run(real.text_to_speech(fresh_sentence))
        head_ms = (time.perf_counter() - started) * 1000
        check(
            head["status"] == "success" and not head["cached"] and head_ms < 1500,
            f"a sentence nobody has heard speaks in {head_ms:.0f} ms with no network ({usable[0]['id']})",
            str(head)[:150],
        )
        # Warming has to mean "the model is in memory", not "one sentence happened
        # to synthesise", because a warm cache hides the load cost entirely.
        piper_ready = [v for v in usable if v["engine"] == "piper"]
        if piper_ready:
            warm_id = piper_ready[0]["id"]
            ve.PROVIDERS["piper"]._voices.pop(warm_id, None)
            warmed, why = asyncio.run(real.warm_voice(warm_id))
            check(
                warmed and warm_id in ve.PROVIDERS["piper"]._voices,
                f"warming {warm_id} loads the model itself rather than speaking a line",
                why,
            )
        network = asyncio.run(real.warm_voice("en-IN-PrabhatNeural"))
        check(
            network[0] and network[1] == "",
            "a network voice has nothing to load, and starting up does not reach the network for it",
            str(network),
        )
        if head["status"] == "success":
            shutil.copyfile(os.path.join(wav_dir, head["filename"]), probe_clip)
            r = client.get("/audio/probe_local_clip.wav")
            check(
                r.status_code == 200 and r.headers.get("content-type", "").startswith("audio/wav"),
                "a WAV clip reaches the phone as audio/wav, not as audio/mpeg",
                f"{r.status_code} {r.headers.get('content-type')}",
            )
            r = client.get(f"/audio/{head['filename'].replace('.wav', '.mp3')}")
            check(r.status_code == 404, "a clip is only ever served under the name it was written as",
                  str(r.status_code))

        from voice_engine import PROVIDERS, VOICE_BY_ID

        entry = VOICE_BY_ID["supertonic-M1"]
        if PROVIDERS["supertonic"].ready(entry)[0]:
            missing_style = {**entry, "model": "Z9"}
            ready, reason = PROVIDERS["supertonic"].ready(missing_style)
            check(
                not ready and "Z9" in reason,
                "a voice style the model does not ship is refused before a thread is spent on it",
                reason,
            )
            check(
                provider_status()["supertonic"]["usable"] > 0,
                "one unusable style is that voice's problem, not the engine's",
                str(provider_status()["supertonic"]),
            )
    finally:
        shutil.rmtree(wav_dir, ignore_errors=True)
        try:
            os.remove(probe_clip)
        except OSError:
            pass


# ------------------------------------------------------------------ D the model

class BoomLlm(GroundedLlm):
    def answer(self, query, passages, timeout=25.0) -> str:
        raise RuntimeError("the model exploded in a way nobody declared")


class WrongPagesLlm(GroundedLlm):
    def answer(self, query, passages, timeout=25.0) -> str:
        return "The username is tomsmith [Page 99]. It also said [Page 9]. [Page 1]"


class NoMarkerLlm(GroundedLlm):
    def answer(self, query, passages, timeout=25.0) -> str:
        return "The username is tomsmith, taken straight from the passage."


class MarkersOnlyLlm(GroundedLlm):
    def answer(self, query, passages, timeout=25.0) -> str:
        return "[Page 99]"


def section_model() -> None:
    print("\nD. The configured model going wrong in every shape it can")
    print("-" * 78)
    from agent import ConversationalAgent
    from rag_engine import PDFRAGEngine

    import tempfile

    rag = PDFRAGEngine()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(make_pdf(DOC_LINES))
        sample_path = handle.name
    try:
        rag.load_pdf(sample_path)
    finally:
        os.remove(sample_path)
    check(len(rag.chunks) > 0, f"model checks have an index to work with ({len(rag.chunks)} chunks)")

    query = "which username does the login form take"

    agent = ConversationalAgent(rag_engine=rag, llm=BoomLlm("b", "http://x/y", {}, "b", "openai"))
    result = agent.process_query(query)
    check(
        "login" in result["answer"] and result["sources"] and not result["is_out_of_context"],
        "an undeclared crash in the model still yields a grounded, cited answer",
        str(result)[:220],
    )
    check(
        "RuntimeError" in result["engine"] and "nobody declared" in result["engine"],
        "and the engine label names the real failure instead of hiding it",
        result["engine"],
    )

    agent = ConversationalAgent(rag_engine=rag, llm=WrongPagesLlm("w", "http://x/y", {}, "w", "openai"))
    result = agent.process_query(query)
    markers = [int(n) for n in re.findall(r"\[Page\s+(\d+)\]", result["answer"], re.I)]
    pages = set(result["sources"])
    check(
        99 not in markers and 9 not in markers,
        "a page number the retrieval never returned is stripped from the answer",
        f"answer={result['answer']!r} sources={sorted(pages)}",
    )
    check(
        bool(markers) and set(markers) <= pages and all(page in {c.page_num for c in rag.chunks} for page in markers),
        "every marker that survives is a page the sources field also reports",
        f"markers={markers} sources={sorted(pages)}",
    )

    agent = ConversationalAgent(rag_engine=rag, llm=NoMarkerLlm("n", "http://x/y", {}, "n", "openai"))
    result = agent.process_query(query)
    check(
        result["sources"] and "[Page" in result["answer"],
        "an answer with no citation is re-cited to a page that was actually retrieved",
        f"answer={result['answer']!r} sources={result['sources']}",
    )

    agent = ConversationalAgent(rag_engine=rag, llm=MarkersOnlyLlm("m", "http://x/y", {}, "m", "openai"))
    result = agent.process_query(query)
    check(
        result["answer"].strip() and not result["answer"].startswith("[Page"),
        "a model that returns only a fabricated page marker is replaced by the extractive answer",
        repr(result["answer"][:120]),
    )
    check("extractive" in result["engine"], f"and the engine says so: {result['engine']}")

    openai = GroundedLlm("o", "http://x/y", {}, "o", "openai")
    gemini = GroundedLlm("g", "http://x/y", {}, "g", "gemini")
    shapes = [
        (openai, []), (openai, {}), (openai, {"choices": []}),
        (openai, {"choices": [{"message": {}}]}),
        (openai, {"choices": [{"message": None}]}),
        (openai, {"choices": [{"message": {"content": ["not", "a", "dict"]}}]}),
        (openai, {"choices": "nope"}),
        (gemini, {"candidates": "nope"}),
        (gemini, {"candidates": []}),
        (gemini, {"candidates": [{"content": None}]}),
        (gemini, {"candidates": [{"content": {"parts": "nope"}}]}),
        (gemini, {"candidates": [{"content": {"parts": [{"nope": 1}]}}]}),
        (gemini, [1, 2, 3]),
    ]
    broken: List[str] = []
    for engine, payload in shapes:
        try:
            value = engine._extract_text(payload)
            if value is not None and not isinstance(value, str):
                broken.append(f"{payload} -> {type(value).__name__}")
        except Exception as error:
            broken.append(f"{str(payload)[:40]} -> {type(error).__name__}: {error}")
    check(not broken, f"{len(shapes)} malformed model payloads are absorbed, none raises", "; ".join(broken[:3]))

    good = gemini._extract_text({"candidates": [{"content": {"parts": [{"text": "Two "}, {"text": "clips"}]}}]})
    check(good == "Two clips", f"a well-formed gemini reply still reads through: {good!r}")

    llm = GroundedLlm.from_env()
    check(llm is None or isinstance(llm, GroundedLlm), "from_env returns a client or nothing at all")


# ---------------------------------------------------------------- E no document

def section_no_document(client) -> None:
    print("\nE. Nothing indexed, which is not the same as nothing found")
    print("-" * 78)
    import server
    from rag_engine import PDFRAGEngine
    from agent import ConversationalAgent

    original_agent = server.agent
    try:
        server.agent = ConversationalAgent(rag_engine=PDFRAGEngine(), llm=None)
        r = client.post("/api/ask_text", json={"query": "what is the username"})
        payload = r.json()
        check(
            r.status_code == 200 and payload["no_document"] is True and payload["is_out_of_context"] is False,
            "an empty index answers with its own signal, not the out-of-context one",
            str({k: payload[k] for k in ("no_document", "is_out_of_context", "engine")})[:200],
        )
        check("upload" in payload["answer"].lower(), f"and asks for a document: {payload['answer']!r}")

        health = client.get("/api/health").json()
        doc = client.get("/api/document").json()
        check(health["current_document"] == "None" and health["total_chunks"] == 0, "health reports no document")
        check(doc == {"loaded": False}, f"and /api/document says so plainly: {doc}")

        r = client.post("/api/speak", json={"texts": ["   "], "voice": "en-IN-PrabhatNeural"})
        check(r.status_code == 400, f"blank speech text is refused (got {r.status_code})", http_detail(r))

        r = client.post("/api/ask_text", json={"query": "   "})
        check(r.status_code == 400, f"blank question is refused (got {r.status_code})", http_detail(r))

        r = client.get("/audio/%2e%2e%2f%2e%2e%2fetc%2fpasswd")
        check(
            r.status_code in (400, 404) and "root:" not in r.text,
            f"an encoded traversal path for audio gives no file (got {r.status_code})",
            http_detail(r),
        )

        r = client.get("/audio/tts_missing.mp3")
        check(r.status_code == 404, "a clip that is no longer on disk is a clean 404 for the phone to work around")

        def crash(query, history=None):
            raise ZeroDivisionError("something inside the agent misbehaved")

        original_process = server.agent.process_query
        try:
            server.agent.process_query = crash
            r = client.post("/api/ask_text", json={"query": "trigger the unforeseen bug"})
            check(
                r.status_code == 500 and "unexpected problem" in http_detail(r) and "ZeroDivisionError" in http_detail(r),
                "an unplanned crash still returns readable JSON, not a bare 500 page",
                f"{r.status_code} {http_detail(r)}",
            )
        finally:
            server.agent.process_query = original_process
    finally:
        server.agent = original_agent


def main() -> int:
    from fastapi.testclient import TestClient
    import server

    # The 500 path is under test, so the client must see the response the phone
    # would see rather than the exception re-raised into this harness.
    client = TestClient(server.app, raise_server_exceptions=False)
    print("=" * 78)
    print("Failure-mode harness - uploads, speech, model, empty index, unplanned crashes")
    print("=" * 78)

    section_uploads(client)
    section_speech_http(client)
    section_clip_files()
    section_local_speech(client)
    section_model()
    section_no_document(client)

    print("\n" + "=" * 78)
    print(f"Resilience: {checks - len(failures)}/{checks} checks passed, {len(failures)} failure(s)")
    if failures:
        for item in failures:
            print(f"  - {item}")
        return 1
    print("Nothing fell over: every planned failure mode is refused, reported or worked around.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
