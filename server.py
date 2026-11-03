"""FastAPI service for the voice-to-voice PDF conversation engine.

Two answering calls exist on purpose:

  * `POST /api/ask_text` returns the grounded answer as soon as retrieval and
    synthesis are done, with no speech work on the critical path;
  * `POST /api/speak` turns the spoken copy into per-sentence clips (WAV from
    the local engines, MP3 from the network one), from a content-addressed
    cache. The local engines run on this machine, so nothing but the text
    crosses the network.

The client therefore shows text immediately and starts playing the first
sentence while the rest are still being synthesised. `POST /api/ask_voice` keeps
the one-call path for the browser demo.
"""

import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from agent import ConversationalAgent, GroundedLlm
from rag_engine import PDFRAGEngine, split_sentences
from voice_engine import VOICE_CATALOG, VoiceEngine, available_voices, provider_status

logger = logging.getLogger("convo.server")


async def warm_local_speech() -> None:
    """Load each local engine's model while the server is starting.

    A Piper or Supertonic session costs 0.6-1.4 s to create. Doing it here rather
    than inside the startup hook keeps `/api/health` answering in milliseconds, so
    a phone polling the address never waits on a model. A voice nobody has used
    yet still pays its own one-off load the first time it is asked to speak.
    """
    started = time.perf_counter()
    wanted: List[str] = []

    def want(voice_id: str) -> None:
        if voice_id and voice_id not in wanted:
            wanted.append(voice_id)

    want(voice_engine.default_voice)
    seen: set = set()
    for entry in available_voices():
        if entry["available"] and entry["local"] and entry["engine"] not in seen:
            seen.add(entry["engine"])
            want(entry["id"])

    for voice_id in wanted:
        try:
            ready, reason = await voice_engine.warm_voice(voice_id)
            if not ready:
                logger.warning("speech warm-up for %s: %s", voice_id, reason)
        except Exception:  # a voice that cannot warm up is never a reason to not start
            logger.exception("speech warm-up for %s raised", voice_id)
    logger.info("speech engines warmed in %.1fs", time.perf_counter() - started)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(warm_local_speech())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Real-Time Voice-to-Voice PDF Conversation API", version="1.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp/realtime_convo_uploads"
AUDIO_DIR = "/tmp/realtime_convo_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Phone pickers hand over lab manuals of a few MB; the cap is there so one
# careless upload cannot fill the disk the audio cache lives on.
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

# The agent owns the live index: swapping it is one attribute assignment, so a
# question already being answered never sees a half-built index.
agent = ConversationalAgent(rag_engine=PDFRAGEngine(), llm=GroundedLlm.from_env())
# With no voice configured the engine picks the fastest speaker this machine can
# actually run (local models first, Edge as the network option) and reports the
# choice through /api/health rather than keeping it to itself.
voice_engine = VoiceEngine(output_dir=AUDIO_DIR, default_voice=os.getenv("CONVO_DEFAULT_VOICE") or None)

PAGE_CITATION_RE = re.compile(r"\s*\[Page\s+\d+\]", re.I)
_MARKDOWN_RE = re.compile(r"\*{1,3}|`{1,3}|#{1,6}\s")
SPEECH_SEGMENT_MAX_CHARS = 190

# Local engines write WAV and Edge writes MP3; a wrong media type is how a clip
# ends up refused by the player.
AUDIO_MEDIA_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav"}


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


@app.exception_handler(Exception)
async def unhandled_error(request: Request, error: Exception):
    """Turn a bug nobody planned for into a readable failure on the phone.

    Without this the client receives a bare `Internal Server Error` page and the
    traceback exists only in whichever log the operator happens to be watching.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": f"The server hit an unexpected problem: {type(error).__name__}: {error}"[:300]},
    )


class Turn(BaseModel):
    query: str
    answer: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    voice: Optional[str] = None
    history: Optional[List[Turn]] = None


class SpeakRequest(BaseModel):
    texts: List[str]
    voice: Optional[str] = None


def speech_text_of(answer: str) -> str:
    """Spoken copy: page markers and markdown read out loud are noise."""
    cleaned = PAGE_CITATION_RE.sub(" ", answer)
    cleaned = _MARKDOWN_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", cleaned)


def speech_segments_of(text: str) -> List[str]:
    """Group the spoken copy into sentence-sized segments for early playback.

    The cap is what decides time-to-first-audio, so a long answer becomes more
    segments rather than one oversized tail clip that has to finish
    synthesising before the phone can start speaking.
    """
    if not text:
        return []
    segments: List[str] = []
    for part in split_sentences(text) or [text]:
        if segments and len(segments[-1]) + len(part) + 1 <= SPEECH_SEGMENT_MAX_CHARS:
            segments[-1] = f"{segments[-1]} {part}".strip()
        else:
            segments.extend(_wrap_words(part, SPEECH_SEGMENT_MAX_CHARS))
    return [segment for segment in segments if segment]


def _wrap_words(text: str, limit: int) -> List[str]:
    """Hard-split a single over-long sentence on word boundaries."""
    if len(text) <= limit:
        return [text]
    pieces: List[str] = []
    current: List[str] = []
    length = 0
    for word in text.split():
        if current and length + len(word) + 1 > limit:
            pieces.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces


def engine_name() -> str:
    return agent.llm.name if isinstance(agent.llm, GroundedLlm) else "extractive bm25 (no model configured)"


def answered(request: QueryRequest) -> Dict[str, Any]:
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    history = [{"query": turn.query, "answer": turn.answer or ""} for turn in (request.history or [])][-4:]
    result = agent.process_query(user_query, history=history)
    spoken_text = speech_text_of(result["answer"])
    return {
        "status": "success",
        "query": user_query,
        "answer": result["answer"],
        "spoken_text": spoken_text,
        "spoken_segments": speech_segments_of(spoken_text),
        "is_out_of_context": result["is_out_of_context"],
        # Nothing indexed is not the same fact as nothing found, and the phone
        # shows different copy for each.
        "no_document": bool(result.get("no_document")),
        "sources": result["sources"],
        "retrieval_score": result["retrieval_score"],
        "confidence_score": result["confidence_score"],
        "engine": result["engine"],
        "timings": dict(result["timings"]),
    }


@app.get("/api/health")
def health_check():
    rag = agent.rag
    return {
        "status": "healthy",
        "current_document": rag.current_filename or "None",
        "total_chunks": len(rag.chunks),
        "available_voices": len(VOICE_CATALOG),
        "default_voice": voice_engine.default_voice,
        "speech": provider_status(),
        "engine": engine_name(),
        "chunking": dict(rag.profile),
    }


@app.get("/api/document")
def document_status() -> Dict[str, Any]:
    rag = agent.rag
    if not rag.current_filename:
        return {"loaded": False}
    return {"loaded": True, **rag.status()}


@app.get("/api/voices")
def get_voices():
    # Annotated with whether this machine can actually speak each voice, so the
    # phone never offers a speaker that would fail on the first question.
    return {"voices": voice_engine.list_voices(), "default_voice": voice_engine.default_voice}


async def _read_upload(file: UploadFile, save_path: str) -> int:
    """Write the body to disk, refusing anything past the cap rather than filling it."""
    size = 0
    with open(save_path, "wb") as buffer:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"That PDF is over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Trim it or index a chapter.",
                )
            buffer.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="That file is empty.")
    return size


@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    safe_name = os.path.basename((file.filename or "").strip())
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    started = time.perf_counter()
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    try:
        await _read_upload(file, save_path)
        with open(save_path, "rb") as handle:
            header = handle.read(5)
        if not header.startswith(b"%PDF"):
            raise HTTPException(status_code=415, detail=f"{safe_name} is not a PDF, whatever it is called.")

        # Parsing happens in a worker thread against a separate engine: the loop
        # stays free to serve the speech requests of the answer already playing,
        # and a PDF that fails to parse cannot damage the index in use.
        fresh = PDFRAGEngine()
        stats = await asyncio.to_thread(fresh.load_pdf, save_path)
        if stats["total_chunks"] == 0:
            raise HTTPException(
                status_code=422,
                detail=f"{safe_name} holds no selectable text. A scanned PDF has to be OCR'd before it can be asked about.",
            )
        agent.rag = fresh
    except HTTPException:
        _discard(save_path)
        raise
    except Exception as error:
        _discard(save_path)
        raise HTTPException(status_code=422, detail=f"Could not read that PDF: {type(error).__name__}: {error}") from error

    stats["index_ms"] = round((time.perf_counter() - started) * 1000, 1)

    welcome_text = (
        f"{stats['filename']} is ready. {stats['total_pages']} pages, {stats['total_chunks']} indexed passages. "
        "What would you like to know?"
    )
    return {
        "status": "success",
        "stats": stats,
        "welcome_text": welcome_text,
        "welcome_segments": speech_segments_of(welcome_text),
        "engine": engine_name(),
    }


@app.post("/api/ask_text")
def ask_text(request: QueryRequest):
    """Grounded answer with no speech on the critical path."""
    return answered(request)


@app.post("/api/speak")
async def speak(request: SpeakRequest):
    """Synthesise the given spoken segments, in parallel, cache-first."""
    # Re-segmented here so nothing is silently truncated: a client may send one
    # long paragraph and still get every word, split at the playback cap.
    texts = [
        segment
        for text in request.texts
        for segment in speech_segments_of(speech_text_of(text))
    ]
    if not texts:
        raise HTTPException(status_code=400, detail="Provide the text you want to hear.")

    started = time.perf_counter()
    result = await voice_engine.text_to_speech_segments(texts, voice=request.voice)
    tts_ms = round((time.perf_counter() - started) * 1000, 1)

    if result["status"] == "fallback":
        raise HTTPException(status_code=502, detail=(result["errors"] or ["Speech synthesis failed."])[0])

    segments = [
        {**segment, "audio_url": f"/audio/{segment['filename']}"}
        for segment in result["segments"]
        if segment.get("filename")
    ]
    return {
        # `partial` reaches the phone as a warning rather than a shorter answer
        # that sounds complete.
        "status": result["status"],
        "voice": result["voice"],
        # Which engine made the audio, so the phone can show that the answer it
        # is hearing took 0.2 s or 2 s and why.
        "speech_engine": result.get("engine", ""),
        "segments": segments,
        "requested": result["requested"],
        "errors": result["errors"],
        "timings": {"tts_ms": tts_ms, "segments": len(segments), "cached": sum(1 for s in segments if s["cached"])},
    }


@app.post("/api/ask_voice")
async def ask_voice(request: QueryRequest):
    """One-call path used by the browser demo: answer plus speech."""
    started = time.perf_counter()
    payload = answered(request)
    segments = payload["spoken_segments"] or [payload["spoken_text"] or "I am listening."]

    tts_started = time.perf_counter()
    result = await voice_engine.text_to_speech_segments(segments, voice=request.voice)
    payload["timings"]["tts_ms"] = round((time.perf_counter() - tts_started) * 1000, 1)
    payload["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 1)

    spoken = [
        {**segment, "audio_url": f"/audio/{segment['filename']}"}
        for segment in result["segments"]
        if segment.get("filename")
    ]
    payload["voice"] = result["voice"]
    payload["speech_engine"] = result.get("engine", "")
    payload["segments"] = spoken
    payload["audio_url"] = spoken[0]["audio_url"] if spoken else ""
    payload["audio_base64"] = None
    payload["errors"] = result["errors"]
    payload["status"] = result["status"] if spoken else "text_only"
    return payload


@app.get("/audio/{filename}")
def get_audio(filename: str):
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid audio filename.")
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
    media_type = AUDIO_MEDIA_TYPES.get(os.path.splitext(filename)[1].lower(), "application/octet-stream")
    return FileResponse(file_path, media_type=media_type)


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_file = os.path.join(os.path.dirname(__file__), "client_demo.html")
    if os.path.exists(html_file):
        with open(html_file, "r") as handle:
            return handle.read()
    return "<h1>Real-Time Voice-to-Voice PDF Conversation API is Running</h1>"


if __name__ == "__main__":
    import uvicorn

    # uvicorn configures its own loggers and leaves the root one alone, so without
    # this the server's own warnings and the warm-up line are silently dropped.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8080)
