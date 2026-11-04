# Real-Time Voice-to-Voice PDF Conversation Engine

Upload a PDF, ask a question out loud, hear a grounded answer. Retrieval is Okapi BM25 over line-aware chunks, answers are extractive (only text that is physically in your document), and speech is made on this machine by two local neural engines — Supertonic 3 and Piper — with `edge-tts` kept as the network fallback. No model subscription, no third-party speech bill, and no answer text leaves the computer unless you deliberately pick an Edge voice.

Everything here runs on your own machine: `server.py` on the LAN, the React Native app in [mobile/](mobile) on your phone.

---

## The latency design

Speech is the slow part, so it is never on the path that puts text on the screen.

| step | endpoint | what it does |
| --- | --- | --- |
| 1 | `POST /api/ask_text` | retrieve, answer, return the answer plus the spoken copy already split into segments. No TTS. |
| 2 | `POST /api/speak` with the first segment | synthesise one sentence, cache-first, return its `/audio/...` URL. |
| 3 | `POST /api/speak` with the remaining segments | fired at the same moment as step 2; the phone queues them while the first clip is still playing. |

`POST /api/ask_voice` keeps the single-call path for the browser demo. `GET /api/health`, `GET /api/document`, `GET /api/voices` and `POST /api/upload_pdf` complete the surface.

Measured against a local uvicorn process on 2026-11-03 (`python3 test_latency.py --base http://127.0.0.1:8080`, 15-page manual, 48 chunks, loopback, no phone attached):

```
PDF ingest + BM25 index        80.5 ms
ask_text wall                  0.7 - 2.1 ms   (retrieval 0.1 - 1.1 ms, answer 0.0 ms)

one never-heard sentence, per engine
  piper       en_US-ryan-medium   218.4 ms + 3.1 ms to fetch 230956 B audio/wav
  supertonic  supertonic-M1       659.6 ms + 2.4 ms to fetch 534572 B audio/wav
  edge        en-IN-PrabhatNeural 2155.1 ms + 3.5 ms to fetch  46080 B audio/mpeg

whole answer asked again         0.6 ms (2/2 segments from cache)
```

`test_pipeline.py` times the same sentence with the model already loaded, in a directory with no cache to flatter it, and reports what the engine costs relative to the speech it produced: Piper made 4.11 s of audio in 112 ms (0.027x real time), Supertonic made 5.99 s in 661 ms (0.110x). Both are faster than listening to them, which is what "real time" has to mean here — the first sentence exists before you could have finished reading it, and the rest of the answer is synthesised while that first clip plays.

The server loads each engine's model about 1.3 s after it starts, in a background task, so `GET /api/health` answers in milliseconds while the models come up and the first answer of a session does not pay for a load. A voice that has never been used on that process still pays its own one-off load (~1 s) the first time it speaks, then nothing.

So a repeated question puts text on screen in about a millisecond and starts speaking in about two. A brand-new sentence costs one synthesis — about a fifth of a second locally, about two over the network — and the phone starts talking as soon as that first sentence exists rather than waiting for the whole answer.

## Speech: three engines, one contract

Every voice in the catalog is produced through the same call and returns the same shape, so nothing else in the application knows or cares which engine ran.

| engine | where it runs | clip | voices | one never-heard sentence |
| --- | --- | --- | --- | --- |
| `supertonic` | this machine, ONNX | 44.1 kHz WAV | 10 (5 male, 5 female), each one Hindi **and** English | ~0.66 s |
| `piper` | this machine, ONNX | 22.05 kHz WAV | 5 (2 English, 3 Hindi) | ~0.22 s |
| `edge` | Microsoft, over the network | MP3 | 4 (2 Indian English, 2 Hindi) | ~2.15 s |

Both local engines come from their own GitHub projects and are installed as plain wheels: Supertonic 3 (`supertone-inc/supertonic`) and Piper 1 (`OHF-Voice/piper1-gpl`, the GPL rewrite that ships its own espeak bridge — no system `espeak-ng` is needed, confirmed on this machine where `espeak-ng` is not installed and Devanagari still phonemises as Hindi).

Hindi and English in one answer is the reason Supertonic is the default. Its model carries a single voice that reads both, and `lang` is chosen per sentence: Devanagari anywhere in the text selects the Hindi front-end, otherwise English. Piper's voices are one-language models, but its espeak bridge phonemises Devanagari as Hindi even under an English voice — romanised Hinglish is what it reads badly, which is why the extractive answers (always the document's own English text, or its own Devanagari) do not need manual span splitting.

The server picks for you and tells you what it picked: `GET /api/health` returns `default_voice` plus a `speech` block with each engine's `available`, `usable`, `format` and `why_not`. The order is `supertonic-M1` then `en_US-ryan-medium` then `hi_IN-pratham-medium` then the Edge fallback, and the first voice whose model actually exists on disk wins — so an machine with nothing installed still speaks, over Edge, rather than going silent.

Guardrails that hold across engines:

- **one engine per answer.** The voice is chosen per request, so an answer is never half local and half network; the `speech_engine` field names the engine that ran, and each clip URL's extension carries that engine's format.
- **the cache key is the engine too.** Text, voice id and rate select the file, and the file's extension carries the format, so a Supertonic WAV can never be served where an Edge MP3 was asked for.
- **local work is serialised, deliberately.** One ONNX sentence already uses every core; queueing a second in parallel only delays the first clip the phone is waiting for. `edge` keeps its parallel path because it is waiting on a socket, not on CPU.
- **the rate means the same thing everywhere.** The server's default `+8%` (`DEFAULT_RATE`) is a multiplier; the phone sends no rate and inherits it. It inverts into Piper's `length_scale` (a duration) and passes straight through into Supertonic's `speed`.
- **a 20 s ceiling on any one sentence** (`LOCAL_TIMEOUT_S`), with the timeout reported honestly instead of a silent partial success. A worker thread cannot be killed, so the shielded thread is left to finish into its own `.part` file and a reaper deletes abandoned `.part` files older than the young-file guard.
- **warming means the model is in memory.** Startup loads each engine through `warm_voice`, not by speaking a sentence, because a warm-up line is a disk-cache hit on the second run and would leave the session missing while the log claimed the engines were ready.

To hear the voices before choosing, `python3 bench_voice.py` writes clips into `tts_models/samples/` — one file per voice per sentence, named `<engine>_<voice>_<english|devanagari|mixed>.wav`, with the same three sentences for every engine so they are comparable. It speaks at the settings the application uses (Supertonic at 4 steps, which measures 660-780 ms a sentence on this Mac against 1.2-1.4 s at the library default of 8), and its default shortlist is `supertonic_M1/M2/F1/F2` plus all five Piper voices; `--voices` takes any of the other six Supertonic styles.

## Accuracy and grounding

`python3 test_accuracy.py` runs 9 in-context questions, 5 out-of-context guardrail questions, a provenance check (the `[Page N]` markers in an answer must equal the returned `sources`), a fact-token hallucination check (every URL, file name, credential and number in an answer must appear verbatim in the document), and the spoken-copy contract (no page markers read aloud, every segment inside the 190-character playback cap, no words dropped).

The extractive answer path never paraphrases and never pads: an install line is read as the packages it lists and stops at the first ordinary English word after them, so prose like "to prepare the laboratory environment" cannot leak into a package list. A list longer than the answer will read out is named up to eight deep and the remainder is counted ("plus 3 more on that line") rather than dropped, because a silently missing package is an install that fails an hour later.

Last run: **16/16 checks passed**, slowest answer 0.8 ms offline. `python3 test_pipeline.py` additionally synthesises the login answer on every engine this machine can run (Edge MP3, Piper WAV at 22050 Hz, Supertonic WAV at 44100 Hz) and opens each clip with `wave` to assert it is a real sentence.

Both harnesses read a sample PDF from `CONVO_SAMPLE_PDF` (default: `/Users/pratik/Documents/a/STQA Laboratory Manual & Practical Reference Guide.pdf`). Point it at any PDF you own to reproduce.

## Failure modes

`python3 test_resilience.py` drives every way this application can fall over and asserts what the user gets instead. It builds its own PDFs, so it needs no sample file and no model key. Last run: **69/69 checks passed** — 47 of them the original surface, 22 added for the local engines.

| what goes wrong | what happens |
| --- | --- |
| non-PDF bytes named `.pdf`, empty body, wrong filename, over the 80 MB cap, unreadable, no selectable text | refused with 415 / 400 / 400 / 413 / 422 / 422 and a reason, and the previously good index keeps answering |
| a PDF fails halfway through indexing | the fresh engine is discarded; `agent.rag` is swapped only once indexing succeeded, so no torn or empty index is ever published |
| synthesis answers for some sentences only | `200` with `status: "partial"`, `requested` vs the clips that arrived, and the reason; the phone warns instead of playing a short answer that sounds complete |
| synthesis answers for none of them | `502` with the real error, never an empty success |
| a model file is missing (fresh clone, no weights) | the voice is reported `available: false` with the exact command to fetch it, `/api/health` says which engine is `usable`, the default silently becomes one that works, and asking for the broken voice returns its reason rather than silence |
| one voice style of an engine is not in the weights that are on disk | that voice is refused with its own reason while the engine still reports the styles it can speak, so a missing `Z9` never takes Supertonic down for the other ten |
| a local engine fails for every sentence of an answer | the caller falls back to the Edge voice, and `engine` says which one actually spoke |
| a local sentence takes longer than 20 s | a timeout that names the engine, no half-written clip, and the abandoned thread's `.part` file reaped later |
| a stalled speech endpoint | abandoned after 20 s (`SYNTHESIS_TIMEOUT_S`) rather than hanging the request |
| a truncated or zero-byte clip | a failure, not a success: nothing under `MIN_CLIP_BYTES` is committed, and a clip is only ever published by rename, so a half file can never be cached forever |
| a clip asked for under the wrong extension | 404 — a WAV cannot be served as if it were the MP3 cached for the same text |
| the same sentence asked for in two engines | two files, two cache entries; switching voice never replays the other voice's audio |
| an old clip evicted while the phone is replaying it | a young-file guard keeps the last 10 minutes of clips, and a 404 on one clip makes the queue skip it and keep playing, reporting `N of M clips could not be played` at the end |
| the configured model crashes, hangs up, or returns `null`, `[]`, a wrong-shaped object or an empty string | the same retrieved passages still produce a grounded, cited answer, and `engine` names the real failure (`extractive (model call failed: ...)`) |
| the model cites a page retrieval never returned | that marker is stripped, and every marker that survives is a page listed in `sources` |
| the model returns only a page marker and no words | replaced by the extractive answer, `engine` says `cited nothing usable` |
| nothing is indexed | `no_document: true` and "Upload a PDF first", which is a different fact from "not in this document", and the phone shows a different chip for each |
| an unplanned crash anywhere in a handler | a 500 that is still JSON with a readable `detail`, so the phone can show it and offer Retry |

## Chunking

Chunks are built line-aware, not by fixed character windows, and nothing is thrown away:

- a run shorter than `min_chunk_words` is merged into the passage before it instead of dropped, which is what keeps a one-line fact (`pip install selenium`, a single assertion) findable;
- a detected heading is carried into the chunks that follow it, and across page breaks, so "Practical 4" stays attached to its steps;
- body text and heading text are scored separately (`heading_boost`), and a chunk whose heading matches the section you asked about gets a bonus while a chunk whose heading contradicts it is penalised.

The server reports the profile it measured, so this is inspectable rather than asserted — `GET /api/health` and `GET /api/document` include it, and the phone shows it in Settings:

```
chunks 48, words_min 4, words_p50 48, words_max 119, words_avg 56.1,
headings_tracked 38, short_runs_merged 1, pages_with_chunks 15,
unique_terms 469, text_preserved_pct 100.0
```

## Run it

```bash
cd /Users/pratik/Documents/realtime_convo
python3 -m pip install fastapi uvicorn pypdf edge-tts httpx
python3 server.py                      # listens on 0.0.0.0:8080
open http://127.0.0.1:8080             # browser demo
```

That alone works: speech comes from `edge-tts`, because nothing local is installed yet and the server notices. To get the real-time, offline voices, add the two local engines:

```bash
python3 -m pip install piper-tts supertonic onnxruntime soundfile

# Piper models, ~60 MB each. Any five of these is the full catalog.
python3 -m piper.download_voices en_US-ryan-medium en_US-lessac-medium \
    hi_IN-pratham-medium hi_IN-priyamvada-medium hi_IN-rohan-medium \
    --download-dir tts_models

# Supertonic 3 weights, fetched once into the directory the engine looks in.
python3 -c "import supertonic; supertonic.TTS(model_dir='tts_models/supertonic3')"
```

`CONVO_TTS_MODELS` moves the model directory; `CONVO_DEFAULT_VOICE` pins the starting voice instead of letting the server pick the fastest one it can actually use. Both engines are checked at startup and reported by `GET /api/health`, so a missing model is a reason on the voice picker, not a silent failure later.

Then follow [mobile/README.md](mobile/README.md) for the phone: it needs a development build (third-party native modules: `expo-audio`, `expo-speech-recognition`, `expo-document-picker`), and the server address must be your computer's LAN address, for example `http://10.151.207.43:8080` — `localhost` on a phone is the phone.

## What is verified how

- Backend retrieval, chunking, guardrails, provenance and all three speech engines: run as processes on this machine, the numbers above are captured output. `test_accuracy.py` (16/16), `test_resilience.py` (69/69), `test_pipeline.py` and `test_latency.py` all run as-is. Supertonic and Piper synthesis were each run for real on this Mac, including Devanagari and mixed Hindi/English sentences, and all ten Supertonic styles were spoken end to end.
- Browser demo: served by the same `server.py`, voice list rendered from `/api/voices` (19 voices, the server default pre-selected), and the returned WAV clips verified to decode in a real browser element (`readyState` 4, duration measured).
- React Native client: typechecked (`npm run typecheck`), unit-tested (`npm test`, 29 tests over the conversation reducer, the playback queue including a dead clip in the middle of an answer, the answer bubble, the microphone-language rules, and the voice sheet refusing a voice this machine cannot speak), and bundle-exported for Android and iOS. **Not** installed on a physical device in this session, so microphone behaviour and audio routing on real hardware are unverified.
