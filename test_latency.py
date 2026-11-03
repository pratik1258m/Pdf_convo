"""End-to-end latency harness for the running API.

Measures the two paths the phone actually uses and the caching that makes the
second one cheap. Requires a server (`python3 -m uvicorn server:app --port 8080`)
and prints real wall-clock milliseconds; nothing here is estimated.

    python3 test_latency.py [--base http://127.0.0.1:8080]
"""

import argparse
import json
import os
import sys
import time
import uuid

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAMPLE_PDF = os.environ.get(
    "CONVO_SAMPLE_PDF", "/Users/pratik/Documents/a/STQA Laboratory Manual & Practical Reference Guide.pdf"
)

QUERIES = [
    "What username and password does the first Selenium practical send to the login form?",
    "What Excel file stores the student records in the data-driven test?",
    "How does the manual handle a JavaScript alert with an explicit wait?",
    "What is the capital of France?",
]


def timed(client: httpx.Client, method: str, path: str, **kwargs):
    started = time.perf_counter()
    response = getattr(client, method)(path, **kwargs)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    response.raise_for_status()
    return response.json(), elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    args = parser.parse_args()

    if not os.path.exists(SAMPLE_PDF):
        print(f"Sample PDF missing: {SAMPLE_PDF}")
        return 2

    with httpx.Client(base_url=args.base, timeout=60.0) as client:
        try:
            health, ms = timed(client, "get", "/api/health")
        except (httpx.HTTPError, OSError) as error:
            print(f"Server not reachable at {args.base}: {error}")
            print("Start it with: python3 -m uvicorn server:app --host 0.0.0.0 --port 8080")
            return 2
        print(f"Health {ms}ms | engine={health['engine']}")

        with open(SAMPLE_PDF, "rb") as handle:
            _, upload_ms = timed(
                client,
                "post",
                "/api/upload_pdf",
                files={"file": (os.path.basename(SAMPLE_PDF), handle, "application/pdf")},
            )
        print(f"PDF ingest + BM25 index: {upload_ms}ms (server-side index time is in stats.index_ms)")

        document, _ = timed(client, "get", "/api/document")
        print(f"Chunking: {json.dumps(document['chunking'])}")

        print("\nText-first path (no speech on the critical path)")
        print("-" * 66)
        segments_for: list = []
        for query in QUERIES:
            payload, ms = timed(client, "post", "/api/ask_text", json={"query": query})
            server = payload["timings"]
            spoken = payload["spoken_text"]
            print(f"ask_text  wall={ms:>6}ms  retrieval={server['retrieval_ms']:>5}ms "
                  f"answer={server['answer_ms']:>5}ms  score={payload['retrieval_score']:.2f} "
                  f"pages={payload['sources']}")
            print(f"   Q: {query}")
            print(f"   A: {spoken[:120]}")
            if len(payload["spoken_segments"]) > len(segments_for):
                segments_for = payload["spoken_segments"]
            if not payload["is_out_of_context"] and payload["timings"]["answer_ms"] > 40:
                print("   note: a model-backed answer was used (llm path)")

        print("\nSpeech path, exactly as the phone drives it")
        print("-" * 66)
        probe = f"Latency probe {uuid.uuid4().hex[:8]}: this sentence has never been synthesized before."
        fresh, fresh_ms = timed(client, "post", "/api/speak", json={"texts": [probe]})
        print(f"never-heard sentence: {fresh_ms}ms wall, server tts_ms={fresh['timings']['tts_ms']}, "
              f"{fresh['timings']['cached']}/{len(fresh['segments'])} from cache")

        head, *tail = segments_for
        spoken_head, head_ms = timed(client, "post", "/api/speak", json={"texts": [head]})
        started = time.perf_counter()
        clip = client.get(spoken_head["segments"][0]["audio_url"])
        clip_ms = round((time.perf_counter() - started) * 1000, 1)
        print(f"head segment: {head_ms}ms synthesis + {clip_ms}ms for {len(clip.content)}B download "
              f"= voice begins ~{round(head_ms + clip_ms)}ms after the answer reached the screen")

        if tail:
            spoken_tail, tail_ms = timed(client, "post", "/api/speak", json={"texts": tail})
            print(f"tail segments: {tail_ms}ms for {len(spoken_tail['segments'])} clips, enqueued while the head still plays")
        else:
            print("tail segments: none, this answer is a single clip")

        warm, warm_ms = timed(client, "post", "/api/speak", json={"texts": segments_for})
        print(f"whole answer asked again: {warm_ms}ms "
              f"({warm['timings']['cached']}/{len(warm['segments'])} from cache) server tts_ms={warm['timings']['tts_ms']}")

        print("\nEvery speech engine this server can run, one never-heard sentence each")
        print("-" * 66)
        catalog, _ = timed(client, "get", "/api/voices")
        first_of_engine: dict = {}
        for voice in catalog["voices"]:
            if voice.get("available") and voice.get("engine") not in first_of_engine:
                first_of_engine[voice["engine"]] = voice
        for engine, voice in sorted(first_of_engine.items()):
            # A throwaway line first: the initial call also pays for loading the
            # model, which is a one-off per server, not per answer.
            timed(client, "post", "/api/speak",
                  json={"texts": [f"Warm-up line for the {engine} engine."], "voice": voice["id"]})
            probe = f"Engine probe {uuid.uuid4().hex[:8]} for {engine}: this sentence has never been spoken."
            payload, ms = timed(client, "post", "/api/speak", json={"texts": [probe], "voice": voice["id"]})
            url = payload["segments"][0]["audio_url"]
            started = time.perf_counter()
            clip = client.get(url)
            clip_ms = round((time.perf_counter() - started) * 1000, 1)
            print(f"{engine:<11} {voice['id']:<22} {ms:>7}ms synthesis + {clip_ms:>5}ms to fetch "
                  f"{len(clip.content):>7}B {clip.headers.get('content-type')}")

        print("\nOne-call path (browser demo)")
        print("-" * 66)
        _, one_ms = timed(client, "post", "/api/ask_voice", json={"query": QUERIES[1]})
        _, two_ms = timed(client, "post", "/api/ask_voice", json={"query": QUERIES[1]})
        print(f"ask_voice: {one_ms}ms then {two_ms}ms (repeat only; this path is not used by the phone)")

    print("\nVerdict: text appears after ask_text, speech begins after the first /api/speak segment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
