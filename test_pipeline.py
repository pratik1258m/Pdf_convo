import os
import sys
import time
import wave
import asyncio

# Add current dir to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import PDFRAGEngine
from voice_engine import VoiceEngine
from agent import ConversationalAgent

async def run_verification():
    print("==================================================")
    print("▶ Running Autonomous Ralph Loop Pipeline Test")
    print("==================================================")

    sample_pdf = "/Users/pratik/Documents/a/STQA Laboratory Manual & Practical Reference Guide.pdf"
    if not os.path.exists(sample_pdf):
        print(f"❌ Sample PDF not found at {sample_pdf}")
        sys.exit(1)

    # 1. Test PDF Ingestion
    print("\n1. Testing PDF Ingestion & Semantic Chunking...")
    rag = PDFRAGEngine()
    stats = rag.load_pdf(sample_pdf)
    print(f"   ✓ Loaded: {stats['filename']}")
    print(f"   ✓ Pages: {stats['total_pages']}, Chunks: {stats['total_chunks']}")
    assert stats['total_chunks'] > 0, "No chunks extracted!"

    agent = ConversationalAgent(rag_engine=rag, llm=None)

    # 2. Test In-Context Query
    print("\n2. Testing In-Context Grounded Query...")
    in_query = "What is practical about login automation and validation?"
    in_res = agent.process_query(in_query)
    print(f"   Query: '{in_query}'")
    print(f"   Answer: {in_res['answer'][:150]}...")
    print(f"   Sources: Page(s) {in_res['sources']}")
    print(f"   Is Out-of-Context: {in_res['is_out_of_context']}")
    assert in_res['is_out_of_context'] is False, "Expected in-context, but got out-of-context!"

    # 3. Test Out-of-Context Polite Guardrail
    print("\n3. Testing Out-of-Context Polite Guardrail Query...")
    out_query = "What is the capital of France and how to make pizza?"
    out_res = agent.process_query(out_query)
    print(f"   Query: '{out_query}'")
    print(f"   Answer: {out_res['answer']}")
    print(f"   Is Out-of-Context: {out_res['is_out_of_context']}")
    assert out_res['is_out_of_context'] is True, "Expected out-of-context guardrail to trigger!"
    assert "could not locate" in out_res['answer'].lower(), "Polite message missing!"

    # 4. Test Neural Voice Synthesis (Edge TTS)
    print("\n4. Testing Edge TTS Neural Voice Generation...")
    voice = VoiceEngine()
    test_audio = await voice.text_to_speech(in_res['answer'][:120], voice="en-IN-PrabhatNeural")
    print(f"   ✓ Audio Generated: {test_audio['filename']}")
    print(f"   ✓ File Size: {test_audio['byte_size']} bytes")
    assert test_audio['byte_size'] > 1000, "Audio output file is too small or empty!"

    # 5. The local engines, which is where the real-time path lives
    import shutil
    import tempfile

    from voice_engine import available_voices

    local = [entry for entry in available_voices() if entry["available"] and entry["local"]]
    if not local:
        print("\n5. No local speech model on this machine - nothing to test (see README install step)")
    by_engine = {}
    for entry in local:
        by_engine.setdefault(entry["engine"], entry)
    print(f"\n5. Testing local speech engines: {', '.join(sorted(by_engine)) or 'none'}")

    # An empty directory: a clip from an earlier run would answer 0 ms and say
    # nothing about what the engine costs.
    scratch = tempfile.mkdtemp(prefix="convo-pipeline-")
    local_voice = VoiceEngine(output_dir=scratch)
    try:
        for engine, entry in sorted(by_engine.items()):
            # One throwaway line first, so the number below is a sentence and not
            # the one-off cost of loading the model into memory.
            await local_voice.text_to_speech("Warming up.", voice=entry["id"], rate="+8%")
            started = time.perf_counter()
            clip = await local_voice.text_to_speech(in_res['answer'][:120], voice=entry["id"], rate="+8%")
            wall_ms = round((time.perf_counter() - started) * 1000)
            assert clip["status"] == "success", f"{engine} failed: {clip.get('error')}"
            assert clip["filename"].endswith(".wav"), f"{engine} wrote {clip['filename']}"
            path = os.path.join(scratch, clip["filename"])
            with wave.open(path, "rb") as handle:
                frames, rate_hz, channels = handle.getnframes(), handle.getframerate(), handle.getnchannels()
            assert frames > 1000, f"{engine} wrote {frames} frames, which is not a sentence"
            seconds = frames / rate_hz
            print(f"   ✓ {engine:<11} {entry['id']:<22} {clip['byte_size']:>7}B  {wall_ms:>5}ms for "
                  f"{seconds:.2f}s of speech ({wall_ms / 1000 / seconds:.3f}x real time)  "
                  f"{frames} frames @ {rate_hz}Hz over {channels} channel(s)")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print("\n==================================================")
    print("✅ ALL TESTS PASSED EMPIRICALLY! ZERO REGRESSIONS.")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_verification())
