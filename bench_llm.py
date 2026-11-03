"""Head-to-head model benchmark for the grounded answer path.

Scores the same 84 corpus questions against each candidate model and reports the
three numbers that decide a production pick: accuracy, latency and token cost.
Run it before changing `CONVO_LLM_MODEL`, so the choice is measured rather than
assumed - a model that is 4% smarter but costs 3x the tokens may not be worth it
for a voice answer that has to be understood in one hearing.

    python3 bench_llm.py                       # every available candidate
    python3 bench_llm.py openai/gpt-oss-20b    # only these models
    CONVO_BENCH_DOCS=api,prices python3 bench_llm.py
"""

import os
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx  # noqa: E402

from agent import ConversationalAgent, GroundedLlm, _load_dotenv  # noqa: E402
from make_corpus import DOCUMENTS, build  # noqa: E402
from rag_engine import PDFRAGEngine  # noqa: E402
from test_corpus import evaluate  # noqa: E402

DEFAULT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def available_models() -> List[str]:
    _load_dotenv()
    key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return []
    base = os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    try:
        response = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
        response.raise_for_status()
        return [item["id"] for item in response.json().get("data", []) if item.get("id")]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        print(f"Could not list models: {error}")
        return []


def bench(model: str, docs, indexes: Dict[str, Any], documents: Dict[str, str]) -> Dict[str, Any]:
    llm = GroundedLlm(
        name=f"groq:{model}",
        url="https://api.groq.com/openai/v1/chat/completions",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        model=model,
        style="openai",
    )
    wrong: List[str] = []
    walls: List[float] = []
    prompt = completion = 0
    calls = 0
    for doc in docs:
        rag = indexes[doc.name]
        document = documents[doc.name]
        agent = ConversationalAgent(rag_engine=rag, llm=llm)
        for case in doc.questions:
            started = time.perf_counter()
            result = agent.process_query(case["q"])
            walls.append(round((time.perf_counter() - started) * 1000, 1))
            problems = evaluate(case, result, document)
            if llm.last_usage:
                calls += 1
                prompt += llm.last_usage.get("prompt_tokens", 0)
                completion += llm.last_usage.get("completion_tokens", 0)
            if problems:
                wrong.append(f"{doc.name}: {case['q']} -> {problems[0]}")

    walls.sort()
    total = sum(len(doc.questions) for doc in docs)
    return {
        "model": model,
        "correct": total - len(wrong),
        "total": total,
        "wrong": wrong,
        "median_ms": walls[len(walls) // 2] if walls else 0.0,
        "p95_ms": walls[int(len(walls) * 0.95) - 1] if walls else 0.0,
        "calls": calls,
        "prompt_avg": round(prompt / calls) if calls else 0,
        "completion_avg": round(completion / calls) if calls else 0,
    }


def main(argv: List[str]) -> int:
    _load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        print("No GROQ_API_KEY in the environment or .env, so there is nothing to benchmark.")
        return 2

    listed = available_models()
    wanted = argv or [m for m in DEFAULT_MODELS if not listed or m in listed]
    if listed:
        skipped = [m for m in wanted if m not in listed]
        for model in skipped:
            print(f"Skipping {model}: this key cannot see it.")
        wanted = [m for m in wanted if m in listed]
    if not wanted:
        print("No candidate model is available to this key.")
        return 2

    names = os.environ.get("CONVO_BENCH_DOCS", "").split(",") if os.environ.get("CONVO_BENCH_DOCS") else []
    docs = [doc for doc in DOCUMENTS if not names or any(name in doc.name for name in names)]

    indexes: Dict[str, Any] = {}
    documents: Dict[str, str] = {}
    for doc in docs:
        build(doc)
        rag = PDFRAGEngine()
        rag.load_pdf(doc.path)
        indexes[doc.name] = rag
        documents[doc.name] = re.sub(r"\s+", " ", rag.raw_text).lower()
    questions = sum(len(doc.questions) for doc in docs)
    print(f"Benchmarking {len(wanted)} model(s) over {questions} questions in {len(docs)} document(s)\n")

    results = []
    for model in wanted:
        report = bench(model, docs, indexes, documents)
        results.append(report)
        print(f"{model:<24} {report['correct']:>3}/{report['total']} correct  "
              f"median {report['median_ms']:.0f}ms  p95 {report['p95_ms']:.0f}ms  "
              f"{report['prompt_avg']}+{report['completion_avg']} tokens/call over {report['calls']} calls")

    print("\nFailures per model")
    print("-" * 88)
    for report in results:
        counts = Counter(problem.split(" -> ")[1].split(" ")[0] + " " + problem.split(" -> ")[1].split(" ")[1]
                         for problem in report["wrong"])
        print(f"\n{report['model']}  ({len(report['wrong'])} wrong)")
        for symptom, count in counts.most_common():
            print(f"   {count:>2} x {symptom}")
        for problem in report["wrong"][:12]:
            print(f"     - {problem[:120]}")

    best = max(results, key=lambda item: (item["correct"], -item["prompt_avg"]))
    print("\n" + "=" * 88)
    print(f"Recommended CONVO_LLM_MODEL: {best['model']} "
          f"({best['correct']}/{best['total']} correct, {best['prompt_avg']}+{best['completion_avg']} tokens, "
          f"p95 {best['p95_ms']:.0f}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main([arg for arg in sys.argv[1:] if "/" in arg or "." in arg]))
