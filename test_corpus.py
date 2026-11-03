"""Score the answer path against seven different document genres.

    python3 test_corpus.py            # all documents
    python3 test_corpus.py api paper  # only the named ones
    CONVO_CORPUS_VERBOSE=1 python3 test_corpus.py lab   # every answer inline

For every question the truth was authored next to the text in `make_corpus.py`, and
five separate properties are required, not one:

  1. the answer carries the expected fact(s);
  2. it cites a page where that fact actually appears;
  3. provenance is self-consistent - the `[Page N]` markers equal the `sources` list;
  4. no fact token (host, filename, identifier, number) appears unless the document
     contains that string verbatim - the hallucination test;
  5. a question the document cannot answer is refused, cites nothing, and reports a
     retrieval score of exactly 0.0.

The engine under test is the deterministic one (`llm=None`), which is what ships
when no API key is configured. Set `CONVO_CORPUS_LLM=1` to run the same questions
through the configured model instead - that is the production path, and the report
then carries the token cost per answer, because accuracy you cannot afford is not
accuracy. Failures are grouped by symptom at the end, because the symptom is what
tells you which layer has the bug.
"""

import os
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import ConversationalAgent, GroundedLlm  # noqa: E402
from make_corpus import DOCUMENTS, build  # noqa: E402
from rag_engine import PDFRAGEngine  # noqa: E402

CITATION_RE = re.compile(r"\[Page\s+(\d+)\]", re.I)
# Anything that could only have come from the document: hosts, filenames,
# identifiers, decimal figures. Plain English words are not facts - a model that
# writes "the endpoint requires a signature" has used connective vocabulary, not a
# claim that can be looked up and checked. There is no offline English dictionary to
# argue with, so this check is a floor on hallucination, not a ceiling: it cannot
# catch an invented plain word that happens to also be a document entity.
FACT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]*[A-Za-z0-9_.\-/][A-Za-z0-9]*|[A-Za-z0-9_.\-/]+\d[A-Za-z0-9_.\-/]*")
# Devanagari words count as facts too: a Hindi answer that invents a Hindi term is
# as wrong as one that invents a URL.
INDIC_FACT_RE = re.compile(r"[\u0900-\u097F]{3,}")


def fact_tokens(answer: str) -> List[str]:
    stripped = CITATION_RE.sub(" ", answer)
    tokens = list(INDIC_FACT_RE.findall(stripped))
    for raw in FACT_RE.findall(stripped):
        cleaned = raw.strip(".,;:!?()'\"")
        if not cleaned:
            continue
        if cleaned.isdigit() and len(cleaned) < 2:
            continue
        if not re.search(r"[.\-/]|\d", cleaned):
            continue
        tokens.append(cleaned)
    return tokens


def evaluate(case: Dict[str, Any], result: Dict[str, Any], document: str) -> List[str]:
    """Return the symptoms this answer shows. Empty means the answer is correct."""
    problems: List[str] = []
    answer = result["answer"]
    lowered = answer.lower()
    cited = {int(page) for page in CITATION_RE.findall(answer)}
    sources = set(result["sources"])

    if case["refuse"]:
        if not result["is_out_of_context"]:
            problems.append("answered a question the document cannot answer")
        if sources or cited:
            problems.append(f"cited pages {sorted(sources | cited)} for a question with no answer")
        if result["confidence_score"] != 0.0:
            problems.append(f"refusal still reports confidence {result['confidence_score']}")
        return problems

    if result["is_out_of_context"]:
        problems.append("refused a question the document does answer")
        return problems

    missing = [term for term in case["contains"] if term.lower() not in lowered]
    if missing:
        problems.append(f"answer is missing {missing}")
    for term in case["not_contains"]:
        if term.lower() in lowered:
            problems.append(f"answer contains the forbidden {term!r}")
    if case["pages"] and not (sources & set(case["pages"])):
        problems.append(f"cited {sorted(sources)} instead of a page in {case['pages']}")
    if cited != sources:
        problems.append(f"markers {sorted(cited)} disagree with sources {sorted(sources)}")
    invented = sorted({token for token in fact_tokens(answer) if token.lower() not in document})
    if invented:
        problems.append(f"invented {invented}")
    return problems


def run_document(doc, verbose: bool, llm=None) -> Dict[str, Any]:
    rag = PDFRAGEngine()
    started = time.perf_counter()
    info = rag.load_pdf(doc.path)
    ingest_ms = round((time.perf_counter() - started) * 1000, 1)
    agent = ConversationalAgent(rag_engine=rag, llm=llm)
    document = re.sub(r"\s+", " ", rag.raw_text).lower()

    rows: List[Dict[str, Any]] = []
    for case in doc.questions:
        query_started = time.perf_counter()
        result = agent.process_query(case["q"])
        wall_ms = round((time.perf_counter() - query_started) * 1000, 1)
        usage = dict(agent.llm.last_usage) if agent.llm is not None else {}
        rows.append({
            "case": case,
            "result": result,
            "wall_ms": wall_ms,
            "usage": usage,
            "problems": evaluate(case, result, document),
        })
        if verbose:
            mark = "ok  " if not rows[-1]["problems"] else "FAIL"
            cost = "" if not usage else f"  [{usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)} tok]"
            print(f"  {mark} {case['q']}{cost}")
            print(f"       -> {result['answer'][:150]}")
            if rows[-1]["problems"]:
                print(f"       pages={result['sources']} score={result['retrieval_score']} "
                      f"expected={case['contains']} on {case['pages']}")
                for problem in rows[-1]["problems"]:
                    print(f"       * {problem}")

    profile = info["chunking"]
    return {
        "doc": doc,
        "info": info,
        "ingest_ms": ingest_ms,
        "rows": rows,
        "passed": sum(1 for row in rows if not row["problems"]),
        "profile": profile,
    }


def main(argv: List[str]) -> int:
    wanted = argv or [doc.name for doc in DOCUMENTS]
    docs = [doc for doc in DOCUMENTS if any(name in doc.name for name in wanted)]
    if not docs:
        print(f"No corpus document matches {wanted}. Known: "
              + ", ".join(doc.name for doc in DOCUMENTS))
        return 2
    verbose = os.environ.get("CONVO_CORPUS_VERBOSE") == "1"
    llm = GroundedLlm.from_env() if os.environ.get("CONVO_CORPUS_LLM") == "1" else None
    if os.environ.get("CONVO_CORPUS_LLM") == "1" and llm is None:
        print("CONVO_CORPUS_LLM=1 but no API key was found; staying offline.")
    print(f"Engine: {llm.name if llm else 'extractive bm25 (offline, deterministic)'}")
    for doc in docs:
        build(doc)

    print(f"\n{'document':<26} {'pages':>5} {'passages':>8} {'preserved':>9} {'headings':>8} {'questions':>10} {'ingest':>8}")
    print("-" * 88)
    reports = []
    for doc in docs:
        report = run_document(doc, verbose, llm)
        reports.append(report)
        profile = report["profile"]
        print(f"{doc.name:<26} {report['info']['total_pages']:>5} {profile['chunks']:>8} "
              f"{profile['text_preserved_pct']:>8}% {profile['headings_tracked']:>8} "
              f"{report['passed']:>6}/{len(doc.questions):<3} {report['ingest_ms']:>6}ms")

    total_q = sum(len(r["doc"].questions) for r in reports)
    passed_q = sum(r["passed"] for r in reports)
    symptoms: Counter = Counter()
    print("\nFailures")
    print("-" * 88)
    for report in reports:
        for row in report["rows"]:
            if not row["problems"]:
                continue
            problems = row["problems"]
            for problem in problems:
                symptoms[problem.split(" ")[0] + " " + (problem.split(" ")[1] if len(problem.split()) > 1 else "")] += 1
            print(f"  {report['doc'].name:<24} {row['case']['q']}")
            print(f"      answer : {row['result']['answer'][:120]}")
            print(f"      truth  : {row['case']['contains']} on pages {row['case']['pages']}"
                  + (" (must be refused)" if row["case"]["refuse"] else ""))
            print(f"      got    : pages={row['result']['sources']} score={row['result']['retrieval_score']} "
                  f"ooc={row['result']['is_out_of_context']}")
            for problem in problems:
                print(f"      * {problem}")
    if passed_q == total_q:
        print(f"  none - {passed_q}/{total_q} questions answered correctly across {len(reports)} documents")

    slowest = sorted(
        ((row["wall_ms"], report["doc"].name, row["case"]["q"]) for report in reports for row in report["rows"]),
        reverse=True,
    )[:3]
    print(f"\nSlowest answers ({'live model' if llm else 'offline, no model call'})")
    for wall_ms, name, query in slowest:
        print(f"  {wall_ms:>7}ms  {name}: {query[:56]}")

    asked = [row for report in reports for row in report["rows"] if row["usage"]]
    if asked:
        walls = sorted(row["wall_ms"] for row in asked)
        prompt = sum(row["usage"].get("prompt_tokens", 0) for row in asked)
        completion = sum(row["usage"].get("completion_tokens", 0) for row in asked)
        print(f"\nCost per answer (live model): {len(asked)} calls, "
              f"{prompt // len(asked)} prompt + {completion // len(asked)} completion tokens on average, "
              f"median {walls[len(walls) // 2]:.0f}ms, slowest {walls[-1]:.0f}ms")

    print("\n" + "=" * 88)
    print(f"Corpus accuracy: {passed_q}/{total_q} questions correct across {len(reports)} document genres")
    if passed_q != total_q:
        print(f"{total_q - passed_q} wrong. Symptom counts: {dict(symptoms)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main([arg for arg in sys.argv[1:] if not arg.startswith("-")]))
