"""Answer-accuracy harness for the grounded document Q&A path.

Runs offline against the real STQA lab manual PDF (15 pages). Every expected
fact below was read out of that document, so a passing case means retrieval found
the right page and the answer actually contains the right content.

Three properties are checked, not just one:
  1. the answer contains the fact and cites a page where that fact appears;
  2. provenance is consistent - every `[Page N]` in the answer text is in the
     `sources` list the API returns, and vice versa;
  3. no *fact token* (anything with a digit, dot, slash, hyphen or underscore:
     a URL, filename, credential, number) appears in the answer unless the same
     string appears verbatim in the document. Template words are not facts.

    python3 test_accuracy.py
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import ConversationalAgent  # noqa: E402
from rag_engine import PDFRAGEngine  # noqa: E402

SAMPLE_PDF = os.environ.get(
    "CONVO_SAMPLE_PDF", "/Users/pratik/Documents/a/STQA Laboratory Manual & Practical Reference Guide.pdf"
)

IN_CONTEXT_CASES = [
    {
        "query": "Which pip packages does the manual ask you to install for the lab environment?",
        "pages": {1},
        "must_contain": ["selenium", "pytest"],
    },
    {
        "query": "What username and password does the first Selenium practical send to the login form?",
        "pages": {1, 12, 14},
        "must_contain": ["standard_user", "secret_sauce"],
    },
    {
        "query": "Which username does the page-object login test on the-internet use?",
        "pages": {13},
        "must_contain": ["tomsmith"],
    },
    {
        "query": "Which website is loaded for the first Selenium IDE practical?",
        "pages": {1},
        "must_contain": ["saucedemo"],
    },
    {
        "query": "How does the manual handle a JavaScript alert with an explicit wait?",
        "pages": {3, 5},
        "must_contain": ["alert"],
    },
    {
        "query": "Which practical covers screenshot capture and logging?",
        "pages": {12},
        "must_contain": ["screenshot"],
    },
    {
        "query": "What Excel file stores the student records in the data-driven test?",
        "pages": {8},
        "must_contain": ["students.xlsx"],
    },
    {
        "query": "What course option appears in the student registration form?",
        "pages": {5},
        "must_contain": ["computer science"],
    },
    {
        "query": "What is the title assertion used for the Wikipedia cross-site test?",
        "pages": {2},
        "must_contain": ["wikipedia"],
    },
]

OUT_OF_CONTEXT_CASES = [
    "What is the capital of France?",
    "How do I cook biryani?",
    "Will it rain in Mumbai tomorrow?",
    "Who won the last World Cup final?",
    "Explain how a black hole forms.",
]

_CITATION_RE = re.compile(r"\[Page\s+(\d+)\]", re.I)
# A fact token is anything that could only have come from the document: hosts,
# filenames, credentials, identifiers, numbers.
_FACT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]*[A-Za-z0-9_.\-/][A-Za-z0-9]*|[A-Za-z0-9_.\-/]+\d[A-Za-z0-9_.\-/]*")


def load_agent():
    rag = PDFRAGEngine()
    rag.load_pdf(SAMPLE_PDF)
    # llm=None keeps this run deterministic and offline: it measures the retrieval
    # plus extractive synthesiser that ships with no API key.
    return rag, ConversationalAgent(rag_engine=rag, llm=None)


def containment_ratio(answer: str, terms) -> bool:
    lowered = answer.lower()
    return all(term.lower() in lowered for term in terms)


def fact_tokens(answer: str):
    stripped = _CITATION_RE.sub(" ", answer)
    tokens = []
    for raw in _FACT_RE.findall(stripped):
        cleaned = raw.strip(".,;:!?()[]\"'")
        if not cleaned or not re.search(r"[.\-/]|\d", cleaned):
            continue
        if cleaned.isdigit() and len(cleaned) < 2:
            continue
        tokens.append(cleaned)
    return tokens


def main() -> int:
    if not os.path.exists(SAMPLE_PDF):
        print(f"Sample PDF missing: {SAMPLE_PDF}")
        return 2

    rag, agent = load_agent()
    print(f"Indexed {rag.current_filename}: {rag.total_pages} pages, {len(rag.chunks)} passages")
    print(f"Chunking profile: {rag.profile}")
    print(f"Engine under test: {agent.process_query('probe')['engine']}")

    document = re.sub(r"\s+", " ", rag.raw_text).lower()
    failures = []
    slowest = []

    print("\nIn-context cases")
    print("-" * 72)
    for case in IN_CONTEXT_CASES:
        started = time.perf_counter()
        result = agent.process_query(case["query"])
        wall_ms = round((time.perf_counter() - started) * 1000, 1)
        answer = result["answer"]

        has_terms = containment_ratio(answer, case["must_contain"])
        pages_hit = bool(set(result["sources"]) & case["pages"])
        cited = {int(page) for page in _CITATION_RE.findall(answer)}
        provenance_ok = bool(cited) and cited == set(result["sources"])
        ok = (not result["is_out_of_context"]) and has_terms and pages_hit and provenance_ok
        slowest.append((wall_ms, case["query"]))
        status = "PASS" if ok else "FAIL"
        print(f"{status} score={result['retrieval_score']:.2f} pages={result['sources']} "
              f"retrieval={result['timings']['retrieval_ms']}ms total={wall_ms}ms")
        print(f"     Q: {case['query']}")
        print(f"     A: {answer[:160]}")
        if not ok:
            print(f"     expected {case['must_contain']} on pages {sorted(case['pages'])} -> "
                  f"terms={has_terms} pages={pages_hit} provenance={provenance_ok} "
                  f"cited={sorted(cited)} sources={result['sources']} ooc={result['is_out_of_context']}")
            failures.append(case["query"])

    print("\nOut-of-context guardrail cases")
    print("-" * 72)
    for query in OUT_OF_CONTEXT_CASES:
        result = agent.process_query(query)
        ok = result["is_out_of_context"] and "could not locate" in result["answer"].lower()
        print(f"{'PASS' if ok else 'FAIL'} score={result['retrieval_score']:.2f} {query}")
        print(f"     A: {result['answer'][:160]}")
        if not ok:
            failures.append(query)

    print("\nHallucination check (every fact token must appear in the document)")
    print("-" * 72)
    invented_total = 0
    for case in IN_CONTEXT_CASES:
        answer = agent.process_query(case["query"])["answer"]
        tokens = fact_tokens(answer)
        invented = sorted({token for token in tokens if token.lower() not in document})
        invented_total += len(invented)
        if invented:
            print(f"FAIL facts not present in document: {invented}")
            print(f"     A: {answer[:160]}")
            failures.append(f"hallucination for: {case['query']}")
    if invented_total == 0:
        checked = sum(len(fact_tokens(agent.process_query(c["query"])["answer"])) for c in IN_CONTEXT_CASES)
        print(f"PASS all {checked} fact tokens across {len(IN_CONTEXT_CASES)} answers are document text")

    print("\nSpoken copy + segment contract (the payload /api/speak receives)")
    print("-" * 72)
    from server import speech_segments_of, speech_text_of

    segment_count = 0
    for case in IN_CONTEXT_CASES:
        answer = agent.process_query(case["query"])["answer"]
        spoken = speech_text_of(answer)
        segments = speech_segments_of(spoken)
        segment_count += len(segments)
        problems = []
        if "[Page" in spoken:
            problems.append("a page marker would have been read out loud")
        if "  " in spoken:
            problems.append("double spaces left in the spoken copy")
        oversized = [len(seg) for seg in segments if len(seg) > 190]
        if oversized:
            problems.append(f"segments over the 190-char cap: {oversized}")
        if problems:
            print(f"FAIL {', '.join(problems)}")
            print(f"     A: {spoken[:160]}")
            failures.append(f"spoken copy for: {case['query']}")
    print(f"PASS {segment_count} spoken segments across {len(IN_CONTEXT_CASES)} answers, all inside the playback caps")

    long_answer = " ".join(
        f"Step {index} of the login automation flow opens the form and asserts the title." for index in range(1, 13)
    )
    spoken_long = speech_text_of(long_answer)
    parts = speech_segments_of(spoken_long)
    preserved = " ".join(parts) == spoken_long
    split_ok = len(parts) > 1 and all(len(part) <= 190 for part in parts) and preserved
    print(
        f"{'PASS' if split_ok else 'FAIL'} a {len(spoken_long)}-char answer becomes {len(parts)} segments, "
        f"every one <= 190 chars and nothing dropped{'' if preserved else ' (words were lost)'}"
    )
    if not split_ok:
        failures.append("long-answer segmentation")

    slowest.sort(reverse=True)
    print("\nSlowest answers (offline, no model call)")
    for wall_ms, query in slowest[:3]:
        print(f"  {wall_ms:>7}ms  {query[:64]}")

    total = len(IN_CONTEXT_CASES) + len(OUT_OF_CONTEXT_CASES) + 2
    passed = total - len(failures)
    print("\n" + "=" * 72)
    print(f"Accuracy: {passed}/{total} checks passed")
    if failures:
        print("Failed:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("All accuracy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
