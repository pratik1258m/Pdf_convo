"""Grounded question answering over a loaded PDF.

Two answering engines share one interface:
  * a configured OpenAI-compatible / Gemini model, forced to answer only from
    retrieved passages and to cite pages, and
  * a deterministic extractive synthesiser used when no model is configured or
    when the model call fails.

The `engine` field always reports which one actually produced the answer.
"""

import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import httpx

from rag_engine import (
    DocumentChunk, PDFRAGEngine, SECTION_TITLE_RE, STOP_WORDS, SearchHit, _SPEECH_ACT,
    flow_lines, heading_section, looks_like_heading, query_terms, requested_section,
    split_sentences, stem, tokenize,
)


class LlmAnswerError(RuntimeError):
    pass


def _load_dotenv() -> None:
    """KEY=VALUE lines from a .env beside the code; the real environment always wins and no value is ever logged."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


# Written from the listener's side: one spoken sentence that starts with the thing
# they asked for, in the file's own words. Every clause here answers a failure that
# was actually heard - "a fast LLM" instead of "Groq-hosted model", a paragraph that
# says "the document states" before it says anything, a bullet list a voice reads as
# "asterisk asterisk".
SYSTEM_PROMPT = (
    "Someone asked a spoken question about one document and hears your reply once. "
    "Answer directly in plain words that read aloud cleanly - no bullets, no markdown, "
    "no preamble such as 'the document says'. "
    "A factual lookup is one sentence. An explanation, comparison, summary or 'why/how' "
    "question may take two to four short sentences: combine the facts from as many of the "
    "passages as they need and say how they relate, rather than quoting one line. "
    "Use the conversation so far to resolve a follow-up such as 'it', 'that' or 'the second one'. "
    "Include the exact names, numbers, dates, models and technical choices the passages state "
    "(for example, 'Install selenium and pytest. [Page 1]' or 'The login form takes username tomsmith. [Page 1]'). "
    "If the passages are in another language like Hindi or Sanskrit, answer clearly in the language of the question. "
    "If a passage mentions the person or entity asked about, explain what the text states about them. "
    "A yes or no question starts with Yes or No. A how-many question starts with the number. "
    "Add nothing the passages do not state, and never mention the passages, the search or what "
    "is missing. End with the page or pages the answer draws on, like [Page 4]. "
    "If the passages cannot answer it, reply with exactly: NOT_IN_DOCUMENT"
)

NOT_FOUND_TOKEN = "NOT_IN_DOCUMENT"


def _usage_of(data: Any, style: str) -> Dict[str, int]:
    """Token counts a gateway reported, normalised to openai names, for cost measurement."""
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    if style == "gemini":
        return {
            "prompt_tokens": int(usage.get("promptTokenCount") or 0),
            "completion_tokens": int(usage.get("candidatesTokenCount") or 0),
        }
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


class GroundedLlm:
    def __init__(self, name: str, url: str, headers: Dict[str, str], model: str, style: str):
        self.name = name
        self.url = url
        self.headers = headers
        self.model = model
        self.style = style  # 'openai' or 'gemini'
        self.last_usage: Dict[str, int] = {}

    @classmethod
    def from_env(cls) -> Optional["GroundedLlm"]:
        _load_dotenv()
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            model = os.environ.get("CONVO_LLM_MODEL", "gemini-2.5-flash")
            return cls(
                name=f"gemini:{model}",
                url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                model=model,
                style="gemini",
            )

        key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if key:
            base = os.environ.get(
                "OPENAI_BASE_URL",
                "https://api.groq.com/openai/v1" if os.environ.get("GROQ_API_KEY") else "https://api.openai.com/v1",
            ).rstrip("/")
            provider = "groq" if os.environ.get("GROQ_API_KEY") else "openai"
            default_model = "llama-3.3-70b-versatile" if provider == "groq" else "gpt-4o-mini"
            model = os.environ.get("CONVO_LLM_MODEL", default_model)
            return cls(
                name=f"{provider}:{model}",
                url=f"{base}/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                model=model,
                style="openai",
            )

        return None

    def answer(self, query: str, passages: List[Tuple[int, str, str]], timeout: float = 9.0) -> str:
        """Ask the model, on a clock the caller can actually wait on.

        Nine seconds is the ceiling because the index already holds a grounded answer
        for these same passages: a model that has not replied by then must hand the
        turn over rather than leave a caller listening to silence.
        """
        return self.answer_with_history(query, passages, history=None, timeout=timeout)

    def answer_with_history(self, query: str, passages: List[Tuple[int, str, str]],
                            history: Optional[List[Dict[str, str]]] = None,
                            doc_name: str = "", note: str = "", timeout: float = 9.0) -> str:
        """Grounded answer with the turns that led to the question.

        A follow-up ("and the second one?", "why is that?") cannot be resolved from
        the current question alone, and a model that sees only the new wording has to
        guess which part of the document it refers to. The last turns are included as
        conversation, not as evidence: every claim still has to come from `passages`.

        `note` is a short fact about how this file writes things (an encoding quirk,
        a naming convention). It is not evidence and adds no page of its own.
        """
        context = "\n\n".join(
            f"[Page {page}] {f'({heading}) ' if heading else ''}{text}" for page, heading, text in passages
        )
        turns = []
        for turn in (history or [])[-3:]:
            asked = (turn.get("query") or "").strip()
            replied = (turn.get("answer") or "").strip()
            if not asked:
                continue
            turns.append(f"Caller: {asked}" + (f"\nAssistant: {replied}" if replied else ""))
        conversation = ("\n\nConversation so far:\n" + "\n\n".join(turns)) if turns else ""
        # The question comes last, after the material, and the reply is cued with
        # "Answer:" so the first words out of the model are the answer itself rather
        # than a restatement of the question.
        prompt = (
            f"Document: {doc_name or os.environ.get('CONVO_DOC_NAME', 'the document')}\n"
            f"{note + chr(10) if note else ''}"
            f"Passages:\n{context}{conversation}\n\nQuestion: {query}\nAnswer:"
        )
        return self._complete(SYSTEM_PROMPT, prompt, timeout)

    def suggest_search_terms(self, query: str, doc_hint: str = "", timeout: float = 4.0) -> str:
        """Translate and expand a question into the document's own search vocabulary.

        A reader asks in English about a Hindi book, or uses "refused to fight" where
        the text says "युद्ध नहीं करना". Lexical retrieval cannot bridge that, so the
        model is asked for the handful of words the passage that answers it would use.
        """
        prompt = (
            f"User question: {query}\n"
            f"{doc_hint}\n"
            "Give 3 to 8 essential search words or short phrases that the passage answering this "
            "question would contain in this document.\n"
            "Write them in the same language and script the document uses. If the document is in "
            "Hindi/Devanagari, output Devanagari words, including the usual spellings of any names.\n"
            "Translate the question's meaning; do not just copy the English words.\n"
            "Output ONLY the space-separated search terms, nothing else."
        )
        try:
            raw = self._complete("You extract search terms for a document. Output only the terms.", prompt, timeout)
            cleaned = " ".join(re.findall(r"[\w\u0900-\u097F]+", raw or ""))
            return cleaned
        except Exception:
            return ""

    def _complete(self, system: str, prompt: str, timeout: float) -> str:
        if self.style == "gemini":
            payload = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 220},
            }
        else:
            payload = {
                "model": self.model,
                "temperature": 0.2,
                "max_completion_tokens": 220,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }

        response = None
        # A rate-limited gateway is a pause, not an answer: two short retries keep a
        # burst of questions from dropping to the extractive path mid-conversation.
        for attempt in range(3):
            try:
                response = httpx.post(self.url, headers=self.headers, json=payload, timeout=timeout)
            except httpx.HTTPError as error:
                raise LlmAnswerError(f"network: {error}") from error
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    time.sleep(1.0 if attempt == 0 else 2.5)
                    continue
            break

        if response.status_code >= 400:
            raise LlmAnswerError(f"http {response.status_code}: {response.text[:180]}")

        try:
            data = response.json()
        except ValueError as error:
            raise LlmAnswerError(f"reply was not json: {error}") from error
        self.last_usage = _usage_of(data, self.style)

        try:
            text = self._extract_text(data)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            raise LlmAnswerError(f"unexpected reply shape: {error}") from error
        if text is None:
            raise LlmAnswerError("the model returned no text")
        return text

    def _extract_text(self, data: Any) -> Optional[str]:
        """Pull the answer out of a reply whose shape is not guaranteed.

        Gateways differ here: `content` arrives as a string, as a list of parts,
        or as a candidate the provider left half-filled, so every branch has to
        be shape-checked rather than trusted.
        """
        if not isinstance(data, dict):
            return None
        if self.style == "gemini":
            candidates = [item for item in (data.get("candidates") or []) if isinstance(item, dict)]
            content = candidates[0].get("content") if candidates else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list):
                return None
            return "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip() or None
        choices = [item for item in (data.get("choices") or []) if isinstance(item, dict)]
        message = choices[0].get("message") if choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            return None
        return content.strip() or None


_URL_RE = re.compile(r"https?://([A-Za-z0-9.\-_]+)")
_FILE_RE = re.compile(r"[A-Za-z0-9_\-.]+\.(?:xlsx|xls|csv|html|htm|log|py|json|xml|pdf|txt)\b", re.I)
_PIP_RE = re.compile(r"pip3?\s+install\s+([A-Za-z0-9 ._\-]+)", re.I)
_WAIT_RE = re.compile(r"(?:implicitly_wait|WebDriverWait\s*\(\s*\w+\s*,|time\.sleep|sleep)\s*\(\s*([^)\n]{1,30})\)", re.I)
_QUOTED_RE = re.compile(r"[\"']([^\"'\n]{1,60})[\"']")
_TEXT_NODE_RE = re.compile(r">\s*([^<>]{2,60}?)\s*<")
_SEND_KEYS_RE = re.compile(r"(?:send_keys|enter_username|enter_password|set_text|type)\s*\(\s*[\"']([^\"'\n]{1,40})[\"']", re.I)
_INLINE_CRED_RE = re.compile(r"\b(user[\s_-]?name|username|user|password|pass)\b[\"']?\s*[:=]\s*[\"']([^\"'\n]{0,40})[\"']")
_CRED_KEY_RE = re.compile(r"\b(password|pass|user[\s_-]?name|username|user)\b", re.I)
# A line is code when its shape leaves no room for English: it opens on a statement
# keyword, assigns, or is a cell row. A sentence that merely names one method is still
# a sentence - "Use alert.dismiss() to reject the dialog without confirming" is the
# answer to the question the page exists to ask, and reading it back as code costs the
# caller the only true thing the manual says about the alert.
_CODE_HARD_RE = re.compile(r"^(?:from|import|def|class|return|assert|print|self\.|driver\.|await)\b|\s\|\s|[{}=]")
_CODE_SOFT_RE = re.compile(r"(?:^|\s)(?:from|import|def|class|return|assert|print|self\.|driver\.)\b|\.\w+\(|[<>]")

# Question shape decides which needles are even eligible. "How does the manual
# handle an alert with an explicit wait?" contains *wait* but wants a mechanism,
# not the number 10.
_HOW_RE = re.compile(r"\bhow (?:does|do|did|to|can|could|should|is|are|will)|\bwhy\b|\bexplain\b|\bwhat happens|\bwalk(?: me)? through\b|\bsteps?\b", re.I)
_SCALAR_ASK_RE = re.compile(r"\b(how long|how many|number of|for how long|wait time|timeout value|max wait|what value|which value|how much)\b", re.I)


def _mentions(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


# A question word followed by anything that is not a helper verb is followed by the
# thing being asked about: "which grade is", "what username does", "which file".
_ASK_FILLER = frozenset(
    "is are was were does do did have has had will would can could should must "
    "the a an this that these those there here it they you i of for with to in on at about".split()
)


def _focus_ask(query: str) -> str:
    """The noun the caller put in front of the verb, or "" when the question has none.

    A branch that fires on any word of the question reads "which grade is a fail in the
    excel driven test" as a question about a file, because the test happens to be one.
    Requiring the question's own noun to be the one the branch answers is what keeps a
    value lookup from hijacking a question that never asked for a value.
    """
    words = re.findall(r"[A-Za-z]+", query.lower())
    for index in range(len(words) - 1):
        if words[index] in ("which", "what", "whose") and words[index + 1] not in _ASK_FILLER:
            return stem(words[index + 1])
    return ""


def _asks_for(focus: str, *needles: str) -> bool:
    """True when the question's noun is one this branch answers, or the question names none."""
    if not focus:
        return True
    for needle in needles:
        head = stem(needle.split()[0])
        if head and (head == focus or head in focus or focus in head):
            return True
    return False


def _cred_kind(prefix: str) -> str:
    # Identifier spelling (`enter_username`, `user-name`) is normalised first:
    # the underscore would otherwise swallow the word boundary.
    normalized = re.sub(r"[_.\-]+", " ", prefix.lower())
    keys = _CRED_KEY_RE.findall(normalized)
    if not keys:
        return ""
    return "password" if keys[-1].startswith("pass") else "username"


def _credential_pairs(line: str) -> List[Tuple[str, str]]:
    """Field/value pairs on one line: `send_keys("standard_user")` or `user='standard_user'`."""
    pairs: List[Tuple[str, str]] = []
    for match in _INLINE_CRED_RE.finditer(line):
        value = match.group(2).strip()
        if value:
            pairs.append(("password" if match.group(1).startswith("pass") else "username", value))
    for match in _SEND_KEYS_RE.finditer(line):
        value = match.group(1).strip()
        if not value:
            continue
        kind = _cred_kind(line[: match.start()]) or _cred_kind(line)
        if kind:
            pairs.append((kind, value))
    return pairs


def _value_candidates(line: str) -> List[str]:
    """Payloads a line carries: quoted strings and HTML text nodes."""
    out: List[str] = []
    for value in _QUOTED_RE.findall(line) + _TEXT_NODE_RE.findall(line):
        value = value.strip()
        if value and len(value.split()) <= 8 and value.lower() not in {"id", "class", "css_selector"}:
            out.append(value)
    return out


def _looks_like_code(line: str) -> bool:
    text = line.strip()
    if _CODE_HARD_RE.search(text):
        return True
    markers = len(_CODE_SOFT_RE.findall(text))
    if markers < 2:
        return False
    # Two or more code shapes in a short line means the line is mostly code; the same
    # two markers spread over a long sentence are a sentence that quotes an API.
    words = len(re.findall(r"[A-Za-z]{2,}", text))
    return markers * 3 >= words


class _Line:
    """One quotable line, tagged with where it came from and how well it matches."""

    __slots__ = ("rank", "position", "page", "text", "overlap")

    def __init__(self, rank: int, position: int, page: int, text: str, overlap: float):
        self.rank = rank
        self.position = position
        self.page = page
        self.text = text
        self.overlap = overlap


def _needle(query: str, hits, rag) -> List[_Line]:
    """Flatten retrieved passages into lines weighted by how well they match.

    Line scoring uses the question's words only. The ordinal expansion that
    `search` relies on ("first" -> 1, to reach the heading *Practical 1*) is
    deliberately left out here: it would otherwise match `Site 1 Title:` and
    promote an unrelated line inside the right passage.
    """
    terms = list(Counter(tokenize(query)))
    # A line that carries a synonym of the question's word is the line the question is
    # about: "the report is written to results.xlsx" answers "which file holds the
    # output" although it never says output, and the line above it says input.
    partners = rag.partner_map(terms)
    out: List[_Line] = []
    for rank, hit in enumerate(hits):
        position = 0
        for raw in hit.text.split("\n"):
            line = raw.strip()
            # A single-token line is still a real fact: `driver.get("https://...")`
            # has no spaces but carries the whole answer.
            if len(line) < 4:
                continue
            if not _legible(line):
                continue
            line = _standalone(line)
            position += 1
            line_terms = set(tokenize(line))
            # A word the caller actually used is worth more than a synonym of it: the
            # synonym only breaks a tie, so a line that answers in the question's own
            # wording is never displaced by one that merely means the same.
            overlap = sum(rag.idf(term) for term in terms if term in line_terms)
            overlap += sum(0.5 * rag.idf(term) for term in terms
                           if term not in line_terms
                           and any(partner in line_terms for partner in partners.get(term, ())))
            out.append(_Line(rank, position, hit.page_num, line, overlap))
    return out


def _pick(lines: List[_Line], require_overlap: bool = False) -> Optional[_Line]:
    """Best matching line: most question terms first, then closest to rank 0."""
    if not lines:
        return None
    ordered = sorted(lines, key=lambda line: (-line.overlap, line.rank, line.position))
    best = ordered[0]
    if best.overlap > 0:
        return best
    if require_overlap:
        return None
    return min(lines, key=lambda line: (line.rank, line.position))


def _citation(pages: List[int]) -> str:
    return " ".join(f"[Page {page}]" for page in sorted(set(pages)))


# --------------------------------------------------------------------------- facts

_KV_LINE_RE = re.compile(r"^\s*([^:=\n|]{2,40}?)\s*[:=]\s*(\S.+)$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[a-z][)]|[-*\u2022\u25cf])\s+")
_SUPERLATIVES = {
    "lowest": -1, "minimum": -1, "least": -1, "cheapest": -1, "smallest": -1, "shortest": -1,
    "fastest": -1, "fewest": -1, "highest": 1, "maximum": 1, "most": 1, "costliest": 1,
    "dearest": 1, "biggest": 1, "largest": 1, "longest": 1, "slowest": 1,
}
_TO_MONTHS = {"year": 12.0, "month": 1.0, "week": 0.25, "day": 1 / 30.0, "hour": 1 / 720.0,
              "minute": 1 / 43200.0, "second": 1 / 2592000.0, "ms": 1 / 2592000000.0}


class _Unit:
    """One quotable fact, together with the name the document gave it."""

    __slots__ = ("page", "rank", "position", "label", "value", "key", "kind")

    def __init__(self, page: int, rank: int, position: int, label: str, value: str, key: str, kind: str):
        self.page = page
        self.rank = rank
        self.position = position
        self.label = label
        self.value = value
        self.key = key
        self.kind = kind  # cell | kv | item | code | prose | row


def _quantity(value: str) -> Optional[float]:
    """A value measured on one scale, so rows of a column can be compared.

    The number has to lead the cell, optionally with a unit after it: `6 months` and
    `12499` are measurements, while `APDCR RCT-6000D` and `Type 2` merely contain
    digits and would otherwise be sorted as if they were quantities.
    """
    match = re.match(r"\s*(\d+(?:[.,]\d+)?)\s*(year|month|week|day|hour|minute|second|ms)?", value or "", re.I)
    if not match:
        return None
    try:
        number = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = (match.group(2) or "").lower()
    return number * _TO_MONTHS.get(unit, 1.0)


_ENUM_RE = re.compile(r"\b(list|enumerate|outline|all|every|each|what are the|give me the)\b", re.I)
# A question that opens on its auxiliary can only be answered yes or no. It is a closed
# class of English words and names no subject a document could be about.
_POLAR_ASK_RE = re.compile(
    r"^\s*(?:is|are|was|were|does|do|did|has|have|had|can|could|will|would|should|may)\b", re.I)
# A question that opens by asking who or what a thing *is* wants a description, not a
# value. Closed class of two interrogatives plus the copula.
_DEFINE_ASK_RE = re.compile(r"^\s*(?:who|what)\s+(?:is|are|was|were)\b", re.I)
_DOUBLE_ASK_RE = re.compile(r"\band\b.*\b(what|which|how|who|where|does|should|many|much)\b", re.I)
_SECTIONS_ASK_RE = re.compile(r"\b(sections|headings|chapters|contents|table of contents|structure|titles)\b", re.I)
# A question that names one structural part ("the first chapter") wants that part's
# material, not the whole file. Steps and practicals are excluded: those ask for a
# position inside a section, which the ordinal path already reads.
_PART_ASK_RE = re.compile(r"\b(chapter|section|part|unit|module)\b", re.I)
# The line a Devanagari book prints to close a chapter: "... प्रथमोऽध्यायः ॥१॥",
# which a legacy font stores as "... थमोऽयायः". It is the document's own boundary
# mark, so it survives whatever the typography did to the chapter numbers. The
# closing visarga matters: a verse *about* chapters ("योऽयाया कतताः") is not the
# end of one.
_CHAPTER_MARK_RE = re.compile(r"ोऽयायः")
_STEPS_ASK_RE = re.compile(r"\b(steps?|items?|points?|rules?|checklist|requirements|stages|phases)\b", re.I)
# "What is the first step of practical 3" asks for a position in the file's own order.
# The word is a closed class in every language, and the answer is the nth thing the
# section says, so it is read from the section rather than ranked against it.
_ORDINAL_ASK_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"1st|2nd|3rd|4th|5th|last)\s+(step|part|point|item|line|stage|phase|rule|thing)\b",
    re.I,
)


def _ordinal_ask(query: str) -> Optional[int]:
    """1-based position the caller asked for, or None when the question has no order."""
    match = _ORDINAL_ASK_RE.search(query)
    if not match:
        return None
    word = match.group(1).lower()
    if word == "last":
        return -1
    digits = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}
    if word in digits:
        return digits[word]
    return ["first", "second", "third", "fourth", "fifth",
            "sixth", "seventh", "eighth", "ninth", "tenth"].index(word) + 1
# Asking about the file as a whole, rather than about anything inside it. Another closed
# class: every language has a few of these and they all want the same three facts -
# what it is called, how long it is, and what it is organised into.
_ABOUT_ASK_RE = re.compile(
    r"\b(?:what (?:is|are|was|were)\s+(?:this|the|that)\s+(?:document|file|pdf|manual|guide|book|paper|story)\s*"
    r"(?:about|covering|contain(?:ing)?)?|summar(?:ise|ize|y|y)\s+(?:the|this|it|whole)|"
    r"(?:give|tell)\s+me\s+(?:an|the)?\s*(?:overview|summary|brief)|"
    r"about\s+(?:this|the)\s+(?:document|file|pdf|book|story)|"
    r"what\s+(?:does|do)\s+(?:this|the)\s+(?:document|file|pdf)\s+(?:say|cover|talk\s+about)|"
    r"(?:who|what)\s+(?:is|are|was|were)\s+(?:the\s+)?(?:main|key|central|primary|major)\s+(?:character|characters|topic|topics|finding|findings|purpose|theme|themes|person|people))\b",
    re.I,
)
# A reader who asks for a *run* of the file - summarise it, explain it, walk through
# it - wants the section in order, not the one sentence inside it that repeats their
# wording. Another closed class of verbs, and it names no subject of any document.
_RECAP_ASK_RE = re.compile(
    r"\b(?:summar(?:ise|ize|y)|recap|outline|descri(?:be|ble)|explain|recount|"
    r"walk\s+(?:me\s+)?through|go\s+over|tell\s+me\s+about|give\s+me)\b",
    re.I,
)

_META_RE = re.compile(
    r"\b(?:the\s+)?(?:provided|given|supplied|retrieved|passed)\s+"
    r"(?:(?:document|file|source)\s+)?(?:passages?|context|excerpts?|text|information|snippets?|documents?)\b[^.]{0,60}"
    r"\b(?:do(?:es)?\s+not|did\s+not|not|cannot|can'?t)\b",
    re.I,
)

# A question that wants reasoning, not a single value: the answer is a relation
# between passages, so one 40-word strip is not enough material to build it from.
_ANALYTICAL_ASK_RE = re.compile(
    r"\b(?:compar(?:e|ison|ing)|differ(?:ence|ences|ent|s)?|relationship|relat(?:e[sd]?|ion)|"
    r"contrast|why|reasons?|explain|analys(?:e|is)|analyze|impact|effect|consequence|"
    r"significance|advantage|disadvantage|pros?\s+and\s+cons?|how\s+(?:does|do|did))\b",
    re.I,
)

# A pronoun or a reference back to the previous turn. A follow-up is resolved
# against the last question before retrieval, or "and the second one?" searches
# for "second" in a book that never says it.
_FOLLOW_UP_RE = re.compile(
    r"\b(?:it|its|this|that|these|those|they|them|their|he|him|his|she|her|hers|"
    r"the\s+(?:first|second|third|fourth|fifth|last|next|previous|same)|"
    r"more|another|above|previous|earlier)\b",
    re.I,
)

# The nouns a question uses when it means the file itself, not something inside it.
_DOC_NOUNS = frozenset(
    "document documents file files pdf pdfs manual guide book paper story text content contents "
    "report handbook chapter".split()
)


def _about_document(query: str, rag=None) -> bool:
    """True when the question asks what the file is, not about something in it.

    "What is this document about" and "what does the document say about Draupadi"
    open the same way, but the second names a subject and is a normal lookup. Handing
    the cover pages to a model for the second one is how the publisher's phone number
    became the answer to a question about Draupadi.
    """
    if not _ABOUT_ASK_RE.search(query):
        return False
    # "summarise the first chapter" opens like "summarise the document", but it names
    # one part. The index has that part; the overview does not.
    if re.search(r"\b(?:chapter|module|unit|section|part|practical|experiment|page|step)\b", query, re.I) \
            and re.search(r"\b(?:\d+|first|second|third|fourth|fifth|last)\b", query, re.I):
        return False
    tail = re.search(
        r"\b(?:about|regarding|concerning|on)\s+(?:the\s+|this\s+)?([A-Za-z\u0900-\u097F][\w\u0900-\u097F-]*)",
        query,
        re.I,
    )
    if tail and tail.group(1).lower() not in _DOC_NOUNS:
        return False
    return True


def _cross_lingual(query: str, rag: PDFRAGEngine) -> bool:
    """Does the question speak a different vocabulary from the document itself.

    An English question against a Devanagari book shares no word with it, so BM25
    can only match stray numbers and names. Fewer than half of the question's
    content words being document vocabulary is the signal to translate. A question
    whose words are all stop words ("Who is Karna?", where *karna* is the Hinglish
    verb) has no search terms at all, and needs the model to name what it means.
    """
    terms = [term for term in query_terms(query) if term not in _SPEECH_ACT]
    latin_question = bool(re.search(r"[A-Za-z]{3,}", query)) and not re.search(r"[\u0900-\u097F]", query)
    if getattr(rag, "is_devanagari", False) and latin_question:
        # A Latin-script question about a Devanagari book is a translation problem
        # by definition; a name the transliteration table happens to know is not
        # enough to answer the sentence around it.
        return True
    if not terms:
        return bool(re.search(r"[A-Za-z\u0900-\u097F]{3,}", query))
    partners = rag.partner_map(terms)
    known = sum(1 for term in terms if rag.df(term) or partners.get(term))
    return known * 2 < len(terms)


def _doc_hint(rag: PDFRAGEngine) -> str:
    hint = f"Document: {_friendly_name(rag.current_filename)}. Total pages: {rag.total_pages}."
    if getattr(rag, "is_devanagari", False):
        hint += " The document text is in Hindi/Devanagari script."
    return hint

# Words that are not about the document at all. This is a closed class of greetings,
# acknowledgements and questions about the assistant, which every language carries a
# handful of; it is not a topic list, and nothing here is keyed to any subject a PDF
# could contain. A greeting that reaches the retrieval guardrail comes back as "I could
# not locate that", which makes a working system sound broken.
_GREETING_RE = re.compile(
    r"^\s*(?:hi+|hey+|hello+|yo|hola|namaste|namaskar|salaam|"
    r"good\s+(?:morning|afternoon|evening|day))\b[\s,.!?]*$",
    re.I,
)
_SMALL_TALK_RE = re.compile(
    r"\b(?:who are you|what are you|what can you do|what do you do|how are you|"
    r"are you there|can you hear me|what is your name|are you a bot)\b",
    re.I,
)
_DOC_INFO_RE = re.compile(
    r"\b(?:which|what)\s+(?:pdf|document|file)\s+(?:do\s+you\s+have|is\s+(?:currently\s+|now\s+)?(?:attached|loaded|open|uploaded|connected|in\s+use)|is\s+this)\b"
    r"|\b(?:which|what)\s+(?:is\s+the\s+)?(?:name|title)\s+of\s+(?:the\s+|this\s+)?(?:pdf|document|file)\b"
    r"|\b(?:is\s+(?:there\s+)?(?:a|any)\s+(?:pdf|document|file)\s+(?:attached|loaded|open|connected))\b"
    r"|\b(?:attached|loaded|current|connected)\s+(?:pdf|document|file)\b"
    r"|\b(?:kon|kaun)\s*sa\s*(?:pdf|document|file)\b",
    re.I,
)
_GREETING_PREFIX_RE = re.compile(
    r"^\s*(?:hi+|hey+|hello+|yo|hola|namaste|namaskar|salaam|"
    r"good\s+(?:morning|afternoon|evening|day)|please|kindly)\b[,\s.:!?-]*",
    re.I,
)
_ACK_RE = re.compile(
    r"^\s*(?:thanks|thank you|thx|ty|ok(?:ay)?|okey|nice|great|good|cool|wow|sorry|cool thanks)"
    r"[\s,.!?]*$",
    re.I,
)
# Telling the assistant its answer was wrong is not a new question, and answering it
# by quoting another random line is what made the app feel broken. It is a request to
# try the previous question again with more of the file in front of it.
_CORRECTION_RE = re.compile(
    r"^\s*(?:no[,.]?\s*)?(?:this|that|it|your)?\s*(?:is|was|'s|are|seems|looks)?\s*"
    r"(?:not|n't)\s+(?:a\s+)?(?:correct|right|true|good|useful|helpful|accurate|what i asked)"
    r"[\s\S]{0,24}$|^\s*(?:wrong|incorrect|that'?s (?:wrong|incorrect|not right)|no not|"
    r"try again|search again|look again|that is not it|not this)\s*[.!?]*$",
    re.I,
)


class _Ask:
    """The question's own words, weighted the way the index weights them.

    Every layer that decides *which* piece of the document answers a question measures
    with this one object. That is what replaces the hand-written topic vocabulary the
    first version relied on: the decision is the same idf-weighted word measurement
    retrieval already makes, applied to a label, a row, or a sentence.
    """

    def __init__(self, query: str, rag) -> None:
        self.terms = list(Counter(query_terms(query)))
        self.partners = rag.partner_map(self.terms)
        self._idf = {term: rag.idf(term) for term in set(self.terms)}
        self._known = {term: rag.df(term) > 0 for term in set(self.terms)}

    def found(self, text: str) -> List[str]:
        if not text:
            return []
        words = set(tokenize(text))
        return [term for term in self.terms
                if term in words or any(partner in words for partner in self.partners.get(term, ()))]

    def weight(self, text: str) -> float:
        """How much of what the question asks for this text actually says."""
        return sum(self._idf.get(term, 0.0) for term in set(self.found(text)))

    def coverage(self, text: str, terms: List[str]) -> float:
        """Share of the ask that this text answers, counted in idf mass.

        Words the document has never used are left out of the denominator: no row of a
        grid can be blamed for not saying *carries*, and holding that against a row
        would refuse questions that use a synonym for the document's own word.
        """
        askable = [term for term in set(terms) if self._known.get(term)]
        if not askable:
            return 1.0
        found = set(self.found(text)) & set(askable)
        return sum(self._idf.get(term, 0.0) for term in found) / (
            sum(self._idf.get(term, 0.0) for term in askable) or 1.0
        )


def _ask_quantity(query: str, terms: List[str]) -> Optional[float]:
    """The number the question spells out, on the scale a table cell uses.

    "A five year warranty" and the cell `5 years` share no token the index can see:
    one is a word, the other a digit glued to a unit. Converting both to months is what
    lets a row be chosen for matching the amount the caller actually said.
    """
    numbers = [float(token) for token in terms if re.fullmatch(r"\d+(?:[.,]\d+)?", token or "")]
    if not numbers:
        return None
    unit = next((name for name in _TO_MONTHS if re.search(rf"\b{name}s?\b", query.lower())), "")
    return numbers[0] * _TO_MONTHS.get(unit, 1.0)


def _table_answer(query: str, rag) -> Optional[Tuple[str, List[int]]]:
    """Answer from the table index, where a column means the same thing in every row.

    Retrieval hands back passages, but a grid question is not about a passage: "which
    item has the lowest warranty period" needs every row of the warranty column,
    including rows on the far side of a page break, and those rows never carry the word
    *warranty* at all. So the column is found by name, the rows are pooled from the
    structure the chunker recorded, and the answer is read out of the chosen row.
    """
    if _ENUM_RE.search(query) or not rag.rows:
        return None
    ask = _Ask(query, rag)
    labels: List[str] = []
    for row in rag.rows:
        for label in row.labels:
            if label not in labels:
                labels.append(label)
    asked = sorted((label for label in labels if ask.weight(label) > 0), key=lambda label: -ask.weight(label))
    if not asked:
        # The question can name what a cell *says* instead of the column it sits in:
        # "the maximum value of the limit parameter" matches no header in the sheet, but
        # the row that carries it says both words. Two of the question's own words, in
        # one row, and no other row matching as well is the bar - a single shared word
        # would read out whichever row retrieval happened to reach first.
        ranked = sorted(rag.rows, key=lambda row: -len(set(ask.found(row.text))))
        best = ranked[0]
        matched = set(ask.found(best.text))
        runner_up = len(set(ask.found(ranked[1].text))) if len(ranked) > 1 else 0
        if len(matched) < 2 or len(matched) <= runner_up or ask.coverage(best.text, ask.terms) < 0.5:
            return None
        return _row_say(best, best.labels[0]), [best.page]
    focus_label = asked[0]
    with_focus = [row for row in rag.rows if focus_label in row.labels and row.value(focus_label)]
    if not with_focus:
        return None
    ident = Counter(row.labels[0] for row in with_focus).most_common(1)[0][0]
    pool = [row for row in with_focus if row.labels[0] == ident]
    if not pool:
        return None
    # The row's own name is rarely the thing the question is after. "Which item has the
    # lowest warranty period" uses *item* the way a person means *row*, and the column
    # to compare is the warranty one, so the first named column that is not the identity
    # wins.
    focus = next((label for label in asked if label != ident), ident)

    word = next((term for term in _SUPERLATIVES if re.search(rf"\b{term}\b", query.lower())), "")
    if word:
        measured = [(_quantity(row.value(focus)), row) for row in pool if _quantity(row.value(focus)) is not None]
        if len(measured) < 2:
            return None
        pick = min if _SUPERLATIVES[word] < 0 else max
        _, row = pick(measured, key=lambda item: item[0])
        return _row_say(row, ident, f"The {word} {focus.lower()} is on the "), [row.page]

    want = _ask_quantity(query, ask.terms)

    def row_weight(row) -> float:
        score = ask.weight(row.text)
        if want is not None:
            for label, cell in zip(row.labels, row.cells):
                if label == ident:
                    continue
                measured = _quantity(cell)
                if measured is not None and abs(measured - want) < 1e-6:
                    score += 1.5  # the row carries the amount the caller spoke
        return score

    ranked = sorted(pool, key=lambda row: -row_weight(row))
    best_weight = row_weight(ranked[0])
    # A row is chosen only when a word the question itself brings pins it, and no other
    # row matches as well: "which item is in stock" would otherwise name whichever row
    # the retrieval happened to read first.
    if best_weight < 1.0 or (len(ranked) > 1 and best_weight <= row_weight(ranked[1])):
        return None
    row = ranked[0]
    # Then the row has to carry the question, not just answer to one of its words. The
    # column names are discounted because every row shares them; what is left is the
    # part of the question that is supposed to pick this row out of the grid. A row that
    # holds only a third of it means the question is about prose elsewhere in the file -
    # "what does installation of a hybrid inverter cost" names a hybrid inverter that is
    # in the grid, but the price it asks for is in a sentence that is not.
    label_tokens = {token for label in asked for token in tokenize(label)}
    residual = [term for term in ask.terms if term not in label_tokens]
    # A question that speaks the row's own key - "what does HTTP 429 mean and what
    # should the client do" - has already said which row it wants, and the row is the
    # only place the rest of the question is answered. Nothing has to be left over.
    figures = [value for value in (_quantity(cell) for cell in row.cells) if value is not None]
    keyed = want is not None and any(abs(value - want) < 1e-6 for value in figures)
    if not keyed and ask.coverage(row.text, residual) < 0.7:
        return None
    if focus != ident and ask.weight(row.value(ident)) > 0 and not _DOUBLE_ASK_RE.search(query):
        answer = f"The {focus.lower()} of the {row.value(ident)} is {row.value(focus)}"
        return f"{answer.rstrip('.')}. {_citation([row.page])}.", [row.page]
    return _row_say(row, ident), [row.page]


def _row_say(row, ident: str, lead: str = "") -> str:
    """Read one row aloud: the thing itself, then what the document records about it."""
    parts = []
    for label, cell in zip(row.labels, row.cells):
        if label == ident or not cell:
            continue
        parts.append(f"{label if label.isupper() else label.lower()} {cell}")
    body = ", ".join(parts[:6])
    text = f"{row.value(ident)}, {body}" if body else row.value(ident)
    if lead:
        return f"{lead}{text.rstrip('.')}. {_citation([row.page])}."
    return f"The {text.rstrip('.')}. {_citation([row.page])}."


def _units(hits) -> List[_Unit]:
    """Turn passages into labelled facts, with no idea what the document is about.

    Four shapes cover what PDFs actually contain: a table row, which the chunker has
    already written as `Warranty: 2 years | Price: 14990`; a definition or assignment,
    which reads `Term: meaning` or `password = "secret_sauce"`; a list item; and
    everything else as sentences. Which fact answers the question is then decided by
    matching the question's words against the *label* - so the same code answers "what
    is earthing", "which SKU costs 340" and "what does 429 mean" without a single
    hand-written topic list, which is what made the old version a prototype.
    """
    out: List[_Unit] = []
    for rank, hit in enumerate(hits):
        position = 0
        last_label = ""
        for raw in hit.text.split("\n"):
            line = raw.strip()
            if len(line) < 4:
                continue
            # Every unit below can become the sentence that is read aloud, so the same
            # two tests apply here as in `_candidate_lines`: no scanner residue, and no
            # opening clause that belongs to the line above it.
            if not _legible(line):
                continue
            position += 1
            if " | " in line:
                # A grid row is not a fact of its own: columns are compared, and the
                # whole row is read out. That is `_table_answer`'s job, done against the
                # row index so it can see rows this passage does not contain.
                last_label = ""
                continue
            if _looks_like_code(line):
                last_label = ""
                for kind, value in _credential_pairs(line):
                    out.append(_Unit(hit.page_num, rank, position, kind, value, "", "kv"))
                out.append(_Unit(hit.page_num, rank, position, "", line, "", "code"))
                continue
            match = _KV_LINE_RE.match(line)
            if match and match.group(1).strip() and not match.group(1).endswith(("http", "https")):
                last_label = match.group(1).strip()
                out.append(_Unit(hit.page_num, rank, position, last_label,
                                 match.group(2).strip(), "", "kv"))
                continue
            if looks_like_heading(line):
                last_label = ""
            stripped = _LIST_ITEM_RE.sub("", line).strip()
            if stripped and stripped != line:
                last_label = ""
                out.append(_Unit(hit.page_num, rank, position, "", stripped, "", "item"))
                continue
            # A handbook entry or a glossary term is written over several lines and only
            # the first carries the name: `3.3 Earned Leave: fifteen days per year.` then
            # `Earned leave may be accumulated up to sixty days.`. The second line is the
            # same entry's fact, and it is usually the one the question wants, so it takes
            # the label over - but only when it repeats a word of that label, which stops
            # an unrelated paragraph below the entry from being read as part of it.
            inherited = bool(last_label) and not looks_like_heading(line) and bool(
                set(tokenize(last_label)) & set(tokenize(line))
            )
            for sentence in (split_sentences(line) or [line]):
                sentence = _standalone(sentence.strip())
                if len(sentence) > 3:
                    out.append(_Unit(hit.page_num, rank, position, last_label if inherited else "",
                                     sentence, "", "kv" if inherited else "prose"))
    return out


def _say(label: str, key: str, value: str) -> str:
    """One fact, phrased the way a person would say it out loud."""
    name = (label or "").strip(" .:").lower()
    text = value.strip().rstrip(".")
    if key and key != text and name:
        return f"The {name} of {key} is {text}"
    if name:
        # A short value is a predicate; a long one is its own sentence, and forcing
        # "is" in front of it turns a definition into broken English.
        if len(text.split()) > 6:
            return f"{label.strip().rstrip(':')}: {text}"
        return f"{label.strip().rstrip(':')} is {text}"
    return text


def _narrow(value: str, ask: _Ask) -> str:
    """Keep the clause of a long fact that carries the answer.

    A handbook writes `Sick Leave: ten days per year, available from the first month.
    Sick leave of more than three consecutive days requires a medical certificate`, and
    asked *when* the certificate is needed, reading the whole entry out loud wastes the
    caller's time on the quota. The label is still the subject, so nothing is invented.
    """
    sentences = split_sentences(value) or []
    if len(sentences) < 2:
        return value
    ranked = sorted(sentences, key=lambda sentence: (-ask.weight(sentence.strip()), len(sentence)))
    best = ranked[0].strip()
    return best if ask.weight(best) > 0 else value


def _unit_answer(query: str, hits, rag) -> Optional[Tuple[str, List[int]]]:
    """Pick the labelled fact the question asks for, or None to let the sentence path try."""
    if _ENUM_RE.search(query):
        return None  # an enumeration needs every item, not the best-scoring one
    units = _units(hits)
    if not units:
        return None
    ask = _Ask(query, rag)

    scored: List[Tuple[float, _Unit]] = []
    for unit in units:
        if not unit.label or len(unit.value.split()) > 30:
            continue  # an unlabelled line is the sentence path's, a paragraph is nobody's
        # Measured as a share of the ask rather than a raw idf sum, because in a short
        # document no word is common enough for an absolute score to clear: the question
        # may consist of one word, and then naming that word exactly is a perfect match
        # worth 0.69, not a failure.
        named = ask.coverage(unit.label, ask.terms)
        if named <= 0.0:
            continue
        score = 3.0 * named + 0.6 * ask.weight(unit.value) - 0.5 * unit.rank
        if looks_like_heading(unit.value):
            score -= 2.0  # a title is never the answer to a question about a value
        scored.append((score, unit))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1].rank, item[1].position))
    best = scored[0][1]
    # The name of the fact has to carry most of what the question asks. An absolute idf
    # floor cannot work here, because a short document never makes a word rare enough:
    # in a two-page checklist, *earthing* is worth 0.94 no matter how exactly the
    # question names it.
    if ask.coverage(best.label, ask.terms) < 0.5:
        return None

    q_lower = query.lower()
    extreme = next((_SUPERLATIVES[word] for word in _SUPERLATIVES
                    if re.search(rf"\b{word}\b", q_lower)), 0)
    if extreme:
        group = [unit for _, unit in scored if unit.label == best.label and _quantity(unit.value) is not None]
        if len(group) > 1:
            chosen = min(group, key=_quantity) if extreme < 0 else max(group, key=_quantity)
            answer = _say(chosen.label, chosen.key, chosen.value)
            return f"{answer.rstrip('.')}. {_citation([chosen.page])}.", [chosen.page]

    answer = _say(best.label, best.key, _narrow(best.value, ask))
    return f"{answer.rstrip('.')}. {_citation([best.page])}.", [best.page]


def _needle_answer(query: str, hits, rag) -> Optional[Tuple[str, List[int]]]:
    """Answer 'which file / what username / which site' questions with a value copied out.

    Every value is taken verbatim from a retrieved line, and the page cited is the
    page that line came from, so sources can never disagree with the answer text.
    """
    q = query.lower()
    focus = _focus_ask(query)
    lines = _needle(query, hits, rag)
    if not lines:
        return None
    asks_how = bool(_HOW_RE.search(q))
    asks_scalar = bool(_SCALAR_ASK_RE.search(q))

    if _mentions(q, "pip", "package", "library", "librar", "dependenc", "install", "setup", "environment"):
        chosen = _pick([line for line in lines if _PIP_RE.search(line.text)])
        if chosen:
            # The capture runs to the end of the line, so prose after the last
            # package ("... to prepare the lab") has to be dropped. Packages are
            # the leading run: the first English function word ends the list, and
            # nothing after it can be one.
            tokens = _PIP_RE.search(chosen.text).group(1).split()
            names: List[str] = []
            for token in tokens:
                if token.lower().strip(".,;:") in STOP_WORDS:
                    break
                names.append(token.rstrip(".,;:"))
            if names:
                # Short enough to read out, but a cap must not quietly drop a
                # package the user then fails to install.
                shown = names[:8]
                rest = len(names) - len(shown)
                packages = ", ".join(shown) + (f", plus {rest} more on that line" if rest else "")
                return f"Set it up with pip install {packages} {_citation([chosen.page])}.", [chosen.page]

    if _mentions(q, "username", "user name", "password", "credential", "log in with", "signed in as") \
            and _asks_for(focus, "username", "user", "password", "credential", "login", "log"):
        found: Dict[str, Tuple[str, int]] = {}
        for rank in sorted({line.rank for line in lines}):
            group = sorted((line for line in lines if line.rank == rank), key=lambda line: line.position)
            for line in group:
                for kind, value in _credential_pairs(line.text):
                    found.setdefault(kind, (value, line.page))
            if "username" in found and "password" in found:
                break
        if found:
            if len(found) == 2:
                user, user_page = found["username"]
                password, password_page = found["password"]
                if user_page == password_page:
                    return (
                        f"The username is {user} and the password is {password} {_citation([user_page])}.",
                        [user_page],
                    )
                return (
                    f"The username is {user} [Page {user_page}] and the password is {password} [Page {password_page}].",
                    [user_page, password_page],
                )
            value, page = next(iter(found.values()))
            return f"The login value is {value} {_citation([page])}.", [page]

    if _mentions(q, "file", "excel", "workbook", "csv", "filename", "sheet", "output file") \
            and _asks_for(focus, "file", "excel", "workbook", "csv", "filename", "sheet"):
        chosen = _pick([line for line in lines if _FILE_RE.search(line.text)])
        if chosen:
            names: List[str] = []
            for name in _FILE_RE.findall(chosen.text):
                if name not in names:
                    names.append(name)
            if names:
                listed = " and ".join(names[:2]) if len(names) > 1 else names[0]
                return f"The file is {listed} {_citation([chosen.page])}.", [chosen.page]

    if not asks_how and _mentions(q, "option", "value", "title", "heading", "label", "assert", "course", "message", "text", "error") \
            and _asks_for(focus, "option", "value", "title", "heading", "label", "assert", "course", "message", "text", "error"):
        chosen = _pick([line for line in lines if _value_candidates(line.text)], require_overlap=True)
        if chosen:
            values = _value_candidates(chosen.text)
            value = max(values, key=lambda item: (len(item.split()), len(item)))
            return f"The value is {value} {_citation([chosen.page])}.", [chosen.page]

    if _mentions(q, "website", "web site", "url", "link", "domain", "site") and not _mentions(q, "assertion", "title"):
        chosen = _pick([line for line in lines if _URL_RE.search(line.text)])
        if chosen:
            hosts: List[str] = []
            for host in _URL_RE.findall(chosen.text):
                if host not in hosts:
                    hosts.append(host)
            if hosts:
                listed = hosts[0] if len(hosts) == 1 else f"{hosts[0]} and {hosts[1]}"
                return f"The site used is {listed} {_citation([chosen.page])}.", [chosen.page]

    if asks_scalar and not asks_how:
        chosen = _pick([line for line in lines if _WAIT_RE.search(line.text)], require_overlap=True)
        if chosen:
            return f"The wait is set to {_WAIT_RE.search(chosen.text).group(1).strip()} {_citation([chosen.page])}.", [chosen.page]

    return None


# Words that cannot finish an English sentence: determiners, prepositions,
# conjunctions, possessives, objects of a clause that never came. A run of text that
# stops on one of these was cut by the page rather than by the writer. Verbs and
# adverbs are deliberately absent - "where Krishna is, there victory must be." is a
# complete sentence, and every language that puts its verb last writes them.
_NON_FINAL = frozenset(
    "a an the of in on at to by for with without into onto upon and but or nor as than "
    "from his her its their our your my thine which whose whom he she they we i".split()
)


def _dangling_tail(text: str) -> bool:
    """True when a run of words stops on a word that cannot end a sentence.

    This is how a page-cut line and a mis-detected heading look: "Arjuna. And",
    "the history of". Neither is a title and neither is an answer.
    """
    tail = text.rstrip(".!?:;,)\"'").split()
    return bool(tail) and len(tail) < 20 and tail[-1].lower().rstrip(".,;:\"'()") in _NON_FINAL


# Characters that have no business inside a spoken sentence. An OCR book produces them
# in its running heads and broken lines (";^| MAHABHARATA."), and quoting one reads out
# a stutter rather than an answer.
_OCR_JUNK_RE = re.compile("[~^`|\\\\\u00b1\u00f7\u00d7\ufffd]")


def _legible(text: str) -> bool:
    """True when a line is words rather than the residue of a scanned page.

    A grid row is written with pipes and is not prose, so it is exempt: `_table_answer`
    reads those rows by column, and rejecting them here would refuse a question the
    document does answer.
    """
    if " | " in text:
        return True
    return not _OCR_JUNK_RE.search(text)


def _standalone(text: str) -> str:
    """Quote only the part of a line that can stand as its own sentence.

    A scanned book wraps mid-thought, so the line a page hands you often begins with the
    tail of the previous one: "all must be, and where Krishna is, there victory must be."
    The first clause carries nothing, and a caller hears it as a broken record. Cutting at
    the first comma and keeping the rest is what makes the quote sayable; the words are the
    book's own and nothing is added but the capital that starts a sentence.
    """
    if not re.match(r"^[a-z]", text):
        return text
    if " | " in text or _looks_like_code(text):
        # Code and grid rows carry their payload in the leading tokens; trimming the
        # head of `assert "..." in driver.page_source` would throw the answer away.
        return text
    parts = re.split(r"(?<=[,;])\s+", text, maxsplit=1)
    if len(parts) > 1 and len(parts[-1].split()) >= 4:
        tail = parts[-1].strip()
        return tail[:1].upper() + tail[1:]
    return text


def _candidate_lines(hit, ask: Optional["_Ask"] = None) -> List[Tuple[int, str, bool]]:
    """Line-level quotable units, with a real section title offered as prose.

    A heading is the name of a section, not a statement about its subject, so it is
    only a possible answer when the question itself names that section: "what is the
    aim of practical 2" is answered by the Practical 2 title, while "what is software
    testing" is not answered by whichever practical happens to use the word testing.
    """
    out: List[Tuple[int, str, bool]] = []
    if hit.heading and SECTION_TITLE_RE.match(hit.heading):
        title = hit.heading.rstrip(".:")
        if ask is None or ask.coverage(title, ask.terms) >= 0.6:
            out.append((hit.page_num, title, True))
    # The model is handed the same flow: a page that set three words per line has to be
    # read back as sentences before any of it can be quoted to a caller.
    for raw in [line.strip() for line in flow_lines(hit.text).split("\n") if line.strip()]:
        for part in (split_sentences(raw) or [raw]):
            part = part.strip()
            if len(part.split()) < 3:
                continue
            # A line whose words are still recoverable is quotable; one that carries the
            # scanner's residue is not, however well it matches the question.
            if not _legible(part):
                continue
            part = _standalone(part)
            # A line that opens lowercase and carries no capital or figure is the
            # middle of someone else's sentence - which is what a scanned book with
            # one word per text span produces. Quoting it gives a caller a fragment
            # with a page number, so it is not offered as something to say out loud.
            # A short run that opens mid-sentence is a piece of someone else's line,
            # which is what a scanned page hands you. A long one that happens to start
            # where the previous passage ended is still a sentence worth quoting.
            fragment = len(part.split()) < 8 and not re.search(
                r"[A-Z\u0900-\u097F\u0E00-\u0E7F0-9]", part[:2]
            ) and not any(character.isdigit() for character in part)
            # A line that stops on "as", "the" or "and" was cut by the page, not by the
            # sentence, and a caller hears half a thought presented as a fact with a page.
            dangling = _dangling_tail(part)
            out.append((hit.page_num, part, not _looks_like_code(part) and not fragment and not dangling))
    return out


def _trim(line: str, words: int = 24) -> str:
    parts = line.split()
    if len(parts) <= words:
        return line
    return " ".join(parts[:words]).rstrip(",;:") + "."


def _digest_answer(hits) -> Tuple[str, List[int]]:
    """A short reading of the passages for a summary request the model could not polish.

    A caller who asks for a chapter hears what that chapter contains: one sentence
    from each of a few distinct pages, in the order the file presents them, with
    those pages cited. It is the fallback for a failed model call, so it stays close
    to the text and adds nothing the passages do not say.
    """
    sentences: List[str] = []
    pages: List[int] = []
    for hit in hits:
        if hit.page_num in pages:
            continue
        text = " ".join((hit.chunk.text or "").split())
        if len(text.split()) < 6:
            continue
        sentence = re.split(r"(?<=[.!?\u0964])\s+", text, maxsplit=1)[0]
        line = _trim(sentence, 32)
        if not line:
            continue
        sentences.append(line)
        pages.append(hit.page_num)
        if len(sentences) >= 3:
            break
    if not sentences:
        return "", []
    return " ".join(sentences) + " " + _citation(pages), pages


# A model may cite one page or several in one bracket ("[Page 26, Page 30]").
# Both are provenance, and parsing only the singular form left real pages out of
# `sources` while the marker stayed in the text.
_CITATION_RE = re.compile(r"\[Pages?\b[^\]]*\]", re.I)
_PAGE_NUMBER_RE = re.compile(r"\d+")
_DECORATION_RE = re.compile(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002190-\U000021FF\uFE0F]+")


def _spoken_shape(raw: str) -> str:
    """Strip the writing conventions a voice cannot read out of a model reply.

    The instruction asks for plain sentences, and a model tuned on chat logs still
    reaches for `**bold**`, a leading dash or an icon. That is decoration around the
    answer, not part of it, so removing it keeps the synthesiser from spelling out
    "asterisk" on the caller's speaker.
    """
    text = _DECORATION_RE.sub("", raw)
    text = re.sub(r"^\s*(?:[-*+]|#{1,6})\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:answer|response)\s*:\s*", "", text, flags=re.I | re.M)
    for marker in ("**", "__", "~~", "`"):
        text = text.replace(marker, "")
    return re.sub(r"\s*\n+\s*", " ", text).strip()


def _verified_citation(raw: str, hits) -> Tuple[str, List[int]]:
    """Drop the page markers the retrieval did not return, and re-cite the answer.

    A model that names page 7 for a passage on page 3 has put real text at the
    wrong address, and the phone would show a reply whose sources disagree with
    it. What survives is text whose markers and `sources` are both pages that
    were actually handed to the model.
    """
    pages = {hit.page_num for hit in hits}
    cited = sorted({
        int(number)
        for match in _CITATION_RE.finditer(raw)
        for number in _PAGE_NUMBER_RE.findall(match.group(0))
    } & pages)

    def keep(match: "re.Match") -> str:
        numbers = [int(number) for number in _PAGE_NUMBER_RE.findall(match.group(0))]
        kept = [number for number in numbers if number in pages]
        return " ".join(f"[Page {number}]" for number in kept)

    # One bracket holding three verified pages becomes three markers, and a bracket
    # holding only invented pages disappears. The answer text and `sources` then
    # describe the same evidence, which is what the phone renders.
    text = _CITATION_RE.sub(keep, raw)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "", []
    if not cited:
        cited = [hits[0].page_num]
    if not _CITATION_RE.search(text):
        text = f"{text} [Page {cited[0]}]"
    return text, cited


class ConversationalAgent:
    """Grounded document Q&A with polite out-of-context guardrails."""

    def __init__(self, rag_engine: PDFRAGEngine, llm: Optional[GroundedLlm] = None, threshold: float = 0.34,
                 top_k: int = 4):
        # `llm=None` means answer from the index alone. Reaching for a key is the
        # composition root's job (see server.py), so a test that asks for the
        # deterministic engine really does get it and never pays for a network call.
        self.rag = rag_engine
        self.llm = llm
        self.threshold = threshold
        self.top_k = top_k
        # Rotates the wording of greetings and refusals so three "hello"s in a row
        # are not the same sentence read back three times.
        self._turn = 0
        # Translation is deterministic for a given question and document, and the
        # same turn may search twice (first pass, then a widened retry).
        self._translation_cache: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------------ helpers

    def _expand_follow_up(self, query: str, history: Optional[List[Dict[str, str]]]) -> str:
        """Short follow-ups ('and page 2?', 'why is that?') inherit the previous question.

        A question with no content words of its own inherits everything. A short one
        that points back with a pronoun or an ordinal inherits the subject but keeps
        its own wording, so "and the second one?" searches the previous subject rather
        than the word "second" alone.
        """
        if not history:
            return query
        words = tokenize(query)
        previous = (history[-1] or {}).get("query") or ""
        if not tokenize(previous):
            return query
        if not words:
            # No searchable word is not the same as no question: "Who is Karna?" is a
            # name the Hinglish stop-word list happens to contain, and inheriting the
            # previous subject would answer a question the caller did not ask.
            complete = re.match(
                r"^\s*(?:who|what|which|where|when|why|how|is|are|was|were|does|do|did|"
                r"can|could|should|would|will|has|have|had)\b",
                query,
                re.I,
            )
            if complete and len(re.findall(r"[\w\u0900-\u097F]+", query)) >= 3:
                return query
            return f"{previous} {query}".strip()
        if len(words) <= 5 and _FOLLOW_UP_RE.search(query):
            return f"{previous} {query}".strip()
        return query

    def _translated_query(self, query: str, rag: PDFRAGEngine) -> str:
        """The question in the document's own vocabulary, cached for the turn."""
        key = (query, rag.current_filename or "")
        if key in self._translation_cache:
            return self._translation_cache[key]
        expanded = ""
        suggest = getattr(self.llm, "suggest_search_terms", None)
        if callable(suggest):
            try:
                expanded = (suggest(query, doc_hint=_doc_hint(rag)) or "").strip()
            except Exception:
                expanded = ""
        if len(self._translation_cache) >= 64:
            self._translation_cache.pop(next(iter(self._translation_cache)))
        self._translation_cache[key] = expanded
        return expanded

    def _usable_query(self, text: str, rag: PDFRAGEngine) -> str:
        """Drop translated words this file has no spelling for.

        A model translating a question into the document's language cannot know
        that a legacy font mangled it ("अस्वीकार" is stored as "अवकार"). The
        unknown words contribute nothing to BM25, but they do make the gate read
        the question as mostly unanswerable and reject the passages the one
        recognisable name found. What survives here is what the index can rank.
        """
        terms = list(Counter(tokenize(text)))
        partners = rag.partner_map(terms)
        return " ".join(term for term in terms if rag.df(term) or partners.get(term))

    def _retrieve(self, query: str, rag: PDFRAGEngine, top_k: int, threshold: float
                  ) -> Tuple[List[SearchHit], bool, float]:
        """Retrieve passages, translating the question into the file's language when needed.

        Lexical retrieval only finds words the document uses. A question asked in
        English about a Hindi book, or one paraphrased past recognition, has to be
        asked again in the document's vocabulary before "not in this file" is a
        truthful answer. The translated search never invents evidence: it returns
        documents' own passages, and the gate still has to accept them.
        """
        hits, is_in_context, score = rag.search(query, top_k=top_k, threshold=threshold)
        if self.llm is None:
            return hits, is_in_context, score
        if not (_cross_lingual(query, rag) or not is_in_context or not hits):
            return self._prefer_introduction(query, hits), is_in_context, score

        expanded = self._translated_query(query, rag)
        if not expanded:
            return self._prefer_introduction(query, hits), is_in_context, score
        searchable = self._usable_query(expanded, rag)
        if not searchable:
            return self._prefer_introduction(query, hits), is_in_context, score
        exp_hits, exp_ok, exp_score = rag.search(searchable, top_k=top_k, threshold=threshold)
        if not exp_ok or not exp_hits:
            return self._prefer_introduction(query, hits), is_in_context, score
        if not is_in_context or not hits:
            return self._prefer_introduction(query, exp_hits), True, exp_score
        # Both found something on a cross-lingual question: the translated hits are
        # the more trustworthy, because the original words only matched by accident
        # (a number, a Latin name in a caption).
        seen = {hit.chunk.chunk_id for hit in exp_hits}
        merged = list(exp_hits) + [hit for hit in hits if hit.chunk.chunk_id not in seen]
        return self._prefer_introduction(query, merged[:top_k]), True, score

    @staticmethod
    def _prefer_introduction(query: str, hits: List[SearchHit]) -> List[SearchHit]:
        """Put the page that introduces the subject ahead of its later mentions.

        "Who is Karna" in a 2,000-page epic ranks the page that repeats his name
        most often (a battle scene) above the page that says who he was. Books
        introduce a person the first time they name him, so for a definition or a
        relationship the earliest retrieved page is put in front; the rest keep
        their ranking, and every page is still one retrieval returned.
        """
        if len(hits) < 2 or not (_DEFINE_ASK_RE.search(query) or _ANALYTICAL_ASK_RE.search(query)):
            return hits
        earliest = min(range(len(hits)), key=lambda index: (hits[index].page_num, index))
        if earliest == 0:
            return hits
        return [hits[earliest]] + hits[:earliest] + hits[earliest + 1:]

    def _overview_hits(self, rag: PDFRAGEngine, limit: int = 10) -> List[SearchHit]:
        """A spread of the file for a question about the file as a whole.

        Reading the first two pages of a 2,000-page book gives the publisher's
        address, not the book. The overview keeps the opening, then samples one
        passage from the start of each distinct section, so the model sees the
        document's actual range instead of its imprint.
        """
        content = [chunk for chunk in rag.chunks if not chunk.front_matter and not chunk.index_page]
        if not content:
            content = list(rag.chunks[:limit])
        picks: List[DocumentChunk] = []
        seen: set = set()

        def add(chunk: DocumentChunk) -> None:
            if chunk.chunk_id not in seen and len(picks) < limit:
                seen.add(chunk.chunk_id)
                picks.append(chunk)

        for chunk in content[:2]:
            add(chunk)
        starts: List[DocumentChunk] = []
        previous = None
        for chunk in content:
            heading = (chunk.heading or "").strip().lower()
            if heading and heading != previous:
                previous = heading
                starts.append(chunk)
        if not starts:
            step = max(1, len(content) // limit)
            starts = content[::step]
        if len(starts) > limit:
            step = len(starts) / limit
            starts = [starts[int(index * step)] for index in range(limit)]
        for chunk in starts:
            add(chunk)
        return [
            SearchHit(chunk, score=1.0, coverage=1.0, terms=1, idf_mass=1.0,
                      concepts=1, ideas=1.0, focus=True)
            for chunk in picks
        ]

    def _chapter_hits(self, rag: PDFRAGEngine, ordinal: int, limit: int = 10) -> List[SearchHit]:
        """Evenly spread passages from the chapter the reader named.

        "Summarise the first chapter" cannot be a lexical lookup: the chapter's body
        never repeats the words "first chapter", and the pages of a contents list do.
        The file prints its own closing line for every chapter, so the nth chapter is
        the run of text ending at the nth such line. Sampling across that run lets the
        model summarise the chapter rather than the page that repeats the question.
        """
        markers = [
            index for index, chunk in enumerate(rag.chunks)
            if _CHAPTER_MARK_RE.search(chunk.text or "")
        ]
        if ordinal == -1:
            ordinal = len(markers)
        if ordinal < 1 or ordinal > len(markers):
            return []
        end = markers[ordinal - 1]
        # The closing line can carry its own heading, so walk back to the last chunk
        # that names the body and take that heading's whole run.
        anchor = None
        for back in range(1, 12):
            if end - back < 0:
                break
            if (rag.chunks[end - back].heading or "").strip():
                anchor = end - back
                break
        start = end
        if anchor is not None:
            heading = (rag.chunks[anchor].heading or "").strip().lower()
            start = anchor
            while start > 0 and (rag.chunks[start - 1].heading or "").strip().lower() == heading:
                start -= 1
        if end - start < 3:
            # No usable heading run: the previous chapter's close is the next-best
            # boundary the document itself provides.
            start = markers[ordinal - 2] + 1 if ordinal > 1 else start
        span = rag.chunks[start:end + 1]
        if len(span) > limit:
            step = len(span) / limit
            span = [span[int(index * step)] for index in range(limit)]
        return [
            SearchHit(chunk, score=1.0, coverage=1.0, terms=1, idf_mass=1.0,
                      concepts=1, ideas=1.0, focus=True)
            for chunk in span
        ]

    def _ask_model(self, query: str, passages: List[Tuple[int, str, str]],
                   history: Optional[List[Dict[str, str]]]) -> str:
        """Call the configured model, keeping hand-written test doubles working.

        A subclass that overrides `answer` was written against the old signature,
        so only the real engine is offered the conversation-aware call.
        """
        with_history = getattr(self.llm, "answer_with_history", None)
        if callable(with_history) and type(self.llm).answer is GroundedLlm.answer:
            return with_history(query, passages, history=history,
                                doc_name=_friendly_name(self.rag.current_filename),
                                note=self._vocabulary_note(query))
        return self.llm.answer(query, passages)

    def _vocabulary_note(self, query: str) -> str:
        """Tell the model how this file spells the names the caller used.

        A legacy-font scan stores "Draupadi" as "ौपद" and "Krishna" as "ीकृ".
        The passages are right, but a model that cannot recognise the name in
        them answers that the file never mentions her. The mapping is read from
        the index itself, so it is a fact about this file, not a guess.
        """
        rag = self.rag
        if rag is None or not getattr(rag, "is_devanagari", False):
            return ""
        pairs: List[str] = []
        for term in Counter(tokenize(query)):
            if not re.search(r"[A-Za-z]", term) or rag.df(term):
                continue
            for alt in rag.partner_map([term]).get(term, []):
                if re.search(r"[\u0900-\u097F]", alt) and rag.df(alt):
                    pairs.append(f"'{term}' is written '{alt}'")
                    break
            if len(pairs) >= 6:
                break
        if not pairs:
            return ""
        return ("Encoding note: this scan's font renders some Devanagari conjuncts "
                "truncated, so the names appear shortened: " + "; ".join(pairs) + ".\n")

    def _passages(self, hits, rag: PDFRAGEngine) -> List[Tuple[int, str, str]]:
        """Passages for the model, with the rows of any grid written back as pairs.

        A table survives extraction into text as a flat line - "LLM Benchmark a fast
        hosted model (e.g. Groq-hosted model) Generate answers from..." - and a model
        reading that answers "a fast LLM" when the row it was given says Groq. The
        chunker already kept each row with its column names, so the pairs go back into
        the prompt and the answer can name what the document actually chose.
        """
        out: List[Tuple[int, str, str]] = []
        for hit in hits:
            text = flow_lines(hit.text)
            rows: List[str] = []
            for row in rag.rows:
                if row.page != hit.page_num:
                    continue
                pairs = "; ".join(f"{label}: {value}" for label, value in zip(row.labels, row.cells) if value)
                if pairs and pairs not in rows:
                    rows.append(pairs)
                if len(rows) >= 6:
                    break
            if rows:
                text = f"{text}\nRows of this page:\n" + "\n".join(rows)
            # A heading that stops on a function word was made by the page break, not
            # by the typographer, and telling the model it is a section name misleads it.
            heading = "" if _dangling_tail(hit.heading or "") else hit.heading
            out.append((hit.page_num, heading, text))
        return out

    def _detail_terms(self, passages, rag: PDFRAGEngine) -> set:
        """Words the file uses once or twice - the specifics a reply is judged on.

        A hedged answer such as "the document specifies a fast model" is made only of
        ordinary words plus what the question already said. The detail the caller
        wanted is a name the file writes down once. Comparing the two sets is how the
        turn notices the model summarised instead of answered.
        """
        out = set()
        for _, _, text in passages:
            for token in set(tokenize(text)):
                if len(token) >= 4 and token not in STOP_WORDS and 0 < rag.df(token) <= 2:
                    out.add(token)
        return out

    def _previous_question(self, history: Optional[List[Dict[str, str]]]) -> str:
        """The last real question the caller asked, skipping any earlier complaints."""
        for turn in reversed(history or []):
            previous = (turn or {}).get("query") or ""
            if tokenize(previous) and not _CORRECTION_RE.search(previous):
                return previous
        return ""

    def _conversation_answer(self, query: str) -> Optional[str]:
        """A short human reply to the words that are not about the document at all.

        Greetings, acknowledgements and questions about the assistant itself never
        reach retrieval, because the honest answer to "hello" is not a citation. The
        reply is built from what is actually loaded, so it is grounded in real state
        rather than a canned line, and it rotates so the same greeting twice does not
        sound like a recording of a recording.
        """
        text = (query or "").strip()
        if not text:
            return None
        raw_name = (self.rag.current_filename or "").strip()
        name = _friendly_name(raw_name) if raw_name else ""
        turn = self._turn % 3

        # Check for inquiries about the attached / current document
        if _DOC_INFO_RE.search(text):
            if not raw_name or not self.rag.chunks:
                return "There is no document loaded yet. Please upload a PDF first."
            pages_desc = f"{self.rag.total_pages} page" + ("" if self.rag.total_pages == 1 else "s")
            return (
                f"Currently, '{raw_name}' is attached with {pages_desc} and {len(self.rag.chunks)} indexed passages. What would you like to know about it?",
                f"I have '{raw_name}' loaded ({pages_desc}). Feel free to ask me anything from it.",
                f"The active document is '{raw_name}', consisting of {pages_desc}.",
            )[turn]

        cleaned_text = _GREETING_PREFIX_RE.sub("", text).strip()
        if cleaned_text and _DOC_INFO_RE.search(cleaned_text):
            if not raw_name or not self.rag.chunks:
                return "Hello. There is no document loaded yet. Please upload a PDF first."
            pages_desc = f"{self.rag.total_pages} page" + ("" if self.rag.total_pages == 1 else "s")
            return f"Hello. Currently, '{raw_name}' is attached with {pages_desc}. What would you like to know about it?"

        if len(text.split()) > 8:
            return None
        if _GREETING_RE.search(text):
            if not name:
                return (
                    "Hello. Upload a PDF and I can answer questions from it.",
                    "Hi. Choose a document first, then ask me anything.",
                    "Hello. There is no document loaded yet.",
                )[turn]
            return (
                f"Hello. {name} is loaded, so ask me anything in it.",
                f"Hi. What would you like to know about {name}?",
                f"Hello. I have {name} open.",
            )[turn]
        if _SMALL_TALK_RE.search(text):
            if not name:
                return (
                    "I am a voice assistant that answers questions from a PDF you load.",
                    "I read the PDF you choose and answer from it, out loud.",
                    "I am your PDF reader. Load a file and ask.",
                )[turn]
            return (
                f"I read {name} and answer from it, out loud.",
                f"I am your assistant for {name}. Ask a question and I answer from the file.",
                f"I am the reader for {name}.",
            )[turn]
        if _ACK_RE.search(text):
            return ("Anytime.", "Glad that helped.", "Sure. Ask me anything else.")[turn]
        return None

    def _polite_out_of_context(self, query: str) -> str:
        """Say no in one breath, then point at something the file really does hold.

        The first version ran to three sentences and ended in a question, which is how
        a form letter sounds. A caller is listening rather than reading, so the refusal
        and one way forward is the whole of it. Every answer stays inside the uploaded
        file, so this is the reply when the file itself cannot answer.
        """
        name = _friendly_name(self.rag.current_filename)
        turn = self._turn % 3
        lead = f"I could not locate that in {name}."
        tail = (
            "It is not something this file answers.",
            "Ask about a section it does cover, or have me read out the headings.",
            "Nothing in here covers it, and I have no answer of my own for it either.",
        )[turn]
        return f"{lead} {tail}"

    def _outside_answer(self, query: str, timings: Dict[str, float]) -> Dict[str, Any]:
        """Say the file cannot answer, and never fill the gap from general knowledge.

        An answer that is not in the uploaded document is the wrong answer however
        true it is elsewhere: a question bank must not come back with a film plot,
        and a chapter of the Mahabharata must not come back with the distance to the
        moon. The turn refuses in one breath, keeps the out-of-context flag the
        phone labels, and cites nothing.
        """
        return self._pack(self._polite_out_of_context(query), [], 0.0,
                          "not in document", timings)

    def _structure_answer(self, query: str, hits, rag: PDFRAGEngine) -> Optional[Tuple[str, List[int]]]:
        """Answer a question about the shape of the file from the index itself.

        "What sections does this have" and "list the steps" are not needle searches:
        the answer is the structure the chunker already recorded, so reading it out is
        both cheaper and more complete than ranking sentences from four passages.
        """
        if _about_document(query, rag):
            if self.llm is not None:
                # When an LLM is configured, generate a synthesized summary of the document
                # from key overview passages rather than listing section titles.
                return None
            named = (set(tokenize(query)) - _SPEECH_ACT) & rag.heading_terms()
            if named and not _SECTIONS_ASK_RE.search(query):
                # The question named one of the file's own titles, so it asks about that
                # part of the document and the layout answer would describe the whole of
                # it. Retrieval has the section ranked already.
                return None
            name = _friendly_name(rag.current_filename)
            name = name[:1].upper() + name[1:]
            sections = rag.sections()
            names = [title.lstrip("# ").strip() for _, title in sections[:3]]
            pages = sorted({page for page, _ in sections[:3]})
            size = f"{rag.total_pages} page" + ("" if rag.total_pages == 1 else "s")
            # The document's own opening line is the cheapest true summary there is:
            # a plan says what it is building in its first sentence, and a reader who
            # asks "what is this about" wants that, not a list of section titles.
            opening = ""
            for line in (rag.chunks[0].text.split("\n") if rag.chunks else []):
                candidate = line.strip()
                if len(candidate.split()) >= 5 and not _looks_like_code(candidate):
                    opening = _trim(candidate, 26)
                    break
            lead = f" In short, {opening.rstrip('.')}." if opening else ""
            if not names:
                return f"{name} is {size}.{lead} {_citation(pages)}", pages
            listed = ", ".join(names[:-1]) + (f" and {names[-1]}" if len(names) > 1 else names[0])
            rest = f", and {len(sections) - len(names)} more" if len(sections) > len(names) else ""
            return (f"{name} is {size}.{lead} It is organised into {listed}{rest}. {_citation(pages)}", pages)

        if _SECTIONS_ASK_RE.search(query):
            sections = rag.sections()
            if not sections:
                return None
            shown = sections[:12]
            names = [title.lstrip("# ").strip() for _, title in shown]
            pages = sorted({page for page, _ in shown})
            listed = ", ".join(names[:-1]) + (f", and {names[-1]}" if len(names) > 1 else names[0])
            extra = len(sections) - len(shown)
            tail = f", plus {extra} more" if extra > 0 else ""
            return (f"It is organised into {listed}{tail} {_citation(pages)}.", pages)

        if _ENUM_RE.search(query) and _STEPS_ASK_RE.search(query) and hits:
            items: List[str] = []
            pages: List[int] = []
            for unit in _units(hits):
                if unit.kind != "item" or unit.value in items:
                    continue
                items.append(unit.value)
                if unit.page not in pages:
                    pages.append(unit.page)
                if len(items) >= 8:
                    break
            if len(items) < 2:
                return None
            listed = ", ".join(items[:-1]) + f", and {items[-1]}"
            return (f"It lists {len(items)} of them: {listed} {_citation(pages)}.", pages)
        return None

    def _extractive_answer(self, query: str, hits, rag: PDFRAGEngine,
                           prose_only: bool = False) -> Tuple[str, List[int]]:
        """Compose the shortest grounded answer, and the pages it is actually from."""
        table = _table_answer(query, rag)
        if table:
            return table
        needle = _needle_answer(query, hits, rag)
        if needle:
            return needle
        fact = _unit_answer(query, hits, rag)
        if fact:
            return fact

        # The nth thing a section says is read from the section, in the file's order:
        # ranking sentences by their overlap with "first step" finds the line that
        # repeats the caller's words, and a manual never writes "step 1" where it means
        # it - it just puts the first instruction first.
        ordinal = _ordinal_ask(query)
        if ordinal is not None:
            named = (set(tokenize(query)) - _SPEECH_ACT) & rag.heading_terms()
            ranked = sorted(
                ((len(named & set(tokenize(hit.heading or ""))), rank, hit) for rank, hit in enumerate(hits)),
                key=lambda item: (-item[0], item[1]),
            )
            if ranked and ranked[0][0] >= min(2, len(named)):
                hit = ranked[0][2]
                units = [line.strip() for line in flow_lines(hit.text).split("\n") if line.strip()]
                if units and abs(ordinal) <= len(units):
                    index = len(units) - 1 if ordinal == -1 else ordinal - 1
                    body = units[index]
                    # One printed line is rarely a whole instruction, so the run takes the
                    # lines under it until it says something a listener can hold on to.
                    while len(body.split()) < 18 and index + 1 < len(units):
                        index += 1
                        body = f"{body} {units[index]}"
                    body = _trim(body, 45)
                    if not body.endswith((".", "!", "?")):
                        body = f"{body}."
                    return f"{body} {_citation([hit.page_num])}", [hit.page_num]

        # A request to summarise a named part of the file is answered by that part as
        # the file wrote it. Picking the one sentence whose words overlap the question
        # keeps the line that repeats the noun and drops everything it was a line about,
        # which is the opposite of the two sentences a caller asked for.
        if _RECAP_ASK_RE.search(query):
            named = (set(tokenize(query)) - _SPEECH_ACT) & rag.heading_terms()
            for hit in hits:
                if named & set(tokenize(hit.heading or "")):
                    section = flow_lines(hit.text)
                    if len(section.split()) >= 5:
                        body = _trim(section, 70)
                        body = body if body.endswith((".", "!", "?")) else f"{body}."
                        return f"{body} {_citation([hit.page_num])}", [hit.page_num]

        ask = _Ask(query, rag)
        candidates: List[Tuple[int, str, bool]] = []
        for hit in hits:
            candidates.extend(_candidate_lines(hit, ask))
        if not candidates:
            return self._polite_out_of_context(query), []

        prose = [line for _, line, is_prose in candidates if is_prose]
        if prose_only and not prose:
            # Nothing here can be said out loud as a sentence about the subject: what
            # is left is code and grid cells, and reading those to a caller who asked
            # a plain question is worse than saying the file does not answer it.
            return self._polite_out_of_context(query), []
        pool = prose or [line for _, line, _ in candidates]
        # A question that can only be answered yes or no wants the one line that settles
        # it, not three passages that each repeat part of it.
        limit = 1 if _POLAR_ASK_RE.search(query) else 3
        chosen = rag.rank_sentences(query, pool, limit=limit)
        # The sentences have to say something the question asked for. Retrieval already
        # decided the passages are about this subject, but a passage is a neighbourhood,
        # and quoting its best-ranked line when that line carries none of the question's
        # own words is how an answer about nothing gets a page number.
        if not chosen or ask.weight(" ".join(chosen)) <= 0.0:
            return self._polite_out_of_context(query), []

        page_of: Dict[str, int] = {}
        for page, line, _ in candidates:
            page_of.setdefault(line, page)
        cited = [page_of.get(line, hits[0].page_num) for line in chosen]

        def sentence(text: str) -> str:
            text = _trim(text)
            return text if text.endswith((".", "!", "?", ":", ")")) else f"{text}."

        # A code answer reads badly out loud unless it is introduced as code.
        lead = "The manual does this with the code " if all(_looks_like_code(line) for line in chosen) else ""
        body = lead + " ".join(sentence(line) for line in chosen)
        # An API sheet puts the thing it describes in the title and the description
        # underneath, so the sentence that answers "which one removes it" is the one
        # that never names it. The title is quoted here, not paraphrased: it is the
        # document's own heading, from the same passage as the sentence above it.
        subject = next((hit.heading for hit in hits if hit.heading and chosen[0] in hit.text), "")
        if (subject and not _dangling_tail(subject) and _looks_like_code(subject)
                and subject.strip() not in body):
            body = f"{subject}: {body}"
        return body + " " + _citation(cited), cited

    # -------------------------------------------------------------------- main

    @staticmethod
    def _pack(answer: str, cited: List[int], score: float, engine: str, timings: Dict[str, float],
              refusal: Optional[bool] = None) -> Dict[str, Any]:
        """One response shape, where an answer that cites nothing is a refusal.

        `retrieval_score` is evidence about the question - how much of it the index
        recognised - so a refusal still reports it. `confidence_score` is about the
        answer, and there is no confidence to claim in an answer that was not found.
        `refusal=False` is only for a reply that was never meant to cite anything:
        a greeting is grounded in the conversation, not in the file.
        """
        refused = not cited if refusal is None else refusal
        return {
            "answer": answer,
            "is_out_of_context": refused,
            "sources": [] if refused else cited,
            "retrieval_score": score,
            "confidence_score": 0.0 if refused else score,
            "engine": engine,
            "timings": timings,
        }

    def process_query(self, query: str, history: Optional[List[Dict[str, str]]] = None,
                      _retry: bool = False) -> Dict[str, Any]:
        started = time.perf_counter()
        self._turn += 1
        # One index reference for the whole turn: an upload that finishes while this
        # question is being answered must not mix two documents into one reply.
        rag = self.rag
        if not rag.chunks:
            return {
                "answer": "Upload a PDF first, and I can answer from it.",
                "is_out_of_context": False,
                "no_document": True,
                "sources": [],
                "retrieval_score": 0.0,
                "confidence_score": 0.0,
                "engine": "none (no document indexed)",
                "timings": {"retrieval_ms": 0.0, "answer_ms": 0.0, "total_ms": _ms(started)},
            }

        search_query = self._expand_follow_up(query, history)
        # Words that are about the conversation rather than the file never reach the
        # index: the honest reply to "hello" cites nothing, and routing it through
        # retrieval is what made it come back as a refusal.
        chat = self._conversation_answer(search_query)
        if chat:
            return self._pack(chat, [], 0.0, "conversation",
                              {"retrieval_ms": 0.0, "answer_ms": 0.0, "total_ms": _ms(started)}, refusal=False)

        # Being told the last answer was wrong is not a new question, and quoting some
        # other line at it is what made the app sound broken. What the caller wants is
        # the previous question answered again with more of the file in front of it.
        if not _retry and _CORRECTION_RE.search(query):
            previous = self._previous_question(history)
            if previous:
                result = self.process_query(previous, history, _retry=True)
                if not result["is_out_of_context"]:
                    body = result["answer"]
                    result["answer"] = f"Let me look again. {body[0].lower()}{body[1:]}"
                    return result
                return self._outside_answer(previous, result["timings"])

        cleaned_search = _GREETING_PREFIX_RE.sub("", search_query).strip()
        if cleaned_search and tokenize(cleaned_search):
            search_query = cleaned_search

        about = _about_document(search_query, rag)
        # A named structural part is resolved from the file's own order, not by
        # ranking sentences against the words "first chapter".
        part = requested_section(search_query) if _PART_ASK_RE.search(search_query) else None
        numbered = any(heading_section(chunk.heading) for chunk in rag.chunks) if part is not None else False
        top_k = self.top_k * 2 if _retry else self.top_k
        # A file cut into short passages needs more of them in front of the model:
        # three thin mentions of a name read to a careful model as "this file never
        # explains him", which is the difference between an answer and a refusal.
        if rag.words_avg < 20:
            top_k = max(top_k, 8)
        if (_DEFINE_ASK_RE.search(search_query) or _ANALYTICAL_ASK_RE.search(search_query)
                or _RECAP_ASK_RE.search(search_query) or about):
            # "Who is X" is answered by the one line in the book that describes X, and a
            # 686-page scan buries it under forty mentions of him. A lookup needs the top
            # passage; a definition or a comparison needs the top few, and retrieval is
            # sub-millisecond, so the only cost is a slightly longer prompt.
            top_k = max(top_k, 8)
        retrieved = time.perf_counter()
        if about and rag.chunks and self.llm is not None:
            hits = self._overview_hits(rag)
            is_in_context = True
            score = 1.0
        elif part is not None and not numbered:
            chapter_hits = self._chapter_hits(rag, part)
            if chapter_hits:
                hits, is_in_context, score = chapter_hits, True, 1.0
            else:
                hits, is_in_context, score = self._retrieve(search_query, rag, top_k, self.threshold)
        else:
            hits, is_in_context, score = self._retrieve(search_query, rag, top_k, self.threshold)
        retrieval_ms = _ms(retrieved)
        synthesised = time.perf_counter()

        def timed() -> Dict[str, float]:
            return {"retrieval_ms": retrieval_ms, "answer_ms": _ms(synthesised), "total_ms": _ms(started)}

        # A question about the shape of the file is answered from the index, before the
        # content gate: the gate asks whether the *subject* of a question is in the
        # document, and "what sections does this have" has no subject outside it.
        structure = self._structure_answer(search_query, hits, rag)
        if structure:
            answer, cited = structure
            return self._pack(answer, cited, score, "index structure", timed())

        if not is_in_context or not hits:
            return self._outside_answer(search_query, timed())

        if self.llm is not None:
            # A fact is rarely one 40-word strip: the sentence before names the
            # subject and the one after states the consequence, so the model gets the
            # hits plus their neighbours.
            model_hits = rag.context_window(hits, window=1, limit=max(12, top_k + 4))
            try:
                raw = _spoken_shape(self._ask_model(search_query, self._passages(model_hits, rag), history))
            except Exception as error:
                # Any model failure at all, including one nobody anticipated, costs
                # the polished wording and nothing else: the same passages still
                # produce a grounded, citable answer.
                return self._from_index(search_query, hits, rag, score, timed(),
                                        f"extractive (model call failed: {_brief(error)})")
            if NOT_FOUND_TOKEN in raw.upper() or _META_RE.search(raw):
                # Before accepting the decline, hand the model more of the
                # neighbourhood: a comparison or a rule often needs the paragraph
                # on either side of every hit, and one declined call is cheap
                # evidence that the context was cut too tight.
                wider = rag.context_window(hits, window=2, limit=max(16, top_k * 2 + 6))
                if len(wider) > len(model_hits):
                    try:
                        retried = _spoken_shape(
                            self._ask_model(search_query, self._passages(wider, rag), history)
                        )
                    except Exception:
                        retried = ""
                    if retried and NOT_FOUND_TOKEN not in retried.upper() and not _META_RE.search(retried):
                        answer, cited = _verified_citation(retried, wider)
                        if answer:
                            return self._pack(answer, cited, score, self.llm.name, timed())
                # The model's own not-found marker anywhere in the reply, however
                # worded around it, is a decline - as is explaining that the material
                # handed to it does not contain the answer. Its length is not evidence,
                # and a sentence about "the provided passages" is not something a
                # caller should hear as an answer.
                #
                # A question of causation or comparison cannot be answered by the
                # sentence that merely repeats its subject. The model read the wider
                # neighbourhood and found no explanation there; a lexical line that
                # shares the name is a guess, so the turn says the file does not
                # answer instead of quoting one.
                if _ANALYTICAL_ASK_RE.search(search_query):
                    return self._outside_answer(search_query, timed())
                return self._from_index(search_query, hits, rag, score, timed(),
                                        f"extractive ({self.llm.name} declined)", prose_only=True)
            answer, cited = _verified_citation(raw, model_hits)
            if not answer:
                return self._from_index(search_query, hits, rag, score, timed(),
                                        f"extractive ({self.llm.name} cited nothing usable)")
            return self._pack(answer, cited, score, self.llm.name, timed())

        answer, cited = self._extractive_answer(search_query, hits, rag)
        return self._pack(answer, cited, score, "extractive bm25 (no model configured)", timed())

    def _mention_answer(self, query: str, hits, rag: PDFRAGEngine) -> Tuple[str, List[int]]:
        """The file's own line about a subject it never stops to define.

        A narrative mentions its people rather than describing them, so "who is
        Krishna" is answered honestly by the sentence the book itself says about him,
        with its page - and not by a claim that a 686-page Mahabharata does not cover
        Krishna, which is what a strict answer-shape test produces on every book.
        """
        spoken = list(Counter(query_terms(query)))
        known = [term for term in spoken if rag.df(term) > 0]
        if not known:
            return "", []
        focus = min(known, key=rag.df)
        pool: List[str] = []
        page_of: Dict[str, int] = {}
        for hit in hits:
            for page, line, is_prose in _candidate_lines(hit):
                if is_prose and focus in tokenize(line) and line not in page_of:
                    page_of[line] = page
                    pool.append(line)
        if not pool:
            return "", []
        chosen = rag.rank_sentences(query, pool, limit=1)
        if not chosen:
            return "", []
        line = _trim(chosen[0])
        if not line.endswith((".", "!", "?")):
            line = f"{line}."
        page = page_of[chosen[0]]
        return f"{line} {_citation([page])}", [page]

    def _from_index(self, query: str, hits, rag: PDFRAGEngine, score: float,
                    timings: Dict[str, float], engine: str,
                    prose_only: bool = False) -> Dict[str, Any]:
        """Answer from the passages the model could not use, or refuse honestly.

        A decline is the model's judgement about wording, not about the index, so the
        same hits go through the extractive synthesiser. It is held to a higher bar in
        this branch: the model has already said the passages do not answer, so quoting
        a line of HTML from them would be a guess dressed as a citation. When the prose
        that carries the question's own words fails too, a summary question gets a
        digest of the passages and everything else is refused - the answers stay inside
        the uploaded file either way.
        """
        answer, cited = self._extractive_answer(query, hits, rag, prose_only=prose_only)
        if not cited:
            # A summary has no needle to find: the caller wants the gist of a part or
            # a whole, which lives across the passages rather than in one line. When
            # the model could not polish it, read a sentence from each of a few pages
            # instead of pretending the file has nothing to say. A Devanagari scan is
            # left out: this library's legacy font mangles conjuncts, so quoting it
            # verbatim would be read aloud as nonsense rather than as a summary.
            if (_RECAP_ASK_RE.search(query) or requested_section(query) is not None) \
                    and not getattr(rag, "is_devanagari", False):
                digest, pages = _digest_answer(hits)
                if digest:
                    return self._pack(digest, pages, score,
                                      f"{engine} (summary of the file's passages)", timings)
            # The passages hold the subject but not an answer-shaped sentence for it.
            # Saying the file's own words about that subject is honest; saying a book
            # that names him 203 times does not cover him is not.
            mention, pages = self._mention_answer(query, hits, rag)
            if mention and score >= 0.5:
                return self._pack(mention, pages, score, f"{engine} (the file's own words)", timings)
            return self._outside_answer(query, timings)
        return self._pack(answer, cited, score, engine, timings)


def _friendly_name(raw: str) -> str:
    """A document name a listener can follow - underscores and .pdf are not speech."""
    stem = re.sub(r"[_\-]+", " ", (raw or "").rsplit(".", 1)[0])
    return re.sub(r"\s+", " ", stem).strip() or "this document"


def _brief(error: BaseException) -> str:
    """One displayable line naming a model failure, for the `engine` field."""
    return f"{type(error).__name__}: {error}"[:140]


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)
