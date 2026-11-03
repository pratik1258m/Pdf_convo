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

from rag_engine import PDFRAGEngine, SECTION_TITLE_RE, STOP_WORDS, looks_like_heading, query_terms, split_sentences, tokenize


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


SYSTEM_PROMPT = (
    "You answer questions strictly from the document passages given below. "
    "Write at most three short sentences that read well out loud. "
    "Append the page reference of the passage you used, like [Page 4]. "
    "Never state anything that is not present in the passages. "
    "If the passages do not contain the answer, reply with exactly: NOT_IN_DOCUMENT"
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

    def answer(self, query: str, passages: List[Tuple[int, str, str]], timeout: float = 25.0) -> str:
        context = "\n\n".join(
            f"[Page {page}] {f'({heading}) ' if heading else ''}{text}" for page, heading, text in passages
        )
        prompt = f"Passages from {os.environ.get('CONVO_DOC_NAME', 'the document')}:\n\n{context}\n\nQuestion: {query}"

        if self.style == "gemini":
            payload = {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 220},
            }
        else:
            payload = {
                "model": self.model,
                "temperature": 0.2,
                "max_completion_tokens": 220,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }

        try:
            response = httpx.post(self.url, headers=self.headers, json=payload, timeout=timeout)
        except httpx.HTTPError as error:
            raise LlmAnswerError(f"network: {error}") from error

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
_CODE_RE = re.compile(r"(?:^|\s)(?:from|import|def|class|return|assert|print|self\.|driver\.)\b|[{}<>=]|\.\w+\(")

# Question shape decides which needles are even eligible. "How does the manual
# handle an alert with an explicit wait?" contains *wait* but wants a mechanism,
# not the number 10.
_HOW_RE = re.compile(r"\bhow (?:does|do|did|to|can|could|should|is|are|will)|\bwhy\b|\bexplain\b|\bwhat happens|\bwalk(?: me)? through\b|\bsteps?\b", re.I)
_SCALAR_ASK_RE = re.compile(r"\b(how long|how many|number of|for how long|wait time|timeout value|max wait|what value|which value|how much)\b", re.I)


def _mentions(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


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
    return bool(_CODE_RE.search(line))


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
    out: List[_Line] = []
    for rank, hit in enumerate(hits):
        position = 0
        for raw in hit.text.split("\n"):
            line = raw.strip()
            # A single-token line is still a real fact: `driver.get("https://...")`
            # has no spaces but carries the whole answer.
            if len(line) < 4:
                continue
            position += 1
            line_terms = set(tokenize(line))
            overlap = sum(rag.idf(term) for term in terms if term in line_terms)
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
_DOUBLE_ASK_RE = re.compile(r"\band\b.*\b(what|which|how|who|where|does|should|many|much)\b", re.I)
_SECTIONS_ASK_RE = re.compile(r"\b(sections|headings|chapters|contents|table of contents|structure|titles)\b", re.I)
_STEPS_ASK_RE = re.compile(r"\b(steps?|items?|points?|rules?|checklist|requirements|stages|phases)\b", re.I)


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
        return None
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
    if ask.coverage(row.text, residual) < 0.7:
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
        for raw in hit.text.split("\n"):
            line = raw.strip()
            if len(line) < 4:
                continue
            position += 1
            if " | " in line:
                # A grid row is not a fact of its own: columns are compared, and the
                # whole row is read out. That is `_table_answer`'s job, done against the
                # row index so it can see rows this passage does not contain.
                continue
            if _looks_like_code(line):
                for kind, value in _credential_pairs(line):
                    out.append(_Unit(hit.page_num, rank, position, kind, value, "", "kv"))
                out.append(_Unit(hit.page_num, rank, position, "", line, "", "code"))
                continue
            match = _KV_LINE_RE.match(line)
            if match and match.group(1).strip() and not match.group(1).endswith(("http", "https")):
                out.append(_Unit(hit.page_num, rank, position, match.group(1).strip(),
                                 match.group(2).strip(), "", "kv"))
                continue
            stripped = _LIST_ITEM_RE.sub("", line).strip()
            if stripped and stripped != line:
                out.append(_Unit(hit.page_num, rank, position, "", stripped, "", "item"))
                continue
            for sentence in (split_sentences(line) or [line]):
                sentence = sentence.strip()
                if len(sentence) > 3:
                    out.append(_Unit(hit.page_num, rank, position, "", sentence, "", "prose"))
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
        label = ask.weight(unit.label)
        value = ask.weight(unit.value)
        if not unit.label or label <= 0:
            continue  # an unlabelled line is what the sentence path is for
        if len(unit.value.split()) > 30:
            continue  # a paragraph, not a fact
        score = label + 0.5 * value - 0.7 * unit.rank
        if looks_like_heading(unit.value):
            score -= 2.0  # a title is never the answer to a question about a value
        if score > 0:
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

    if _mentions(q, "username", "user name", "password", "credential", "log in with", "signed in as"):
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

    if _mentions(q, "file", "excel", "workbook", "csv", "filename", "sheet", "output file"):
        chosen = _pick([line for line in lines if _FILE_RE.search(line.text)])
        if chosen:
            names: List[str] = []
            for name in _FILE_RE.findall(chosen.text):
                if name not in names:
                    names.append(name)
            if names:
                listed = " and ".join(names[:2]) if len(names) > 1 else names[0]
                return f"The file is {listed} {_citation([chosen.page])}.", [chosen.page]

    if not asks_how and _mentions(q, "option", "value", "title", "heading", "label", "assertion", "course", "message", "text", "error"):
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


def _candidate_lines(hit) -> List[Tuple[int, str, bool]]:
    """Line-level quotable units, with a real section title offered as prose."""
    out: List[Tuple[int, str, bool]] = []
    if hit.heading and SECTION_TITLE_RE.match(hit.heading):
        out.append((hit.page_num, hit.heading.rstrip(".:"), True))
    for raw in hit.text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        for part in (split_sentences(line) or [line]):
            part = part.strip()
            if len(part.split()) < 3:
                continue
            out.append((hit.page_num, part, not _looks_like_code(part)))
    return out


def _trim(line: str, words: int = 24) -> str:
    parts = line.split()
    if len(parts) <= words:
        return line
    return " ".join(parts[:words]).rstrip(",;:") + "."


_CITATION_RE = re.compile(r"\[Page\s+(\d+)\]", re.I)


def _verified_citation(raw: str, hits) -> Tuple[str, List[int]]:
    """Drop the page markers the retrieval did not return, and re-cite the answer.

    A model that names page 7 for a passage on page 3 has put real text at the
    wrong address, and the phone would show a reply whose sources disagree with
    it. What survives is text whose markers and `sources` are both pages that
    were actually handed to the model.
    """
    pages = {hit.page_num for hit in hits}
    cited = sorted({int(match.group(1)) for match in _CITATION_RE.finditer(raw)} & pages)
    text = _CITATION_RE.sub(lambda match: match.group(0) if int(match.group(1)) in pages else "", raw)
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

    # ------------------------------------------------------------------ helpers

    def _expand_follow_up(self, query: str, history: Optional[List[Dict[str, str]]]) -> str:
        """Short follow-ups ('and page 2?') inherit the previous question's terms."""
        if tokenize(query) or not history:
            return query
        previous = (history[-1] or {}).get("query") or ""
        return f"{previous} {query}".strip() if tokenize(previous) else query

    def _passages(self, hits) -> List[Tuple[int, str, str]]:
        return [(hit.page_num, hit.heading, hit.text) for hit in hits]

    def _polite_out_of_context(self, query: str) -> str:
        doc_name = self.rag.current_filename or "this document"
        q_lower = query.lower()
        if "who are you" in q_lower or "what can you do" in q_lower:
            hint = "I am your voice assistant for reading and answering from this PDF"
        elif "capital of" in q_lower or "weather" in q_lower or "time" in q_lower:
            hint = "that is general knowledge rather than something this file covers"
        else:
            hint = "that topic is not covered by the indexed text"
        return (
            f"I could not locate this specific information inside {doc_name}. "
            f"While {hint}, I can answer anything the document does cover. "
            f"Want me to summarise a section or read out the headings?"
        )

    def _structure_answer(self, query: str, hits, rag: PDFRAGEngine) -> Optional[Tuple[str, List[int]]]:
        """Answer a question about the shape of the file from the index itself.

        "What sections does this have" and "list the steps" are not needle searches:
        the answer is the structure the chunker already recorded, so reading it out is
        both cheaper and more complete than ranking sentences from four passages.
        """
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

    def _extractive_answer(self, query: str, hits, rag: PDFRAGEngine) -> Tuple[str, List[int]]:
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

        candidates: List[Tuple[int, str, bool]] = []
        for hit in hits:
            candidates.extend(_candidate_lines(hit))
        if not candidates:
            return self._polite_out_of_context(query), []

        prose = [line for _, line, is_prose in candidates if is_prose]
        pool = prose or [line for _, line, _ in candidates]
        chosen = rag.rank_sentences(query, pool, limit=3)
        if not chosen:
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
        return lead + " ".join(sentence(line) for line in chosen) + " " + _citation(cited), cited

    # -------------------------------------------------------------------- main

    @staticmethod
    def _pack(answer: str, cited: List[int], score: float, engine: str, timings: Dict[str, float]) -> Dict[str, Any]:
        """One response shape, where an answer that cites nothing is a refusal.

        `retrieval_score` is evidence about the question - how much of it the index
        recognised - so a refusal still reports it. `confidence_score` is about the
        answer, and there is no confidence to claim in an answer that was not found.
        """
        refused = not cited
        return {
            "answer": answer,
            "is_out_of_context": refused,
            "sources": [] if refused else cited,
            "retrieval_score": score,
            "confidence_score": 0.0 if refused else score,
            "engine": engine,
            "timings": timings,
        }

    def process_query(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        started = time.perf_counter()
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
        retrieved = time.perf_counter()
        hits, is_in_context, score = rag.search(search_query, top_k=self.top_k, threshold=self.threshold)
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
            return self._pack(self._polite_out_of_context(query), [], score, "guardrail", timed())

        if self.llm is not None:
            try:
                raw = self.llm.answer(search_query, self._passages(hits))
            except Exception as error:
                # Any model failure at all, including one nobody anticipated, costs
                # the polished wording and nothing else: the same passages still
                # produce a grounded, citable answer.
                answer, cited = self._extractive_answer(search_query, hits, rag)
                return self._pack(answer, cited, score, f"extractive (model call failed: {_brief(error)})", timed())
            if NOT_FOUND_TOKEN in raw.upper() and len(raw) < 90:
                return self._pack(self._polite_out_of_context(query), [], score,
                                  f"{self.llm.name} declined", timed())
            answer, cited = _verified_citation(raw, hits)
            if not answer:
                answer, cited = self._extractive_answer(search_query, hits, rag)
                engine = f"extractive ({self.llm.name} cited nothing usable)"
            else:
                engine = self.llm.name
            return self._pack(answer, cited, score, engine, timed())

        answer, cited = self._extractive_answer(search_query, hits, rag)
        return self._pack(answer, cited, score, "extractive bm25 (no model configured)", timed())


def _brief(error: BaseException) -> str:
    """One displayable line naming a model failure, for the `engine` field."""
    return f"{type(error).__name__}: {error}"[:140]


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)
