"""Retrieval over PDF text.

Ranking is Okapi BM25 (no external dependencies, no model download); the
0-1 value reported to callers is idf-weighted query-term *coverage* of the best
passage, which is a measurement of what was actually found, not a model score.
"""

import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pypdf

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "make", "made", "also", "therefore", "however", "much", "many", "anyone", "someone",
    "get", "got", "use", "used", "using", "want", "need", "tell", "say", "said", "explain",
}

# Latin words, Devanagari words, and numerals. The Devanagari range has to be
# spelled out: `\w` covers the letters but not the matras and the nukta, which are
# nonspacing marks, so `\w` alone would cut "यूज़रनेम" into five unusable fragments.
_TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F'’\-]*|\d+(?:[.,]\d+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[]?[A-Z0-9])")
_HEADING_MAX_CHARS = 90

# A lab manual is titled by its sections: "Module 1 - Practical 3: Manual Test
# Case Design". Without this pattern the strongest line in the document is
# treated as ordinary prose and never labels the passages under it.
SECTION_TITLE_RE = re.compile(
    r"^\s*(?:(?:module|unit|chapter|part|lab|section|experiment)\s+\d+[\w\s.\-]*?)?"
    r"(?:practical|experiment|activity|assignment|problem|question)\s*\d+\s*[:.\-]",
    re.I,
)
_CODE_NOISE_RE = re.compile(r"[\"'(){}\[\]<>={}|]|->|=>|==|\.\w")
# "1. Open", "3. Capture failure on error" are numbered steps inside a table, not
# section titles; only a dotted number ("2.3 Approach") heads a block on its own.
_DOTTED_NUMBER_RE = re.compile(r"^\d+\.\d+(\.\d+)*\s+\S")
# A statement of code is never a title, however neatly it is capitalised.
_CODE_LINE_RE = re.compile(
    r"^\s*(?:from|import|def|class|return|assert|print|if|for|while|with|try|except|await)\b"
    r"|\b(?:driver|self|page|login|wait)\s*\.\w+\s*\("
    r"|[\w\]\)]\s*=\s*[^=]",
    re.I,
)

# Light suffix folding: enough to match "automation/automate", "tests/test",
# "required/require" without a stemming library. Longest suffix first.
_SUFFIXES = sorted(
    ("ization", "isations", "isation", "ements", "ement", "ingly", "edly", "ation", "ations",
     "ness", "ment", "ments", "ing", "ings", "ed", "es", "s"),
    key=len,
    reverse=True,
)

# A Hindi question inflects the same noun three different ways in one sentence
# ("अंकों", "अंक", "अंक से"), so folding has to run on Devanagari too or the
# rarest word in the question never reaches the passage that answers it.
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_DEVANAGARI_SUFFIXES = sorted(
    ("ों", "ें", "िय", "ता", "ती", "ते", "ो", "े", "ा", "ी", "ि", "ु", "ू", "ं", "ः"),
    key=len,
    reverse=True,
)

_HINDI_STOP_WORDS = {
    "और", "के", "की", "का", "किया", "है", "हैं", "था", "थी", "थे", "क्या", "में", "से", "को", "पर",
    "एक", "यह", "वह", "कि", "भी", "या", "कर", "करने", "करता", "करती", "लिए", "अपना", "अपनी",
    "होता", "होती", "होते", "होना", "दिया", "गया", "जाता", "जाती", "अन्य", "कुछ", "इस", "इसके",
    "जिस", "जो", "तथा", "उस", "उसके", "कोई", "नहीं", "बहुत", "पहले", "बाद", "तब", "इसलिए",
    "माना", "माने", "कहा", "कहते", "रहा", "रही", "गई", "गया", "लेना", "देना", "वाला", "वाली",
}
STOP_WORDS |= {word for word in _HINDI_STOP_WORDS}

# Words that say what the caller wants *done*, or *where* to look, rather than what
# the document is about. They are excluded from the in-context decision: a spoken
# "summarise the reimbursement page" asks about reimbursement, and the verb
# "summarise" is not evidence that the question is outside this file. This is a
# closed class of English speech-act verbs and structure nouns - not a topic list.
_SPEECH_ACT = {
    "summarise", "summarize", "summary", "list", "enumerate", "outline", "overview",
    "describe", "detail", "define", "mention", "recap", "brief", "elaborate", "recount",
    "narrate", "highlight", "walkthrough",
    "section", "heading", "chapter", "page", "topic", "contents", "segment",
}


def stem(word: str) -> str:
    if _DEVANAGARI_RE.search(word):
        for suffix in _DEVANAGARI_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word
    if len(word) <= 4:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> List[str]:
    """Stemmed tokens, with identifier parts indexed separately.

    A lab manual is full of `user-name`, `students.xlsx`, `pytest-order`,
    `the-internet.herokuapp.com`. Splitting on identifier punctuation is what
    lets a question about "the-internet" reach the passage that names it.

    A lone digit is kept, unlike a lone letter: "practical 3", "grade 5" and "5 years"
    are the facts a spoken question then asks for, and the ordinal and number-word
    expansions on the query side have nothing to match against while digits vanish.
    """
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        seen: set = set()
        for piece in [raw] + re.split(r"[-_./\\]", raw):
            token = piece.strip("'’-\u0964\u0965")
            if token in STOP_WORDS or (len(token) < 2 and not token.isdigit()):
                continue
            token = stem(token)
            if not token or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return tokens


# "first Selenium IDE practical" has to reach the passage headed
# "Module 1 - Practical 1"; without this the ordinal is an unmatchable word.
_ORDINALS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}

# A price list says "5 years" and the customer asks "how long is the five year
# warranty". Numbers spoken as words and written as figures are the same fact, and
# a ranker that cannot see that refuses a document it has clearly answered.
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
}

# The pairs a reader actually paraphrases across, curated rather than learned: a
# lexical ranker has no other way to connect "which file holds the output" to the
# line that says "the report is written to results.xlsx". Expansion is only allowed
# for a group whose alternative occurs in this document, and it is used for ranking
# only - the coverage score a caller sees still comes from the words the user spoke.
_PARAPHRASE_GROUPS: List[set] = []
for _group in (
    ("output", "written", "saved", "stored", "report", "result", "generated"),
    ("input", "read", "loaded", "source", "taken from"),
    ("file", "document", "workbook", "sheet", "fixture"),
    ("cost", "price", "charge", "amount", "quote", "rs", "rupees", "fees"),
    ("wait", "delay", "sleep", "timeout", "pause"),
    ("first", "initial", "opening", "beginning"),
    ("last", "final", "closing", "ending"),
    ("use", "employ", "apply", "invoked"),
    ("install", "setup", "requirement", "prerequisite", "dependencies"),
    ("fail", "error", "defect", "reject", "refuse"),
    ("check", "verify", "assert", "validate", "confirm"),
    ("count", "number", "quantity", "how many"),
    ("who", "person", "role", "instructor", "officer", "manager", "responsible"),
    ("website", "url", "site", "domain", "link", "host"),
    ("allow", "permit", "must", "prohibited", "required"),
    ("summary", "summarise", "overview", "outline", "list", "sections"),
    ("introduce", "present", "propose", "release", "contribute"),
    ("dataset", "benchmark", "corpus", "collection"),
    ("discontinue", "retire", "deprecated", "unavailable", "withdrawn", "end of life"),
    ("ban", "banned", "bans", "prohibit", "prohibited", "forbid", "forbidden", "bar", "disallow"),
    ("duration", "period", "length", "span"),
    ("page", "section", "heading", "clause"),
    ("employ", "utilise", "leverage", "used"),
):
    _expanded = set()
    for _word in _group:
        _pieces = tokenize(_word)
        if _pieces:
            _expanded.add(_pieces[-1])
    if len(_expanded) >= 2:
        _PARAPHRASE_GROUPS.append(_expanded)
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def query_terms(text: str) -> List[str]:
    """Query-side tokenisation: stems plus word-form equivalents for ordinals,
    spelled-out numbers, and the Hindi numerals a voice transcript can deliver."""
    terms = tokenize(text.translate(_DEVANAGARI_DIGITS))
    lowered = text.lower()
    for word, digit in _ORDINALS.items():
        if re.search(rf"\b{word}\b", lowered) and digit not in terms:
            terms.append(digit)
    for word, digit in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered) and digit not in terms:
            terms.append(digit)
    for token in list(terms):
        if token.isdigit() and len(token) <= 3:
            for word, digit in _NUMBER_WORDS.items():
                if digit == token and word not in terms:
                    terms.append(word)
    return terms


# "Practical 4" is more specific than "Module 2", and it is the practical a
# question is asking about, so the specific number always wins.
_SPECIFIC_SECTION_RE = re.compile(r"\b(?:practical|experiment|activity|assignment|problem|question)\s*(\d{1,2})\b", re.I)
_GENERAL_SECTION_RE = re.compile(r"\b(?:module|unit|chapter|section|lab|step)\s*(\d{1,2})\b", re.I)
_SECTION_NOUNS = r"practical|experiment|module|unit|chapter|assignment|activity|lab|question|problem|step"


def heading_section(heading: str) -> Optional[int]:
    text = heading or ""
    match = _SPECIFIC_SECTION_RE.search(text) or _GENERAL_SECTION_RE.search(text)
    return int(match.group(1)) if match else None


def requested_section(query: str) -> Optional[int]:
    """The passage number a question asks for, from `practical 4` or `the first practical`.

    Explicit numbers win over ordinals, so "the first step of practical 3" is
    section 3 rather than section 1.
    """
    lowered = query.lower()
    explicit = _SPECIFIC_SECTION_RE.search(lowered) or _GENERAL_SECTION_RE.search(lowered)
    if explicit:
        return int(explicit.group(1))
    for word, digit in _ORDINALS.items():
        if re.search(rf"\b{word}\b[\w\s\-]{{0,40}}?\b(?:{_SECTION_NOUNS})\b", lowered, re.I):
            return int(digit)
    return None


def split_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = _SENTENCE_RE.split(cleaned)
    sentences = []
    for part in parts:
        part = part.strip()
        if len(part.split()) >= 3 or (sentences and len(part) < 4):
            sentences.append(part)
        elif sentences:
            sentences[-1] = f"{sentences[-1]} {part}".strip()
        else:
            sentences.append(part)
    return sentences


def looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > _HEADING_MAX_CHARS:
        return False
    if SECTION_TITLE_RE.match(stripped):
        return True
    # "Verified -> Closed Status transition strictly" reads as title case but is
    # a step in a state diagram; section titles carry no code punctuation.
    if _CODE_NOISE_RE.search(stripped) or _CODE_LINE_RE.search(stripped):
        return False
    if stripped.endswith((".", ";", ",", "!", "?", ":", "-")):
        return False
    words = stripped.split()
    if len(words) < 2 or len(words) > 10:
        return False
    caps = sum(1 for word in words if word[:1].isupper())
    # "1. Open" is a numbered step. Only a dotted section number stands alone.
    dotted = bool(_DOTTED_NUMBER_RE.match(stripped))
    title_case = caps == 2 if len(words) == 2 else caps >= max(2, int(len(words) * 0.6))
    upper = stripped.isupper() and len(words) <= 6
    return dotted or upper or title_case


class DocumentChunk:
    def __init__(self, text: str, page_num: int, chunk_id: int, heading: str = ""):
        self.text = text.strip()
        self.page_num = page_num
        self.chunk_id = chunk_id
        self.heading = heading
        self.token_counts: Counter = Counter()
        self.body_counts: Counter = Counter()
        self.heading_counts: Counter = Counter()
        self.length = 0
        self.body_length = 0
        self.heading_length = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"chunk_id": self.chunk_id, "page_num": self.page_num, "text": self.text, "heading": self.heading}


class SearchHit:
    """One ranked passage, with the measurements that put it in the top set."""

    def __init__(self, chunk: DocumentChunk, score: float, coverage: float, terms: int,
                 idf_mass: float = 0.0, concepts: int = 0, focus: bool = False):
        self.chunk = chunk
        self.score = score
        self.coverage = coverage
        self.terms = terms
        self.idf_mass = idf_mass
        # How many of the question's *ideas* the passage carries, counting a word the
        # document spells differently as the same idea.
        self.concepts = concepts
        # Whether the passage carries the question's focus word - the first content
        # word, which in "which shoes are required" is `shoes`, not `required`.
        self.focus = focus

    @property
    def page_num(self) -> int:
        return self.chunk.page_num

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def heading(self) -> str:
        return self.chunk.heading


class TableRow:
    """One row of a table, with the document's own column names attached.

    The agent answers grid questions from these rather than from a text line: a
    row keeps its identity even when the row and the column it is asked about
    land in different passages, or on opposite sides of a page break.
    """

    __slots__ = ("page", "section", "labels", "cells")

    def __init__(self, page: int, section: str, labels: List[str], cells: List[str]):
        self.page = page
        self.section = section
        self.labels = labels
        self.cells = cells

    def value(self, label: str) -> str:
        try:
            return self.cells[self.labels.index(label)]
        except ValueError:
            return ""

    @property
    def text(self) -> str:
        return " | ".join(f"{label}: {cell}" for label, cell in zip(self.labels, self.cells))


class PDFRAGEngine:
    """Zero-dependency BM25 index with page attribution and coverage gating."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, target_words: int = 60, min_chunk_words: int = 8,
                 heading_boost: float = 1.4, section_match_bonus: float = 2.2,
                 section_conflict_penalty: float = 0.35):
        self.k1 = k1
        self.b = b
        self.target_words = target_words
        self.min_chunk_words = min_chunk_words
        self.heading_boost = heading_boost
        self.section_match_bonus = section_match_bonus
        self.section_conflict_penalty = section_conflict_penalty
        self.current_filename: str = ""
        self.chunks: List[DocumentChunk] = []
        self.raw_text: str = ""
        self.doc_summary: str = ""
        self.total_pages: int = 0
        self.profile: Dict[str, Any] = {}
        self._df: Dict[str, int] = {}
        self._avg_len: float = 1.0
        self._avg_body_len: float = 1.0
        self._avg_heading_len: float = 1.0
        self._doc_count: int = 0
        self._headings_seen = 0
        self._merged_runs = 0
        self._titles: List[str] = []
        self._table_headers: Dict[int, List[str]] = {}
        self._table_rows_labeled = 0
        self.rows: List[TableRow] = []

    def load_pdf(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found at {file_path}")

        reader = pypdf.PdfReader(file_path)
        self.current_filename = os.path.basename(file_path)
        self.total_pages = len(reader.pages)
        self.chunks = []
        self._headings_seen = 0
        self._merged_runs = 0
        self._titles: List[str] = []
        self._table_headers: Dict[int, List[str]] = {}
        self._table_rows_labeled = 0
        self.rows = []
        page_texts: List[str] = []

        raw_pages: List[List[Tuple[str, float, float]]] = []

        for page in reader.pages:
            rows = self._page_rows(page)
            if rows is None:
                # No usable typography: the plain text path, unchanged.
                text = self._clean_page(page.extract_text() or "")
                if not text:
                    continue
                page_texts.append(text)
                rows = [(unit, 0.0, 0.0) for unit in self._line_units(text)]
            else:
                page_texts.append("\n".join(text for text, _, _ in rows))
            if rows:
                raw_pages.append(rows)

        body_size = self._body_size(raw_pages)
        # The right edge has to come from the whole document before any line is
        # judged against it: one page's narrow column is another page's paragraph.
        right_edge = self._right_edge(raw_pages, body_size)
        section = ""
        for page_num, rows in enumerate(raw_pages, start=1):
            rows = self._label_rows(self._glue_wraps(rows, right_edge), page_num)
            # A section that begins at the foot of a page still labels its
            # continuation, so heading state is carried between pages.
            section = self._chunk_rows(rows, page_num, section, body_size)

        self.raw_text = "\n".join(page_texts)
        self._build_index()
        self.doc_summary = self._generate_summary()

        return {
            "filename": self.current_filename,
            "total_pages": self.total_pages,
            "total_chunks": len(self.chunks),
            "summary": self.doc_summary,
            "chunking": dict(self.profile),
        }

    # ------------------------------------------------------------------ extraction

    def _page_rows(self, page) -> Optional[List[Tuple[str, float]]]:
        """One entry per visual line, with its font size, when the PDF reveals both.

        A PDF that lays a table out publishes the geometry: the cells of a row sit
        on one baseline at increasing x, and a section title is set larger than the
        body under it. pypdf's plain text throws both away, which is why a table
        arrived as one value per line and a sentence-case title looked like prose.
        A document with no typography to reveal - a Google Docs export reports
        every span at size 1.0 - returns None and keeps the old path.
        """
        spans: List[Tuple[str, float, float, float]] = []

        def visit(text: str, _cm, tm, _font, size) -> None:
            if text and text.strip():
                spans.append((text, float(tm[4] if tm else 0.0), float(tm[5] if tm else 0.0),
                              float(size or 0.0)))

        try:
            page.extract_text(visitor_text=visit)
        except Exception:
            return None
        if len(spans) < 3 or len({round(s[3], 2) for s in spans if s[3] > 1.01}) < 2:
            return None

        # A row is a run of cells on one baseline whose x keeps increasing: equal y
        # with x reset to the margin means separate flowables that happen to share a
        # local coordinate, which is how a list of one-line items comes out.
        rows: List[List[Tuple[float, str, float]]] = []
        row_y: Optional[float] = None
        row_last_x = 0.0
        for text, x, y, size in spans:
            for offset, piece in enumerate(text.split("\n")):
                piece = piece.strip()
                if not piece:
                    continue
                # Only the first piece of a span carries the span's own position.
                cell_x = x if offset == 0 else row_last_x + 1.0
                cell = (cell_x, piece, size)
                if rows and row_y is not None and abs(row_y - y) <= 1.5 and cell_x > row_last_x + 0.5:
                    rows[-1].append(cell)
                else:
                    rows.append([cell])
                    row_y = y
                row_last_x = cell_x

        out: List[Tuple[str, float, float]] = []
        for row in rows:
            xs = [cell[0] for cell in row]
            cells = sorted(row, key=lambda cell: cell[0])
            size = max(cell[2] for cell in cells)
            # pypdf's visitor reports where a span starts, never where it ends, so the
            # end of a line is estimated from its character count. The estimate only
            # has to be good enough to tell a line that ran to the margin from one that
            # stopped short, which is the difference between a wrapped sentence and a
            # finished one - and every line of a paragraph is set at the same size.
            x1 = max(cell_x + len(piece) * max(size, 1.0) * 0.5 for cell_x, piece, _ in cells)
            if len(cells) >= 3 and len(set(round(value, 1) for value in xs)) >= 3 and max(xs) - min(xs) >= 12:
                # A row of distinct columns: keep the cells in one line, separated,
                # so a parameter row or a price row stays a single quotable fact.
                out.append((" | ".join(cell[1] for cell in cells), size, x1))
            else:
                out.append((" ".join(cell[1] for cell in cells), size, x1))
        return out or None

    @staticmethod
    def _right_edge(pages_rows: List[List[Tuple[str, float, float]]], body_size: float) -> float:
        """The x a full line of body text reaches, or 0.0 when the PDF has no geometry.

        The ninth-tenths line, not the widest: one overflowing heading would otherwise
        move the edge rightwards and make every real paragraph look short.
        """
        ends = sorted(
            x1
            for rows in pages_rows
            for text, size, x1 in rows
            if body_size > 1.01 and abs(size - body_size) < 0.6 and " | " not in text
        )
        if len(ends) < 4:
            return 0.0
        return ends[max(0, int(len(ends) * 0.9) - 1)]

    @staticmethod
    def _body_size(pages_rows: List[List[Tuple[str, float, float]]]) -> float:
        """The size most lines are set in; anything larger labels a block."""
        counts: Counter = Counter()
        for rows in pages_rows:
            for _, size, _ in rows:
                if size > 1.01:
                    counts[round(size, 1)] += 1
        return counts.most_common(1)[0][0] if counts else 0.0

    @staticmethod
    def _is_size_heading(text: str, size: float, body_size: float) -> bool:
        if not body_size or size <= 1.01 or size <= body_size + 0.01:
            return False
        stripped = text.strip()
        if len(stripped) > _HEADING_MAX_CHARS or not stripped:
            return False
        return not stripped.endswith((".", "!", "?", ";", ","))

    @staticmethod
    def _clean_page(raw: str) -> str:
        raw = raw.replace("\x07", "\t")
        lines = [line.rstrip() for line in raw.splitlines()]
        kept = [line for line in lines if not re.fullmatch(r"\s*page\s*\d+\s*", line, re.I)]
        text = "\n".join(kept)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_rows(self, rows: List[Tuple[str, float, float]], page_num: int, inherited_heading: str,
                    body_size: float) -> str:
        """Split one page's lines into searchable passages and return the open heading.

        Nothing is discarded: a run shorter than `min_chunk_words` is merged
        into the passage before it rather than dropped, which is what keeps a
        one-line fact (`pip install ...`, a single assertion) findable. A line is
        a heading either because its font is larger than the body's - the signal a
        PDF actually carries - or because the old capitalisation heuristics read it
        as a title, which is all a document with no typography can offer.
        """
        heading = inherited_heading
        attached = False
        buffer: List[str] = []
        buffer_words = 0

        def flush() -> None:
            nonlocal buffer, buffer_words, attached
            if not buffer:
                return
            # Newlines survive chunking so the answer builder can pick a single
            # line (for example one code statement) instead of a whole block.
            text = "\n".join(buffer).strip()
            previous = self.chunks[-1] if self.chunks else None
            if (
                len(text.split()) < self.min_chunk_words
                and previous is not None
                and previous.page_num == page_num
                and previous.heading == heading
                and len(previous.text.split()) + len(text.split()) <= self.target_words + self.min_chunk_words
            ):
                previous.text = f"{previous.text}\n{text}".strip()
                self._merged_runs += 1
            else:
                self.chunks.append(
                    DocumentChunk(text=text, page_num=page_num, chunk_id=len(self.chunks), heading=heading)
                )
                attached = bool(heading)
            buffer, buffer_words = [], 0

        for text, size, _ in rows:
            if self._is_size_heading(text, size, body_size) or (
                looks_like_heading(text) and not re.search(r"[.!?]$", text)
            ):
                flush()
                label = text.rstrip(" :")
                # A heading replaced before any passage carried it is a title above a
                # title, not a fact. It is kept in `titles` for the summary rather than
                # merged into the heading below it: swallowing the document title puts
                # words like "manual" and "laboratory" into every section's heading
                # field, where they out-rank the passage the question was really about.
                if heading and not attached and heading not in self._titles:
                    self._titles.append(heading)
                heading = label
                attached = False
                self._headings_seen += 1
                continue
            words = len(text.split())
            if buffer and buffer_words + words > self.target_words:
                flush()
            buffer.append(text)
            buffer_words += words
        flush()
        return heading

    _CONTINUED_START = re.compile(r"^[.)\]}<>=+&|,;:\\/?]")
    _OPEN_END = re.compile(r"[(\[{<,:+&|\\/=?]$")
    _SENTENCE_END = re.compile(r"[.!?\u0964\u0965]\s*$")
    _CODEISH_RE = re.compile(r"[{}<>=\[\]]|\w+\.\w+\(|^\s*(?:from|import|def|class|return|assert|print)\b")

    def _line_units(self, page_text: str) -> List[str]:
        """One unit per source line, with lone lines split into sentences."""
        units: List[str] = []
        for block in re.split(r"\n{2,}", page_text):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            if len(lines) == 1:
                units.extend(split_sentences(lines[0]) or lines)
                continue
            units.extend(lines)
        return units

    _CAPTION_RE = re.compile(r"^(?:table|figure|tab|exhibit|chart)\s*\d+\b", re.I)
    # A code line is full of pipe characters inside string literals, and a row of a
    # real grid is not. Braces, quotes and a `'` before a word are the shapes only
    # source code has, so a line carrying them is never split into cells.
    _CODE_ROW_RE = re.compile(r"[{}\"]|'[A-Za-z]|->|::")

    def _header_cells(self, cells: List[str]) -> Optional[List[str]]:
        """The column names of a table, or None when this row is data.

        A legend row is words only and short; a data row carries a value, and a value
        starts with a digit. `Precision at 4` is a column name even though it contains
        a figure, which is why the test looks at the start of the cell rather than
        anywhere inside it. A caption is not a column: `Table 1 | Ranker | F1` names
        three columns only after the caption is dropped, and the data rows under it
        have three cells.
        """
        if len(cells) >= 2 and self._CAPTION_RE.match(cells[0]):
            cells = cells[1:]
        if len(cells) < 2 or any(cell[:1].isdigit() for cell in cells):
            return None
        if any(len(cell.split()) > 3 or len(cell) > 24 for cell in cells):
            return None
        return cells if len(set(cells)) == len(cells) else None

    def _label_rows(self, rows: List[Tuple[str, float, float]], page_num: int) -> List[Tuple[str, float, float]]:
        """Attach each table column's header to the cell under it, on every row.

        A price grid says `Luminous NXG+ 5520 | NXG5520 | 14990 | 2 years`, and the
        word *warranty* exists only in the header line above it. Asked "which
        inverter carries a five year warranty?", a lexical ranker cannot reach that
        row at all - the row simply does not contain the question's words - and a
        model handed the flattened grid has to guess which column was which. Repeating
        the header inside the row as `Warranty: 2 years` makes the row self-describing,
        which is what the published table-QA work measures as the biggest single gain.

        The header is remembered per column count and survives a page break, because a
        long grid continues below without repeating its header row.

        Each labelled row is also kept as structured data in `self.rows`, because a
        question about a column ("which item has the lowest warranty") can only be
        answered over every row of that table, and retrieval hands back four passages,
        not a table.
        """
        out: List[Tuple[str, float, float]] = []
        section = ""
        for text, size, x1 in rows:
            cells = [cell.strip() for cell in text.split(" | ")] if " | " in text else []
            if cells and self._CODE_ROW_RE.search(text):
                cells = []
            if not cells and looks_like_heading(text) and not re.search(r"[.!?]$", text):
                section = text.rstrip(" :")
                out.append((text, size, x1))
                continue
            if len(cells) >= 2:
                header = self._header_cells(cells)
                if header is not None:
                    self._table_headers[len(header)] = header
                    out.append((text, size, x1))
                    continue
                stored = self._table_headers.get(len(cells))
                if stored:
                    pairs = [f"{label}: {cell}" for label, cell in zip(stored, cells) if cell]
                    if len(pairs) >= 2:
                        self._table_rows_labeled += 1
                        kept = [(label, cell) for label, cell in zip(stored, cells) if cell]
                        self.rows.append(TableRow(page_num, section, [label for label, _ in kept],
                                                  [cell for _, cell in kept]))
                        out.append((" | ".join(pairs), size, x1))
                        continue
            out.append((text, size, x1))
        return out

    _LIST_START = re.compile(r"^(?:\d+[.)\-]|[a-z][.)]|\-|\*|\u2022|\u25cf|>)\s")

    def _glue_wraps(self, rows: List[Tuple[str, float, float]], right_edge: float) -> List[Tuple[str, float, float]]:
        """Fold wrapped lines back into the sentence or statement they belong to.

        Two kinds of wrap are repaired. A soft line break inside a sentence is the
        one that costs accuracy: "the ranker reaches an F1" ending a visual line
        and "of 0.82 against 0.74" starting the next means every quotable unit is
        cut before its own number, so an answer reads as a half sentence with the
        fact missing. And a continuation of code (`)` on the next line, or a line
        left open by `(`) is folded back onto the statement.

        Whether a line was broken by the page is read from the geometry: a line whose
        estimated end reaches the column's right edge did not finish there. That
        replaces the old demand that the next line start lowercase, which silently
        dropped "…in the best passage exceeds / 0.34." - the number was the fact the
        question asked for and it sat one line too low. A document that reveals no
        geometry keeps the lowercase rule, so plain-text extraction is unchanged.

        Code is deliberately excluded from the prose rule: two consecutive Python
        statements both start lowercase and neither ends in a full stop, and gluing
        them would destroy the one-line-per-statement structure that makes code
        quotable. Same font size is required, which keeps a table row from being
        welded onto the paragraph under it.
        """
        out: List[Tuple[str, float, float]] = []
        for text, size, x1 in rows:
            previous = out[-1] if out else None
            if previous:
                previous_text, previous_size, previous_x1 = previous
                continuation = bool(self._CONTINUED_START.match(text)) or bool(
                    self._OPEN_END.search(previous_text)
                )
                same_flow = not (previous_size and size) or abs(previous_size - size) < 0.2
                # `x1` is an estimate, not a printed number: glyph widths are assumed,
                # and a dense font comes out around two characters short of the column.
                # The tolerance therefore scales with the size of the text instead of
                # using a fixed distance, which is what lets "…accumulated beyond" reach
                # the line below it instead of ending the answer early.
                wrapped = bool(right_edge) and previous_x1 >= right_edge - max(8.0, 2.5 * size)
                prose_wrap = bool(
                    same_flow
                    and not self._SENTENCE_END.search(previous_text)
                    and not previous_text[-1:] == ":"
                    and not self._CODEISH_RE.search(previous_text)
                    and not self._CODEISH_RE.search(text)
                    and not self._CONTINUED_START.match(previous_text)
                    and not looks_like_heading(previous_text)
                    and not looks_like_heading(text)
                    and not self._LIST_START.match(text)
                    and " | " not in previous_text and " | " not in text
                    and (wrapped or (not right_edge and text[:1].islower()))
                )
                if (continuation or prose_wrap) and len(previous_text) + len(text) < (400 if prose_wrap else 320):
                    out[-1] = (f"{previous_text} {text}".strip(), max(previous_size, size),
                               max(previous_x1, x1))
                    continue
            out.append((text, size, x1))
        return out

    # -------------------------------------------------------------------- index

    def _build_index(self) -> None:
        self._df = {}
        lengths: List[int] = []
        body_lengths: List[int] = []
        heading_lengths: List[int] = []
        for chunk in self.chunks:
            body_tokens = tokenize(chunk.text)
            heading_tokens = tokenize(chunk.heading) if chunk.heading else []
            chunk.body_counts = Counter(body_tokens)
            chunk.heading_counts = Counter(heading_tokens)
            chunk.body_length = len(body_tokens)
            chunk.heading_length = len(heading_tokens)
            chunk.token_counts = chunk.body_counts + chunk.heading_counts
            chunk.length = chunk.body_length + chunk.heading_length
            lengths.append(chunk.length)
            body_lengths.append(max(1, chunk.body_length))
            heading_lengths.append(max(1, chunk.heading_length))
            for term in set(chunk.token_counts):
                self._df[term] = self._df.get(term, 0) + 1

        self._doc_count = len(self.chunks)
        self._avg_len = (sum(lengths) / len(lengths)) if lengths else 1.0
        self._avg_body_len = (sum(body_lengths) / len(body_lengths)) if body_lengths else 1.0
        self._avg_heading_len = (sum(heading_lengths) / len(heading_lengths)) if heading_lengths else 1.0
        self.profile = self._chunk_profile(lengths)

    def _chunk_profile(self, lengths: List[int]) -> Dict[str, Any]:
        """Measurement of the segmentation itself, reported to the client."""
        if not lengths:
            return {"chunks": 0}
        ordered = sorted(lengths)
        indexed_words = sum(
            len((f"{chunk.heading} {chunk.text}" if chunk.heading else chunk.text).split()) for chunk in self.chunks
        )
        document_words = len(self.raw_text.split()) or 1
        pages = {chunk.page_num for chunk in self.chunks}
        return {
            "chunks": len(ordered),
            "words_min": ordered[0],
            "words_p50": ordered[len(ordered) // 2],
            "words_max": ordered[-1],
            "words_avg": round(self._avg_len, 1),
            "target_words": self.target_words,
            "headings_tracked": self._headings_seen,
            "titles_retained": len(self._titles),
            "table_rows_labeled": self._table_rows_labeled,
            "short_runs_merged": self._merged_runs,
            "pages_with_chunks": len(pages),
            "unique_terms": len(self._df),
            "text_preserved_pct": round(100.0 * min(indexed_words, document_words) / document_words, 1),
        }

    def df(self, term: str) -> int:
        """How many passages use this word. Zero means the document never says it."""
        return self._df.get(term, 0)

    def idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if self._doc_count == 0:
            return 0.0
        return math.log(1 + (self._doc_count - df + 0.5) / (df + 0.5))

    def _bm25(self, counts: Counter, length: int, query_terms: List[str], avg: Optional[float] = None) -> float:
        score = 0.0
        reference = self._avg_len if avg is None else avg
        norm = self.k1 * (1 - self.b + self.b * (length / reference if reference else 1.0))
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            score += self.idf(term) * (tf * (self.k1 + 1)) / (tf + norm)
        return score

    def coverage(self, counts: Counter, query_terms: List[str]) -> float:
        """idf-weighted share of the question's informative terms found in a passage.

        Terms that do not occur anywhere in the document are excluded from the
        denominator; the 'no such topic here' signal is carried by matched_terms
        instead, so a long question is not penalised for its own vocabulary.
        """
        present = [term for term in query_terms if term in self._df]
        total = sum(self.idf(term) for term in present)
        if total <= 0:
            return 0.0
        found = sum(self.idf(term) for term in present if counts.get(term, 0))
        return found / total

    def matched_term_count(self, counts: Counter, query_terms: List[str]) -> int:
        return sum(1 for term in query_terms if term in self._df and counts.get(term, 0))

    def matched_idf_mass(self, counts: Counter, query_terms: List[str]) -> float:
        """How much information the matched terms carry, not how many there were.

        "What is earthing?" has one content word, and that word appears in three
        passages of a forty-page manual: it is a complete question, and a rule that
        insists on two matched words refuses it. Rarity is the thing the count was
        trying to measure, so measure it directly.
        """
        return sum(self.idf(term) for term in query_terms if counts.get(term, 0))

    def partner_map(self, terms: List[str]) -> Dict[str, List[str]]:
        """For each spoken term, the document's own alternative spellings of it.

        Used for ranking and for the in-context decision, never for the reported
        coverage: a caller reads `retrieval_score` as "how much of what you asked did
        I find", and quietly counting words the user never said would inflate it.
        """
        out: Dict[str, List[str]] = {}
        seen = set(terms)
        for term in terms:
            for group in _PARAPHRASE_GROUPS:
                if term not in group:
                    continue
                for alternative in group:
                    if alternative == term or alternative not in self._df or alternative in seen:
                        continue
                    seen.add(alternative)
                    out.setdefault(term, []).append(alternative)
        return out

    def _paraphrase_partners(self, terms: List[str]) -> List[str]:
        """Document-side synonyms of the words the user spoke, for ranking only."""
        return [partner for partners in self.partner_map(terms).values() for partner in partners]

    @staticmethod
    def _concept_count(counts: Counter, terms: List[str], partners: Dict[str, List[str]]) -> int:
        """How many of the question's ideas the passage carries, under any spelling."""
        return sum(1 for term in terms
                    if counts.get(term, 0) or any(counts.get(partner, 0) for partner in partners.get(term, ())))

    def search(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.25,
        min_terms: int = 2,
        per_page: int = 2,
    ) -> Tuple[List[SearchHit], bool, float]:
        """Rank passages, then cap how many come from one page.

        Diversifying by page matters because the answer cites pages: three
        passages from page 2 hide the fact that page 1 also matched.
        """
        if not self.chunks:
            return [], False, 0.0

        spoken = list(Counter(query_terms(query)))
        if not spoken:
            return [], False, 0.0
        # The words the user actually said, before ordinals and spelled-out
        # numbers are added: the gate below asks whether *they* are document
        # vocabulary, and an injected "five" would fail its own test.
        content = [term for term in Counter(tokenize(query)) if term not in _SPEECH_ACT]
        informative = [term for term in spoken if term not in _SPEECH_ACT]
        partners = self.partner_map(spoken)
        terms = spoken + self._paraphrase_partners(spoken)
        focus = content[0] if content else ""
        focus_rare = self._df.get(focus, 10 ** 6) <= max(2, 0.4 * self._doc_count)
        # A question word the document has never used, and has no synonym for, is the
        # signal that the question is about something outside this file.
        unknown = [term for term in content if term not in self._df and term not in partners]

        scored: List[SearchHit] = []
        target_section = requested_section(query)
        numbered_sections = any(heading_section(chunk.heading) for chunk in self.chunks)
        for chunk in self.chunks:
            body = self._bm25(chunk.body_counts, chunk.body_length, terms, self._avg_body_len)
            heading = self._bm25(chunk.heading_counts, chunk.heading_length, terms, self._avg_heading_len)
            raw = body + self.heading_boost * heading
            if target_section is not None and numbered_sections:
                # A question about "the first practical" is answered by section 1,
                # not by whatever page repeats its words most often.
                placed = heading_section(chunk.heading)
                if placed == target_section:
                    raw *= self.section_match_bonus
                elif placed is not None:
                    raw *= self.section_conflict_penalty
            if raw <= 0:
                continue
            scored.append(
                SearchHit(
                    chunk,
                    score=round(raw, 4),
                    coverage=self.coverage(chunk.token_counts, spoken),
                    terms=self.matched_term_count(chunk.token_counts, spoken),
                    idf_mass=self.matched_idf_mass(chunk.token_counts, spoken),
                    concepts=self._concept_count(chunk.token_counts, informative, partners),
                    focus=bool(chunk.token_counts.get(focus, 0)),
                )
            )

        if not scored:
            return [], False, 0.0

        scored.sort(key=lambda hit: hit.score, reverse=True)
        kept: List[SearchHit] = []
        per_page_count: Dict[int, int] = {}
        for hit in scored:
            used = per_page_count.get(hit.page_num, 0)
            if used >= per_page:
                continue
            per_page_count[hit.page_num] = used + 1
            kept.append(hit)
            if len(kept) >= top_k:
                break

        best = max(kept, key=lambda hit: hit.coverage)

        def accepted(hit: SearchHit) -> bool:
            if hit.coverage < threshold:
                return False
            if hit.concepts >= min_terms:
                return True
            # The two-idea floor exists to stop one generic word from carrying a whole
            # question, so it is released only when the question's focus word is the
            # document's own rare vocabulary and every other word of the question is an
            # idea the document has. "What is earthing" and "which shoes are required"
            # pass; "explain how a black hole forms" fails on the second half - *black*
            # and *hole* are ideas this file never mentions, and matching the word
            # *form* somewhere in it does not answer that question.
            return bool(focus) and not unknown and hit.focus and focus_rare

        matched = [hit for hit in kept if accepted(hit)]
        return matched, bool(matched), round(best.coverage, 4)

    def sections(self) -> List[Tuple[int, str]]:
        """The document's own headings, in reading order, with the page each one opens.

        A question about the shape of the file ("what sections does this have?") is
        answerable from the index itself, and the headings are the only structure a
        PDF reliably carries.
        """
        out: List[Tuple[int, str]] = []
        seen: set = set()
        for chunk in self.chunks:
            title = (chunk.heading or "").strip()
            if len(title) < 3:
                continue
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((chunk.page_num, title))
        return out

    def rank_sentences(self, query: str, sentences: List[str], limit: int = 3, fallback: bool = True) -> List[str]:
        """Pick the sentences that carry the question's rare terms, in document order."""
        if not sentences:
            return []
        terms = list(Counter(query_terms(query)))
        if not terms:
            return sentences[:limit]

        scored = []
        for position, sentence in enumerate(sentences):
            counts = Counter(tokenize(sentence))
            length = sum(counts.values()) or 1
            raw = self._bm25(counts, length, terms)
            scored.append((raw, position, sentence, counts))

        scored.sort(key=lambda item: item[0], reverse=True)
        chosen: List[Tuple[int, str]] = []
        seen_terms: set = set()
        for raw, position, sentence, counts in scored:
            if raw <= 0:
                break
            new_terms = set(counts) & set(terms) - seen_terms
            if not new_terms and len(chosen) >= 1:
                continue
            seen_terms |= new_terms
            chosen.append((position, sentence))
            if len(chosen) >= limit:
                break

        if not chosen:
            chosen = [(scored[0][1], scored[0][2])] if scored else []
        chosen.sort(key=lambda item: item[0])
        return [sentence for _, sentence in chosen]

    def _generate_summary(self) -> str:
        if not self.chunks:
            return "No text available in document."
        headings = [chunk.heading for chunk in self.chunks[:40] if chunk.heading]
        # A title that was only ever a label above another label still names the
        # document, so it belongs in the summary even though no passage carries it.
        for title in self._titles:
            if title not in headings:
                headings.insert(0, title)
        if headings:
            unique: List[str] = []
            for heading in headings:
                if heading not in unique:
                    unique.append(heading)
            return "Sections detected: " + "; ".join(unique[:8]) + "."
        first = " ".join(chunk.text for chunk in self.chunks[:2])
        return (first[:300] + "...") if len(first) > 300 else first

    def status(self) -> Dict[str, Any]:
        return {
            "filename": self.current_filename,
            "total_pages": self.total_pages,
            "total_chunks": len(self.chunks),
            "summary": self.doc_summary,
            "chunking": dict(self.profile),
        }
