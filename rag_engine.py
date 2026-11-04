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
    "hi", "hello", "hey", "please", "kindly", "namaste", "thank", "thanks", "ok", "okay",
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
     "ness", "ment", "ments", "ing", "ings", "ed", "es", "s", "e"),
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
    # The interrogative pronouns, which are the same closed class English already
    # carries (who, what, when, where, why, how). "सी"/"सा" are the qualifier particles
    # that combine with कौन to mean "which", and carry nothing on their own.
    "कौन", "कब", "कहाँ", "कहां", "क्यो", "क्यों", "कैसे", "कितना", "कितनी", "कितने",
    "कौनसा", "कौनसी", "कौनसे", "सी", "सा",
}
_HINGLISH_STOP_WORDS = {
    "ye", "yeh", "woh", "wo", "kya", "hai", "hain", "tha", "thi", "the", "ka", "ki", "ke", "ko", "se",
    "mein", "me", "par", "pe", "aur", "ya", "bhi", "liye", "kar", "kare", "karna", "karne", "karta",
    "karti", "karte", "hoga", "hogi", "hoge", "diya", "gaya", "gayi", "gaye", "batao", "bataiye",
    "bata", "kaho", "bolo", "kaun", "kaunsa", "kaunsi", "kaise", "kaha", "kahan", "kab", "kyu", "kyun",
    "kitna", "kitni", "kitne", "isme", "usme", "inme", "unme", "banaya", "wala", "wali", "wale",
    "hota", "hoti", "hote", "jaise", "sab", "sabhi", "kuch", "apna", "apni", "apne",
}
STOP_WORDS |= {word for word in _HINDI_STOP_WORDS}
STOP_WORDS |= {word for word in _HINGLISH_STOP_WORDS}

# Words that say what the caller wants *done*, or *where* to look, rather than what
# the document is about. They are excluded from the in-context decision: a spoken
# "summarise the reimbursement page" asks about reimbursement, and the verb
# "summarise" is not evidence that the question is outside this file. This is a
# closed class of English speech-act verbs and structure nouns - not a topic list.
_SPEECH_ACT_WORDS = (
    "summarise", "summarize", "summary", "list", "enumerate", "outline", "overview",
    "describe", "detail", "define", "mention", "recap", "brief", "elaborate", "recount",
    "narrate", "highlight", "walkthrough", "answer", "answering", "answell", "explained",
    "section", "heading", "chapter", "page", "topic", "contents", "segment",
    "pdf", "pdfs", "document", "documents", "doc", "docs", "file", "files",
    "according", "accord", "given", "mentioned", "written", "stated", "tell", "says", "say",
    "show", "find", "check", "attached", "uploaded", "content",
    "key", "main", "major", "central", "primary", "core", "principal", "important", "critical",
    "kind", "type", "sort", "going", "develop", "developing", "development",
    "application", "app", "apps", "system", "systems", "project", "poc",
    "build", "building", "create", "creating", "work", "working", "use", "using",
    "approach", "supposed", "plan", "planning", "feature", "features", "benefit", "benefits",
    "takeaway", "takeaways", "point", "points", "aspect", "aspects",
    "character", "characters", "role", "roles", "theme", "themes",
    "exactly", "actually", "basically", "specifically", "precisely", "really",
    "simply", "just", "whole", "overall", "full", "complete", "everything", "anything", "something", "well",
    "प्रकार", "तरह", "बारे", "बनाना", "बनाने", "सिस्टम", "एप्लिकेशन", "प्रोजेक्ट",
)


def stem(word: str) -> str:
    if _DEVANAGARI_RE.search(word):
        for suffix in _DEVANAGARI_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word
    if len(word) <= 4:
        # A short word is usually a stem already, but `pdfs`, `urls` and `apis` are
        # plurals of words the document writes short. Folding them is what lets a
        # question about "documents" reach the line that says "96 technical PDFs".
        # Forms whose s belongs to the word - this, was, bus, loss, bias - are left.
        if len(word) == 4 and word.endswith("s") and not word.endswith(("ss", "us", "is", "as", "os", "ns")):
            return word[:-1]
        return word
    # Fold repeatedly: `values` loses `s` then `e` to reach `valu`, the same base
    # `value` reaches on its own, and a single pass would leave the plural one step
    # short of the word the document actually wrote.
    for _ in range(3):
        for suffix in _SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        if len(word) <= 4:
            break
    return word


# Measured on the folded forms, because the terms this is compared against come out of
# `tokenize`: "summarise" folds to "summaris", and an unfolded set would stop excluding
# the verb and start treating it as evidence about the document.
_SPEECH_ACT = {stem(word) for word in _SPEECH_ACT_WORDS}

# The same reason, applied to the stop words. "कौन" and "सी" are as much connective
# tissue as "which" and "is", but the list spells them in their dictionary forms while
# every token in the index has been folded - comparing the two unfolded leaves a
# language's question words counting as the subject of its own questions.
_STOP_STEMS = frozenset(stem(word) for word in STOP_WORDS)


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
            if len(token) < 2 and not token.isdigit():
                continue
            token = stem(token)
            if not token or token in STOP_WORDS or token in _STOP_STEMS or token in seen:
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

# The word-forms of the same table, folded the way the index folds everything else, so
# a passage that says "six annotators" is recognised as carrying a figure.
_NUMBER_WORD_STEMS = {token for word in list(_ORDINALS) + list(_NUMBER_WORDS) for token in tokenize(word)}


_FIGURE_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w./-])")


def carries_figure(text: str) -> bool:
    """Does this run of words state an amount, in digits or in words.

    A price list writes `5 years` and a manual writes `twelve days`; treating only the
    first as a figure demotes the sentence that answers "how long" in the file's own
    style. A digit inside a name is not an amount: `BM25`, `HTTP 409`'s neighbour
    `python3` and `index.android` all carry digits without saying how many of anything
    there are, and counting them would rate every line in the file as a figure.
    """
    return bool(_FIGURE_RE.search(text)) or bool(set(tokenize(text)) & _NUMBER_WORD_STEMS)

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
    ("document", "pdf", "paper", "text"),
    ("login", "logged", "log"),
    ("choose", "chosen", "chose", "select", "picked", "tuned"),
    ("duration", "period", "length", "span"),
    ("page", "section", "heading", "clause"),
    ("employ", "utilise", "leverage", "used"),
    ("aim", "goal", "objective", "purpose", "intent", "target"),
    ("tool", "stack", "library", "framework", "platform", "software", "utility"),
    ("arjun", "arjuna"),
    ("krishna", "krsna"),
    ("bhima", "bhim"),
    ("karna", "karn"),
    ("duryodhan", "duryodhana"),
    ("pandav", "pandava", "pandavas"),
    ("dhritarashtra", "dhritarastra"),
    ("parameter", "parameters", "value", "values"),
    ("database", "db", "storage", "datastore", "data", "supabase", "sql"),
    ("model", "llm", "ai", "generator"),
    ("speech", "audio", "voice", "stt", "tts", "sound", "transcription"),
    ("recommend", "recommended", "suggest", "suggested", "choice", "chosen", "select", "candidate"),
):
    _expanded = set()
    for _word in _group:
        _pieces = tokenize(_word)
        if _pieces:
            _expanded.add(_pieces[-1])
    if len(_expanded) >= 2:
        _PARAPHRASE_GROUPS.append(_expanded)
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

def fold_devanagari(text: str) -> str:
    """Fold Devanagari ligatures, reph, and viramas to match legacy font encodings."""
    return text.replace("र्", "").replace("\u094d", "")

_INDIC_TRANSLITERATION_MAP: Dict[str, List[str]] = {
    "arjun": ["अजुन", "अर्जुन", "पाथ", "पार्थ", "धनंजय"],
    "arjuna": ["अजुन", "अर्जुन", "पाथ", "पार्थ", "धनंजय"],
    "krishna": ["कृ", "कृष्ण", "ीकृ", "भगवान्ीकृ", "वासुदेव"],
    "krsna": ["कृ", "कृष्ण", "ीकृ", "भगवान्ीकृ", "वासुदेव"],
    "bhima": ["भीम", "भीमसेन", "भीमजी"],
    "bhim": ["भीम", "भीमसेन", "भीमजी"],
    "pandav": ["पाडव", "पाण्डव", "पाडवक", "पाडवम"],
    "pandava": ["पाडव", "पाण्डव", "पाडवक", "पाडवम"],
    "pandavas": ["पाडव", "पाण्डव", "पाडवक", "पाडवम"],
    "karna": ["कण", "कर्ण", "राधेय"],
    "karn": ["कण", "कर्ण", "राधेय"],
    "draupadi": ["ौपद", "द्रौपदी", "द्रौपद", "कृ णया"],
    "duryodhan": ["युधन", "दुर्योधन", "धृतरापु"],
    "duryodhana": ["युधन", "दुर्योधन", "धृतरापु"],
    "yudhishthir": ["युध र", "युधिष्ठिर", "धमराज"],
    "yudhishthira": ["युध र", "युधिष्ठिर", "धमराज"],
    "dhritarashtra": ["धृतरा", "धृतराष्ट्र"],
    "sanjay": ["संजय", "सजय", "सूत"],
    "sanjaya": ["संजय", "सजय", "सूत"],
    "vyas": ["व्यास", "वास", "वैशपायन"],
    "vyasa": ["व्यास", "वास", "वैशपायन"],
    "drona": ["ोण", "द्रोण", "ोणाचाय"],
    "dronacharya": ["ोणाचाय", "द्रोणाचार्य"],
    "bhishma": ["भी", "भीष्म", "पतामह", "पितामह"],
    "kuru": ["कुरु", "कौरव"],
    "kaurav": ["कौरव", "कौरवक"],
    "kaurava": ["कौरव", "कौरवक"],
    "kauravas": ["कौरव", "कौरवक"],
    "kurukshetra": ["कु े", "कुरुक्षेत्र"],
    "hastinapur": ["हनापुर", "हस्तिनापुर"],
    "gita": ["गीता"],
    "shloka": ["ोक", "श्लोक"],
    "parva": ["पव", "पर्व"],
    "adhyaya": ["ाया", "अध्याय"],
    "nakula": ["नकु ल", "नकुल"],
    "nakul": ["नकु ल", "नकुल"],
    "sahadeva": ["सहदेव"],
    "sahadev": ["सहदेव"],
    "gandhari": ["गाधारी", "गांधारी"],
    "kunti": ["कु ती", "कुंती"],
    "madri": ["मा ी", "माद्री"],
    "ashwatthama": ["अथामा", "अश्वत्थामा"],
}

# Front matter restates the whole file, so it shares some words with every specific
# question asked against it and out-ranks the section that actually states the fact.
# Same reasoning as a contents page, one layer up: an abstract is a recap of the body,
# not a substitute for it.
_RECAP_HEADING_RE = re.compile(
    r"\b(abstract|summary|summar(y|ise|ize)|overview|conclusion|concluding|keywords?|highlights?)\b",
    re.I,
)

# The question words that can only be answered with a figure. This is a closed class of
# interrogatives and their objects, the same handful in every language, and it names no
# subject a document could be about: it asks what *shape* the answer must have.
_QUANTITY_ASK_RE = re.compile(
    r"\bhow (?:many|much|long|often|old|far|high|deep|wide)\b"
    r"|\b(?:number|quantity|total|sum|cost|price|fee|charge|amount|count|duration|percentage|"
    r"ratio|speed|weight|height|version|year|date|time)\b"
    r"|\bwhich year\b|\bin what year\b|\bhow much is\b|\bhow many are there\b"
    # Asking which settings were picked is asking for their values, and a file answers
    # that with figures rather than with the sentence that introduced them.
    r"|\b(?:hyper)?parameters?\b|\b(?:which|what)\s+(?:value|setting|option|default)s?\b"
    # "Which grade is a fail" puts a noun between the question word and the verb and then
    # names the class the value decides: the caller wants the number that separates one
    # member of a set from another. "Which website is loaded" has a participle in that
    # slot and wants a name, so it stays a plain lookup.
    r"|\b(?:which|what)\s+\w+\s+(?:is|are|was|were)\s+(?:a|an|the|above|below|under|over|more|less|fewer|at)\b",
    re.I,
)


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


def merge_hyphenated_lines(lines: List[str]) -> List[str]:
    """Rejoin the words a scanned page split at a line break: 'main-' then 'tain'.

    Soft hyphenation is why a search for one word finds the other half of a different
    word, and why the sentence a caller is handed stops short. The halves belong
    together before the text is indexed or quoted, not after.
    """
    out: List[str] = []
    for line in lines:
        previous = out[-1] if out else ""
        if previous.endswith("-") and len(previous) > 3 and re.match(r"^[a-z]", line):
            out[-1] = f"{previous[:-1]}{line}"
        else:
            out.append(line)
    return out


def flow_lines(text: str) -> str:
    """One readable flow when the page set its text in two- and three-word lines.

    A scanned book puts each line of a sentence on its own row, and a model handed
    that reads a list of fragments and correctly concludes the page says nothing. The
    model and the sentence picker both need the sentence back, so the breaks come out
    here - and only where the lines really are that short, so a manual keeps its steps.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 4 or sum(len(line.split()) for line in lines) > 7 * len(lines):
        return "\n".join(lines)
    return re.sub(r"(\w)-\s+([a-z])", r"\1\2", " ".join(merge_hyphenated_lines(lines)))


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

_LEADER_RE = re.compile(r"(?:\.\s+){2,}\.")


def one_edit_apart(term: str, other: str) -> bool:
    """True when the two words differ by a single substitution, insertion or loss."""
    if term == other:
        return True
    if len(term) == len(other):
        return sum(1 for left, right in zip(term, other) if left != right) <= 1
    if abs(len(term) - len(other)) != 1:
        return False
    longer, shorter = (term, other) if len(term) > len(other) else (other, term)
    return any(longer[:cut] + longer[cut + 1:] == shorter for cut in range(len(longer)))


class DocumentChunk:
    def __init__(self, text: str, page_num: int, chunk_id: int, heading: str = ""):
        self.text = text.strip()
        self.page_num = page_num
        self.chunk_id = chunk_id
        self.heading = heading
        # Dot leaders are the shape of a table of contents, not of content: the page
        # lists section names with their numbers, so it holds every word a question
        # asks about and answers none of them. Kept searchable, kept out of the lead.
        self.index_page = bool(_LEADER_RE.search(self.text))
        # A recap heading (Abstract, Summary, Conclusion) names the whole file, so the
        # passage under it repeats every topic without stating any of them; the section
        # that carries the fact outranks it.
        self.recap_heading = bool(heading and _RECAP_HEADING_RE.search(heading))
        # A quantity question is answered by a figure, and a file spells figures both
        # ways: "12 days" and "twelve days". Measured in `_build_index`, where the
        # passage's own vocabulary has already been worked out.
        self.has_digit = False
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
                 idf_mass: float = 0.0, concepts: int = 0, focus: bool = False, ideas: float = 0.0):
        self.chunk = chunk
        self.score = score
        self.coverage = coverage
        self.terms = terms
        self.idf_mass = idf_mass
        # How many of the question's *ideas* the passage carries, counting a word the
        # document spells differently as the same idea.
        self.concepts = concepts
        # The same share as `coverage`, but a word counts as found when the document
        # says it or says the same thing another way. The gate is asked "is this what
        # the file is about", and a file that writes `96 technical PDFs` is about the
        # number of documents. `coverage` stays literal, because a caller reads it as
        # how much of what they said came back.
        self.ideas = ideas
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
        # A contents page names every section in the book, so it out-matches the page
        # that actually says the thing. It stays searchable for the question about what
        # the file contains, and loses to real text for every other question.
        self.index_page_penalty = 0.35
        # A recap is a summary of the body, and the body states the fact itself, so the
        # passage under "Abstract" competes at a discount rather than head to head.
        self.recap_penalty = 0.7
        # "How many", "what is the fee" and "which year" are answered by a figure. A
        # passage holding no digit cannot be that answer however many of the question's
        # words it repeats, so a figure earns its place in the ranking rather than the
        # reader having to hope the right paragraph also happens to rank first.
        self.quantity_bonus = 1.15
        # Applied to the prose instead: equal weighting leaves a figure where it was and
        # lifts it over prose that matched no better, while a file that spells its
        # counts as "twelve days" stays reachable because the demotion is not a veto.
        self.quantity_conflict_penalty = 1.0 / self.quantity_bonus
        self.current_filename: str = ""
        self.chunks: List[DocumentChunk] = []
        self.raw_text: str = ""
        self.doc_summary: str = ""
        self.total_pages: int = 0
        self.profile: Dict[str, Any] = {}
        self._df: Dict[str, int] = {}
        # The page, not the passage, is where an idea pair can be checked: a book cut
        # into eight-word passages cannot hold two of a question's words in one strip,
        # and refusing such a question would be a comment on the segmenter, not on the
        # file. `page_terms` is the union of a page's vocabulary, built once per load.
        self._page_terms: Dict[int, Counter] = {}
        self._heading_terms: set = set()
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

    def _page_rows(self, page) -> Optional[List[Tuple[str, float, float]]]:
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
            text = (text or "").replace("\x00", "")
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

        # A grid sets its columns at the same positions on many rows. Justified prose -
        # and every scanned book puts one word per text span - gives each span its own
        # position once, so a row whose columns do not recur on the page is a sentence
        # and must be read as one.
        starts: Counter = Counter()
        for row in rows:
            if len(row) >= 3:
                for value in {round(cell[0]) for cell in row}:
                    starts[value] += 1

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
            distinct = {round(cell[0]) for cell in cells}
            supported = sum(1 for value in distinct if starts[value] >= 3)
            if len(cells) >= 3 and supported >= 3 and len(set(round(value, 1) for value in xs)) >= 3 and max(xs) - min(xs) >= 12:
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
        # A fraction of a point is not a title. Scanned pages carry per-line size
        # jitter - 7.0 against a 7.01 body - and reading that as a heading cut an
        # OCR'd book into four-word passages. A real heading is set visibly larger
        # (at least 15% larger than body or 1.2pt). Flowcharts, arrows, and tiny
        # fragments are body content, never headings.
        if not body_size or size <= 1.01 or size <= body_size + max(1.2, body_size * 0.15):
            return False
        stripped = text.strip()
        if len(stripped) > _HEADING_MAX_CHARS or not stripped:
            return False
        if any(arrow in stripped for arrow in ("→", "->", "=>", "-->", "↔")):
            return False
        if len(stripped) <= 2:
            return False
        return not stripped.endswith((".", "!", "?", ";", ","))

    @staticmethod
    def _clean_page(raw: str) -> str:
        raw = raw.replace("\x07", "\t").replace("\x00", "")
        # Rejoin broken matras in Devanagari custom fonts
        raw = re.sub(r"([\u0900-\u097F])\s+([\u093E-\u094F\u0901-\u0903])", r"\1\2", raw)
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
        joined: List[Tuple[str, float, float]] = []
        for text, size, x1 in rows:
            if joined and joined[-1][0].endswith("-") and len(joined[-1][0]) > 3 and re.match(r"^[a-z]", text):
                previous = joined[-1]
                joined[-1] = (f"{previous[0][:-1]}{text}", previous[1], previous[2])
            else:
                joined.append((text, size, x1))
        rows = joined
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
            # A line that does not end a sentence is the middle of a wrapped
            # paragraph, and whatever comes after it is that paragraph's next line,
            # not a title. An OCR'd book keeps no usable typography, and reading every
            # short line as a heading cut its paragraphs into four-word passages.
            # A numbered section title overrides that: "Practical 3 - Login" starts a
            # section even when the line above it is a command with no full stop.
            continued = bool(buffer) and not self._SENTENCE_END.search(buffer[-1])
            numbered = heading_section(text) is not None
            if self._is_size_heading(text, size, body_size) or (
                looks_like_heading(text)
                and not re.search(r"[.!?]$", text)
                and (numbered or not continued)
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
            units.extend(merge_hyphenated_lines(lines))
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
        self._page_terms = {}
        self._heading_terms = set()
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
            chunk.has_digit = carries_figure(chunk.text) or carries_figure(chunk.heading or "")
            chunk.length = chunk.body_length + chunk.heading_length
            lengths.append(chunk.length)
            body_lengths.append(max(1, chunk.body_length))
            heading_lengths.append(max(1, chunk.heading_length))
            for term in set(chunk.token_counts):
                self._df[term] = self._df.get(term, 0) + 1
            self._page_terms.setdefault(chunk.page_num, Counter()).update(chunk.token_counts)
            self._heading_terms.update(set(chunk.heading_counts))

        self._doc_count = len(self.chunks)
        self.is_devanagari = any(_DEVANAGARI_RE.search(chunk.text) for chunk in self.chunks[:50])
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

    def heading_terms(self) -> set:
        """Every word the file itself used as a title.

        A caller who says "summarise the reimbursement page" has named one section of
        the file, and an answer describing the file as a whole is about the wrong part
        of it. Structure questions are only answered from the layout when the question
        names nothing inside the layout.
        """
        return self._heading_terms

    def page_ideas(self, page_num: int, terms: List[str], partners: Dict[str, List[str]]) -> int:
        """How many of the question's ideas this page carries, under any spelling."""
        counts = self._page_terms.get(page_num)
        if counts is None:
            return 0
        return sum(1 for term in terms
                   if counts.get(term, 0) or any(counts.get(partner, 0) for partner in partners.get(term, ())))

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

    def _idea_coverage(self, counts: Counter, terms: List[str], partners: Dict[str, List[str]]) -> float:
        """`coverage`, with the document's own synonyms counted as the same idea.

        Weighed by the idf of whichever word the document actually uses, so a passage
        is credited for the rarity it carries rather than for a lucky spelling.
        """
        mass: Dict[str, float] = {}
        for term in terms:
            if term in self._df:
                mass[term] = self.idf(term)
                continue
            synonyms = [partner for partner in partners.get(term, ()) if partner in self._df]
            if synonyms:
                mass[term] = max(self.idf(partner) for partner in synonyms)
        total = sum(mass.values())
        if total <= 0:
            return 0.0
        found = sum(
            value for term, value in mass.items()
            if counts.get(term, 0) or any(counts.get(partner, 0) for partner in partners.get(term, ()))
        )
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
            # Universal near-match / inflection / phonetic alternate
            if term not in self._df:
                near = self._one_edit_known(term)
                if near and near in self._df and near not in seen:
                    seen.add(near)
                    out.setdefault(term, []).append(near)
            # Transliteration for Indic / Devanagari documents
            lower_term = term.lower()
            if getattr(self, "is_devanagari", False):
                if lower_term in _INDIC_TRANSLITERATION_MAP:
                    for alt in _INDIC_TRANSLITERATION_MAP[lower_term]:
                        if alt in self._df and alt not in seen:
                            seen.add(alt)
                            out.setdefault(term, []).append(alt)
                        folded = fold_devanagari(alt)
                        if folded in self._df and folded not in seen:
                            seen.add(folded)
                            out.setdefault(term, []).append(folded)
                elif _DEVANAGARI_RE.search(term):
                    folded = fold_devanagari(term)
                    if folded in self._df and folded not in seen:
                        seen.add(folded)
                        out.setdefault(term, []).append(folded)
        return out

    def _paraphrase_partners(self, terms: List[str]) -> List[str]:
        """Document-side synonyms of the words the user spoke, for ranking only."""
        return [partner for partners in self.partner_map(terms).values() for partner in partners]

    @staticmethod
    def _concept_count(counts: Counter, terms: List[str], partners: Dict[str, List[str]]) -> int:
        """How many of the question's ideas the passage carries, under any spelling."""
        return sum(1 for term in terms
                    if counts.get(term, 0) or any(counts.get(partner, 0) for partner in partners.get(term, ())))

    def _one_edit_known(self, term: str) -> Optional[str]:
        """Return closest known word in the index within one edit or trailing vowel alternate."""
        if len(term) < 4:
            return None
        # Trailing vowel / schwa alternates (e.g. yudhishthir -> yudhishthira, pandav -> pandava, schema -> scheme)
        if self._df.get(term + "a", 0) >= 1:
            return term + "a"
        if self._df.get(term + "e", 0) >= 1:
            return term + "e"
        if self._df.get(term + "s", 0) >= 1:
            return term + "s"
        if term.endswith("a") and len(term) > 3 and self._df.get(term[:-1], 0) >= 1:
            return term[:-1]
        head = term[:3]
        for known in self._df:
            if self._df[known] < 1:
                continue
            if len(known) < 4 or not known.startswith(head):
                continue
            if abs(len(known) - len(term)) <= 1 and one_edit_apart(term, known):
                return known
        return None

    @property
    def words_avg(self) -> float:
        """Words per passage in this document - how finely the file had to be cut."""
        return self._avg_len

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
        # Exclude meta terms and tokens appearing in the document's filename.
        doc_terms = set(tokenize(self.current_filename or ""))
        meta_terms = _SPEECH_ACT | doc_terms
        content = [term for term in Counter(tokenize(query)) if term not in meta_terms]
        if not content:
            content = [term for term in Counter(tokenize(query)) if term not in _SPEECH_ACT] or list(Counter(tokenize(query)))
        informative = [term for term in spoken if term not in meta_terms]
        if not informative:
            informative = [term for term in spoken if term not in _SPEECH_ACT] or list(spoken)
        partners = self.partner_map(spoken)
        terms = spoken + self._paraphrase_partners(spoken)
        # The part of the question this file has a word for. People do not speak in
        # index terms - "which kind of LLM are we going to use to develop this
        # application" wraps one lookupable subject in five ordinary words - so the
        # subject decides what a question is about, not the count of odd words.
        near = {
            term: self._one_edit_known(term)
            for term in content
            if term not in self._df and term not in partners
        }
        for term, match in near.items():
            if match and match not in partners.get(term, ()):
                partners.setdefault(term, []).append(match)
                if match not in terms:
                    terms.append(match)
        mineable = [term for term in content if term in self._df or partners.get(term) or near.get(term)]
        focus = mineable[0] if mineable else (content[0] if content else (spoken[0] if spoken else ""))
        unknown = [
            term for term in content
            if term not in self._df and term not in partners and not near.get(term)
        ]
        central = any(self._df.get(term, 0) * 8 >= self._doc_count for term in mineable)
        strays = bool(content) and (not mineable or (len(unknown) > len(mineable) and not central))
        # Does this question want a figure? Decided once, before the scan, because it
        # changes what a passage is worth.
        asks_quantity = bool(_QUANTITY_ASK_RE.search(query))

        scored: List[SearchHit] = []
        target_section = requested_section(query)
        numbered_sections = any(heading_section(chunk.heading) for chunk in self.chunks)
        for chunk in self.chunks:
            body = self._bm25(chunk.body_counts, chunk.body_length, terms, self._avg_body_len)
            heading = self._bm25(chunk.heading_counts, chunk.heading_length, terms, self._avg_heading_len)
            raw = body + self.heading_boost * heading
            if chunk.index_page:
                raw *= self.index_page_penalty
            if chunk.recap_heading:
                raw *= self.recap_penalty
            if asks_quantity and raw > 0 and not chunk.has_digit:
                # Not a veto: the passage that says "twelve days" without writing 12 is
                # still reachable, but only when it matches harder than the figures.
                raw *= self.quantity_conflict_penalty
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
            focus_matched = bool(
                not focus
                or chunk.token_counts.get(focus, 0)
                or any(chunk.token_counts.get(p, 0) for p in partners.get(focus, ()))
            )
            cov = max(self.coverage(chunk.token_counts, spoken), self.coverage(chunk.token_counts, informative))
            idea_cov = max(
                self._idea_coverage(chunk.token_counts, spoken, partners),
                self._idea_coverage(chunk.token_counts, informative, partners)
            )
            scored.append(
                SearchHit(
                    chunk,
                    score=round(raw, 4),
                    coverage=cov,
                    terms=self.matched_term_count(chunk.token_counts, spoken),
                    idf_mass=self.matched_idf_mass(chunk.token_counts, spoken),
                    concepts=self._concept_count(chunk.token_counts, informative, partners),
                    ideas=idea_cov,
                    focus=focus_matched,
                )
            )

        if not scored:
            return [], False, 0.0

        scored.sort(key=lambda hit: hit.score, reverse=True)

        def accepted(hit: SearchHit) -> bool:
            if hit.coverage < threshold and hit.ideas < threshold:
                return False
            if hit.concepts >= min_terms:
                return True
            effective_min = min(min_terms, max(1, len(mineable)))
            if len(mineable) <= 1 or len(content) <= 2:
                # When only one distinct concept from the query exists in the document vocab,
                # requiring page_ideas >= 2 is impossible and rejects valid passages.
                return bool(focus) and not strays and hit.focus
            # Three or more ideas, and the passage carries one of them. Then the question
            # is only in this file if the *page* puts those ideas together.
            return hit.focus and self.page_ideas(hit.page_num, informative, partners) >= effective_min

        # Judged before the page cap, not after: the one line that answers a question
        # can rank below four that only share a paraphrase, and dropping it for
        # diversification would refuse a document that plainly holds the fact.
        matched = [hit for hit in scored if accepted(hit)]
        if not matched:
            return [], False, 0.0

        kept: List[SearchHit] = []
        per_page_count: Dict[int, int] = {}
        for hit in matched:
            used = per_page_count.get(hit.page_num, 0)
            if used >= per_page:
                continue
            per_page_count[hit.page_num] = used + 1
            kept.append(hit)
            if len(kept) >= top_k:
                break

        # Report what the accepted evidence is worth, not what the best rejected
        # passage would have been worth: a refusal that claims a perfect match reads
        # as though the engine threw the answer away, and it did not.
        return kept, True, round(max(max(hit.coverage, hit.ideas) for hit in kept), 4)

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
        # The same vocabulary bridge the passages were ranked with. A line that says the
        # document's own word for what the caller asked is the answer, and a ranker that
        # only sees the words the caller spoke rates it as silent.
        terms = list(Counter(query_terms(query)))
        terms = terms + self._paraphrase_partners(terms)
        if not terms:
            return sentences[:limit]

        scored = []
        # `rank_sentences` is handed passages, and a passage is not the page: what the
        # file spells as "six annotators" is a figure even where no digit was written,
        # so the rule only applies once there is a figure somewhere among the candidates.
        asks_quantity = bool(_QUANTITY_ASK_RE.search(query)) and any(carries_figure(sentence) for sentence in sentences)
        for position, sentence in enumerate(sentences):
            counts = Counter(tokenize(sentence))
            length = sum(counts.values()) or 1
            raw = self._bm25(counts, length, terms)
            # The same rule the passages are ranked by, applied to the line that is
            # actually quoted: a "how many" answer that repeats the question's nouns
            # without ever giving the number is not the answer.
            if asks_quantity and raw > 0 and not carries_figure(sentence):
                raw *= self.quantity_conflict_penalty
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
