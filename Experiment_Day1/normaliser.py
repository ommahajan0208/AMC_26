"""
normaliser.py
=============
Business Entity Resolution — Pre-cleaning & Normalisation Pipeline

Covers every noise pattern catalogued in ER_Strategy.md:
  Names    : N1–N26  (26 patterns)
  Addresses: A1–A37  (37 patterns)
  Country  : C1–C7   (7 patterns)

Country plugin system:
  BaseNormaliser → USNormaliser
               → IndiaNormaliser
               → FranceNormaliser
               → GenericNormaliser   (unseen countries)

Entry points:
  normalise_record(entity_id, business_name, business_address, country)
      → NormalisedRecord (dataclass)

  canonical_country(raw) → str   ("US" / "India" / "France" / raw.title())
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Optional dependency: indic-transliteration (Q29 — confirmed locked)
# ---------------------------------------------------------------------------
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _itrans
    _INDIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _INDIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional dependency: rapidfuzz (used for fuzzy country matching C4)
# ---------------------------------------------------------------------------
try:
    from rapidfuzz.distance import Levenshtein as _Lev
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RAPIDFUZZ_AVAILABLE = False


# ===========================================================================
# 0. OUTPUT DATACLASS
# ===========================================================================

@dataclass
class NormalisedRecord:
    """All derived fields produced by the normalisation pipeline."""

    # --- identifiers ---
    entity_id: str
    source: str           # "S1" / "S2" / "S3"

    # --- country ---
    country_raw: str
    country_canonical: str   # "US" / "India" / "France" / best-guess title

    # --- name fields ---
    name_raw: str
    name_normalized: str     # cleaned, lowercased, abbrevs expanded
    name_core: str           # legal suffixes stripped
    name_token_set: frozenset # sorted unique tokens of name_core
    name_trade: str          # DBA trade name (empty if none)
    name_legal: str          # legal entity name (= name_normalized if no DBA split)

    # --- name flags ---
    is_domain: bool
    base_domain: str         # e.g. "heassociates" (TLD stripped)
    tld: str                 # e.g. ".com"
    tld_country_signal: str  # "us" / "india" / "france" / "generic" / "unknown"

    # --- address fields ---
    addr_raw: str
    addr_normalized: str
    addr_tokens: frozenset
    addr_numeric_tokens: frozenset  # purely numeric tokens

    # --- address flags ---
    addr_is_null: bool
    has_landmark: bool
    has_bank_atm_landmark: bool

    # --- script ---
    dominant_script: str     # "latin" / "devanagari" / "tamil" / "mixed" / "unknown"


# ===========================================================================
# 1. CONSTANTS & LOOKUP TABLES
# ===========================================================================

# ---- 1A. Country canonical mapping (C1–C3) --------------------------------

_COUNTRY_MAP: dict[str, str] = {
    # United States
    "us": "US", "usa": "US", "u.s.": "US", "u.s.a.": "US",
    "united states": "US", "united states of america": "US",
    "america": "US",
    # India
    "india": "India", "in": "India", "bharat": "India",
    "inde": "India",        # French spelling
    "indien": "India",      # German spelling (edge)
    # France
    "france": "France", "fr": "France",
    "republique francaise": "France",
    "république française": "France",
    "french republic": "France",
}

# ---- 1B. Known TLDs → country signal (Q20) --------------------------------

_KNOWN_TLDS: dict[str, str] = {
    ".com": "generic", ".net": "generic", ".org": "generic",
    ".edu": "us",      ".gov": "us",      ".mil": "us",
    ".in":  "india",   ".co.in": "india",
    ".fr":  "france",
    ".us":  "us",      ".io": "generic",  ".co": "generic",
    ".biz": "generic", ".info": "generic",
    ".uk":  "generic", ".ca": "generic",
    ".de":  "generic", ".au": "generic",
}

# ---- 1C. Legal suffixes to strip (per country) ----------------------------

_LEGAL_SUFFIXES_COMMON = [
    r"\bllc\b", r"\bl\.l\.c\.\b",
    r"\binc\b", r"\bincorporated\b",
    r"\bcorp\b", r"\bcorporation\b",
    r"\bltd\b", r"\blimited\b",
    r"\bco\b(?!\s*\.\s*(?:in|uk))",   # "Co" but not ".co.in"
    r"\bcompany\b",
    r"\bgroup\b",
    r"\bholdings\b", r"\bholding\b",
    r"\benterprises\b", r"\benterprise\b",
    r"\bventures\b", r"\bventure\b",
    r"\bservices\b", r"\bservice\b",
    r"\bsolutions\b", r"\bsolution\b",
    r"\bindustries\b", r"\bindustry\b",
    r"\btrading\b",
    r"\binternational\b", r"\bintl\b",
    r"\bnational\b", r"\bnat'l\b",
    r"\bassociates\b", r"\bassoc\b",
    r"\bpartners\b",
    r"\bbrothers\b", r"\bbros\b",
    r"\bsons\b",
    r"\bmanagement\b", r"\bmgmt\b",
    r"\bconsulting\b",
    r"\btechnologies\b", r"\btechnology\b", r"\btech\b",
    r"\bsystems\b", r"\bsystem\b",
    r"\bglobal\b",
]

_LEGAL_SUFFIXES_US = [
    r"\bllp\b", r"\bl\.l\.p\.\b",
    r"\bpllc\b",
    r"\bpa\b",          # Professional Association
    r"\bpc\b",          # Professional Corporation
    r"\bna\b",          # National Association (banks)
    r"\bfsb\b",         # Federal Savings Bank
    r"\bnfp\b",         # Not-for-profit
]

_LEGAL_SUFFIXES_INDIA = [
    r"\bpvt\b", r"\bprivate\b",
    r"\bpvt\s+ltd\b", r"\bprivate\s+limited\b",
    r"\bopc\b",         # One Person Company
    r"\bllp\b",
    r"\blimited\s+liability\s+partnership\b",
]

_LEGAL_SUFFIXES_FRANCE = [
    r"\bsas\b",    # Société par Actions Simplifiée
    r"\bsarl\b",   # Société à Responsabilité Limitée
    r"\beurl\b",   # Entreprise Unipersonnelle à Responsabilité Limitée
    r"\bsasu\b",   # Société par Actions Simplifiée Unipersonnelle
    r"\bsnc\b",    # Société en Nom Collectif
    r"\bgie\b",    # Groupement d'Intérêt Économique
    r"\bsci\b",    # Société Civile Immobilière
    r"\bsa\b",     # Société Anonyme
    r"\bsei\b",
    r"\bscea\b",
    r"\bgaec\b",
    r"\bearl\b",
]

# ---- 1D. Business abbreviation expansion (N1, N20) ------------------------

_ABBREV_EXPAND: dict[str, str] = {
    # Legal
    "corp":     "corporation",
    "inc":      "incorporated",
    "ltd":      "limited",
    "pvt":      "private",
    "llc":      "limited liability company",
    "llp":      "limited liability partnership",
    # Common
    "intl":     "international",
    "nat'l":    "national",
    "natl":     "national",
    "mktg":     "marketing",
    "mgmt":     "management",
    "assoc":    "associates",
    "assn":     "association",
    "dept":     "department",
    "univ":     "university",
    "hosp":     "hospital",
    "tech":     "technology",
    "svc":      "service",
    "svcs":     "services",
    "mfg":      "manufacturing",
    "bldg":     "building",
    "grp":      "group",
    "hldgs":    "holdings",
    "inv":      "investments",
    "sys":      "systems",
    "comm":     "communications",
    "dev":      "development",
    "res":      "resources",
    "fin":      "financial",
    "acct":     "accounting",
    "const":    "construction",
    "consult":  "consulting",
    "gen":      "general",
    "ind":      "industries",
    "prop":     "properties",
    "realty":   "realty",
    "&":        "and",
}

# ---- 1E. Street / address abbreviations -----------------------------------

_STREET_ABBREV: dict[str, str] = {
    r"\brd\b":     "road",
    r"\bst\b":     "street",
    r"\bave\b":    "avenue",
    r"\bav\b":     "avenue",
    r"\bblvd\b":   "boulevard",
    r"\bdr\b":     "drive",
    r"\bln\b":     "lane",
    r"\bct\b":     "court",
    r"\bpl\b":     "place",
    r"\bpkwy\b":   "parkway",
    r"\bhwy\b":    "highway",
    r"\bfwy\b":    "freeway",
    r"\bexpy\b":   "expressway",
    r"\bsq\b":     "square",
    r"\bcir\b":    "circle",
    r"\bter\b":    "terrace",
    r"\bterr\b":   "terrace",
    r"\bxing\b":   "crossing",
    r"\bjct\b":    "junction",
    r"\bfte\b":    "fort",
    r"\bmtn\b":    "mountain",
    r"\bmt\b":     "mount",
    r"\blk\b":     "lake",
    r"\bvly\b":    "valley",
    r"\bhts\b":    "heights",
    r"\bvil\b":    "village",
    r"\bsts\b":    "streets",
    r"\bn\b":      "north",
    r"\bs\b":      "south",
    r"\be\b":      "east",
    r"\bw\b":      "west",
    r"\bne\b":     "northeast",
    r"\bnw\b":     "northwest",
    r"\bse\b":     "southeast",
    r"\bsw\b":     "southwest",
    r"\bste\b":    "suite",
    r"\bapt\b":    "apartment",
    r"\bfl\b":     "floor",
    r"\bflr\b":    "floor",
    r"\bno\b":     "number",
    r"\bpo\b":     "post office",
}

# ---- 1F. Landmark patterns (A8, A25) --------------------------------------

_LANDMARK_PREFIXES = re.compile(
    r"\b(near|opp|opposite|beside|behind|next\s+to|b/h|in\s+front\s+of|adj|adjacent)\b",
    re.IGNORECASE,
)

_BANK_ATM_PATTERN = re.compile(
    r"\b(sbi|hdfc|icici|axis|pnb|bob|canara|union\s+bank|kotak|atm|bank)\b",
    re.IGNORECASE,
)

# ---- 1G. Number words (N19) -----------------------------------------------

_NUMBER_WORDS: dict[str, str] = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# ---- 1H. Devanagari / Indic digit maps (N24) ------------------------------

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_BENGALI_DIGITS    = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_TAMIL_DIGITS      = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")
_TELUGU_DIGITS     = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")
_GUJARATI_DIGITS   = str.maketrans("૦૧૨૩૪૫૬૭૮૯", "0123456789")

# ---- 1I. Indian script → Scheme mapping for indic-transliteration (Q29) --

# Maps Unicode block ranges to sanscript scheme names
_INDIC_SCHEMES = [
    (0x0900, 0x097F, "devanagari"),   # Hindi, Marathi, etc.
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
]


# ===========================================================================
# 2. UTILITY HELPERS
# ===========================================================================

def _detect_dominant_script(text: str) -> str:
    """Detect the dominant Unicode script of a text string."""
    if not text:
        return "unknown"
    counts: dict[str, int] = {"latin": 0, "devanagari": 0,
                               "tamil": 0, "other_indic": 0, "other": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
            counts["latin"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["tamil"] += 1
        elif any(lo <= cp <= hi for lo, hi, _ in _INDIC_SCHEMES):
            counts["other_indic"] += 1
        elif ch.isalpha():
            counts["other"] += 1

    total = sum(counts.values())
    if total == 0:
        return "unknown"
    dominant = max(counts, key=counts.__getitem__)
    ratio = counts[dominant] / total
    if ratio < 0.60:
        return "mixed"
    return dominant


def _transliterate_indic_to_roman(text: str) -> str:
    """
    Convert any Indic script tokens to IAST Roman equivalents.
    Falls back gracefully if indic_transliteration is not installed.
    N.B. Only converts chars in known Indic Unicode blocks; Latin chars pass through.
    """
    if not _INDIC_AVAILABLE or not text:
        return text

    result_tokens = []
    for word in text.split():
        converted = False
        for lo, hi, scheme in _INDIC_SCHEMES:
            # Check if majority of chars in this word are from this block
            block_chars = sum(1 for ch in word if lo <= ord(ch) <= hi)
            if block_chars / max(len(word), 1) > 0.5:
                try:
                    romanised = _itrans(word, scheme, sanscript.IAST)
                    result_tokens.append(romanised.lower())
                    converted = True
                    break
                except Exception:
                    pass
        if not converted:
            result_tokens.append(word)
    return " ".join(result_tokens)


def _strip_unicode_format_chars(text: str) -> str:
    """Strip Unicode 'Cf' (format/invisible) characters — N17."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _normalise_nfc(text: str) -> str:
    """NFC Unicode normalisation — handles French diacritics (N10)."""
    return unicodedata.normalize("NFC", text)


def _apply_devanagari_digits(text: str) -> str:
    """Convert Indic script digits to ASCII — N24."""
    return (text
            .translate(_DEVANAGARI_DIGITS)
            .translate(_BENGALI_DIGITS)
            .translate(_TAMIL_DIGITS)
            .translate(_TELUGU_DIGITS)
            .translate(_GUJARATI_DIGITS))


# ===========================================================================
# 3. COUNTRY NORMALISATION  (C1–C7)
# ===========================================================================

def canonical_country(raw: Optional[str]) -> str:
    """
    Map raw country string to one of {"US", "India", "France"} or
    best-guess title-case string for unseen countries.

    Handles: aliases (C1–C3), case variants (C5), trailing dots (C6),
    typos via Levenshtein ≤ 2 (C4), unseen → GenericNormaliser (C7).
    """
    if not raw or not isinstance(raw, str):
        return "Unknown"

    cleaned = raw.strip().lower()
    # C6 — strip trailing/internal dots
    cleaned = re.sub(r"\.", "", cleaned).strip()

    # Direct lookup (C1–C3, C5)
    if cleaned in _COUNTRY_MAP:
        return _COUNTRY_MAP[cleaned]

    # Fuzzy Levenshtein ≤ 2 fallback (C4)
    if _RAPIDFUZZ_AVAILABLE:
        for key, canonical in _COUNTRY_MAP.items():
            if abs(len(cleaned) - len(key)) <= 3:
                dist = _Lev.distance(cleaned, key)
                if dist <= 2:
                    return canonical
    else:
        # Pure-Python fallback — simple edit distance for short strings
        for key, canonical in _COUNTRY_MAP.items():
            if abs(len(cleaned) - len(key)) <= 2:
                if _simple_edit_distance(cleaned, key) <= 2:
                    return canonical

    # C7 — unseen country → best-guess title case
    return raw.strip().title()


def _simple_edit_distance(a: str, b: str) -> int:
    """Minimal edit distance (no external deps)."""
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        new_row = [i + 1]
        for j, cb in enumerate(b):
            new_row.append(min(row[j + 1] + 1,
                               new_row[j] + 1,
                               row[j] + (ca != cb)))
        row = new_row
    return row[-1]


# ===========================================================================
# 4. NAME NORMALISATION  (N1–N26)
# ===========================================================================

# ---- Compiled regex constants ----

# N6 — leading non-alphanumeric noise
_RE_LEADING_NOISE = re.compile(
    r'^[^\w\u0900-\uFFFF]+',  # strips *** << ## -- etc. before first word char
    re.UNICODE,
)

# N12 — trailing punctuation
_RE_TRAILING_PUNCT = re.compile(r'[.,;:!?\-\'"]+$')

# N13 — trademark symbols
_RE_TRADEMARK = re.compile(r'[®™©]')

# N15 — repeated legal suffix tokens
_RE_REPEATED_SUFFIX = re.compile(
    r'\b(ltd|pvt|inc|corp|llc|llp|limited|private|incorporated|corporation)\b'
    r'(\s+\1)+',
    re.IGNORECASE,
)

# N16 — HTML entities
# (handled by html.unescape — no regex needed)

# N18 — ordinal suffixes attached to numbers
_RE_ORDINAL = re.compile(r'(\d+)(st|nd|rd|th)\b', re.IGNORECASE)

# N22 — phone/fax in name
_RE_PHONE_IN_NAME = re.compile(
    r'\b(tel|fax|ph|phone|mobile|mob)\s*:?\s*[\d\-\+\(\)]{7,}',
    re.IGNORECASE,
)

# N23 — emoji / non-printable symbols (keep letters, digits, basic punct, spaces)
_RE_EMOJI = re.compile(r'[^\w\s\-&\'.,/\u0900-\uFFFF]', re.UNICODE)

# N25 — www. prefix
_RE_WWW = re.compile(r'^www\.', re.IGNORECASE)

# N11 — collapse whitespace
_RE_WHITESPACE = re.compile(r'\s+')

# N14 — parenthetical content e.g. "(Formerly ABC Ltd)"
_RE_PARENS = re.compile(r'\(([^)]*)\)')

# N5 — bracket legal suffix e.g. "[LIMITED]"
_RE_BRACKETS = re.compile(r'\[([^\]]*)\]')

# N9 — DBA / trading as split
_RE_DBA = re.compile(r'\b(dba|d/b/a|d\.b\.a\.?|t/a|t\.a\.?|trading\s+as)\b',
                     re.IGNORECASE)


def _expand_abbreviations(text: str) -> str:
    """Expand known business abbreviations — N1, N20."""
    tokens = text.split()
    out = []
    for tok in tokens:
        lower = tok.lower().rstrip('.')
        if lower in _ABBREV_EXPAND:
            out.append(_ABBREV_EXPAND[lower])
        else:
            out.append(tok)
    return " ".join(out)


def _deduplicate_adjacent_tokens(text: str) -> str:
    """Remove repeated adjacent identical tokens — N4."""
    tokens = text.split()
    if not tokens:
        return text
    out = [tokens[0]]
    for tok in tokens[1:]:
        if tok.lower() != out[-1].lower():
            out.append(tok)
    return " ".join(out)


def _normalise_number_words(text: str) -> str:
    """Replace spelled-out numbers 1–10 with digits — N19."""
    tokens = text.split()
    return " ".join(
        _NUMBER_WORDS.get(tok.lower(), tok) for tok in tokens
    )


def _strip_legal_suffixes(text: str, country: str) -> str:
    """
    Strip legal suffixes to get name_core.
    Order: country-specific first, then common.
    """
    suffixes = list(_LEGAL_SUFFIXES_COMMON)
    if country == "US":
        suffixes = _LEGAL_SUFFIXES_US + suffixes
    elif country == "India":
        suffixes = _LEGAL_SUFFIXES_INDIA + suffixes
    elif country == "France":
        suffixes = _LEGAL_SUFFIXES_FRANCE + suffixes

    result = text
    changed = True
    while changed:
        changed = False
        for pat in suffixes:
            new = re.sub(pat, "", result, flags=re.IGNORECASE).strip()
            if new != result:
                result = new
                changed = True
    return _RE_WHITESPACE.sub(" ", result).strip()


def _process_domain_name(name: str) -> dict:
    """
    Detect domain-as-name pattern and extract base_domain + TLD — N8, N21, N25.
    Returns dict with keys: is_domain, base_domain, tld, tld_country_signal.
    """
    clean = name.strip().lower()
    # Must be a single token (no spaces) to be considered a domain
    if " " in clean:
        return {"is_domain": False, "base_domain": clean,
                "tld": "", "tld_country_signal": "unknown"}

    clean = _RE_WWW.sub("", clean)   # strip www.

    for known_tld in sorted(_KNOWN_TLDS.keys(), key=len, reverse=True):
        if clean.endswith(known_tld) and len(clean) > len(known_tld):
            base = clean[: -len(known_tld)]
            return {
                "is_domain": True,
                "base_domain": base,
                "tld": known_tld,
                "tld_country_signal": _KNOWN_TLDS[known_tld],
            }

    return {"is_domain": False, "base_domain": clean,
            "tld": "", "tld_country_signal": "unknown"}


def normalise_name(raw_name: str, country: str) -> dict:
    """
    Full name normalisation pipeline covering N1–N26.

    Returns dict with keys:
      name_normalized, name_core, name_token_set,
      name_trade, name_legal,
      is_domain, base_domain, tld, tld_country_signal
    """
    if not isinstance(raw_name, str):
        raw_name = str(raw_name) if raw_name is not None else ""

    text = raw_name

    # N16 — HTML entities
    text = html.unescape(text)

    # N17 — zero-width / Unicode Cf chars
    text = _strip_unicode_format_chars(text)

    # N10 — NFC normalisation (French diacritics)
    text = _normalise_nfc(text)

    # N24 — Indic/Devanagari digits → ASCII
    text = _apply_devanagari_digits(text)

    # N13 — trademark symbols
    text = _RE_TRADEMARK.sub("", text)

    # N23 — emoji / non-printable (keep Indic Unicode range)
    text = _RE_EMOJI.sub(" ", text)

    # N6 — leading non-alphanumeric noise
    text = _RE_LEADING_NOISE.sub("", text)

    # N12 — trailing punctuation
    text = _RE_TRAILING_PUNCT.sub("", text).strip()

    # N22 — phone/fax in name
    text = _RE_PHONE_IN_NAME.sub("", text)

    # N5 — bracket legal suffix e.g. [LIMITED] → LIMITED
    def _bracket_replace(m: re.Match) -> str:
        return m.group(1)
    text = _RE_BRACKETS.sub(_bracket_replace, text)

    # N14 — parenthetical former/also-known-as names (extract, then strip)
    paren_alts: list[str] = _RE_PARENS.findall(text)
    text = _RE_PARENS.sub("", text).strip()

    # N8 / N21 / N25 — domain name detection (before lowercasing for TLD match)
    domain_info = _process_domain_name(text)

    # N1 — lowercase
    text = text.lower()

    # N11 — collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    # N18 — ordinal suffixes ("2nd" → "2")
    text = _RE_ORDINAL.sub(r"\1", text)

    # N19 — number words → digits
    text = _normalise_number_words(text)

    # Q29 — transliterate Indic script tokens → Roman (for TF-IDF only path)
    # We keep original for BGE-m3; the transliterated form is stored in name_normalized
    script = _detect_dominant_script(text)
    if script in ("devanagari", "tamil", "other_indic", "mixed"):
        text_for_tfidf = _transliterate_indic_to_roman(text)
    else:
        text_for_tfidf = text

    # N9 — DBA split: "Ectolumdrex dba X+ Madison Inc" → legal / trade
    dba_parts = _RE_DBA.split(text_for_tfidf, maxsplit=1)
    if len(dba_parts) >= 3:
        name_legal = dba_parts[0].strip()
        name_trade = dba_parts[-1].strip()
    else:
        name_legal = text_for_tfidf.strip()
        name_trade = ""

    # Include any parenthetical alt-name as trade name if trade is empty
    if not name_trade and paren_alts:
        name_trade = " ".join(paren_alts).lower().strip()

    # Abbreviation expansion (N1, N20) — on legal name
    name_legal_exp = _expand_abbreviations(name_legal)

    # N4 — deduplicate adjacent identical tokens
    name_legal_exp = _deduplicate_adjacent_tokens(name_legal_exp)

    # N15 — deduplicate repeated legal suffix tokens
    name_legal_exp = _RE_REPEATED_SUFFIX.sub(r"\1", name_legal_exp)

    # N26 — strip noise business suffixes ("& Sons", "& Co", "& Partners")
    name_legal_exp = re.sub(
        r'\s*&\s*(sons|co|co\.|partners|associates)\s*$',
        "", name_legal_exp, flags=re.IGNORECASE,
    ).strip()

    # N11 — final whitespace collapse
    name_normalized = _RE_WHITESPACE.sub(" ", name_legal_exp).strip()

    # Strip legal suffixes → name_core
    name_core = _strip_legal_suffixes(name_normalized, country)
    name_core = _RE_WHITESPACE.sub(" ", name_core).strip()

    # Token set of name_core (sorted unique tokens)
    name_token_set = frozenset(name_core.split()) if name_core else frozenset()

    return {
        "name_normalized": name_normalized,
        "name_core": name_core,
        "name_token_set": name_token_set,
        "name_trade": name_trade,
        "name_legal": name_legal,
        "is_domain": domain_info["is_domain"],
        "base_domain": domain_info["base_domain"],
        "tld": domain_info["tld"],
        "tld_country_signal": domain_info["tld_country_signal"],
    }


# ===========================================================================
# 5. ADDRESS NORMALISATION  (A1–A37)
# ===========================================================================

# ---- Compiled regex constants for addresses ----

# A2 — literal NULL token
_RE_NULL_TOKEN = re.compile(r'\bNULL\b', re.IGNORECASE)

# A5 — double-hash prefix e.g. ##2326
_RE_DOUBLE_HASH = re.compile(r'#{2,}\s*')

# A6 — PO Box
_RE_PO_BOX = re.compile(r'\bP\.?O\.?\s*Box\s*\d*\b', re.IGNORECASE)

# A7 — C/O prefix
_RE_CO_PREFIX = re.compile(r'^\s*C/O\s+', re.IGNORECASE)

# A9 — fractional address numbers "19 1/2"
_RE_FRACTION = re.compile(r'(\d+)\s*/\s*\d+')

# A10 — ordinal floors/crossings "2Nd Floor" etc. (reuse name ordinal regex)
# (same _RE_ORDINAL)

# A11 — unit variants
_RE_UNIT_VARIANTS = re.compile(
    r'\bunit\s+(building|apartment|apt|ste|suite|block)?\s*',
    re.IGNORECASE,
)

# A12 — floor variants "Fl 3", "2F", "2nd Flr", "F-0"
_RE_FLOOR_VARIANTS = re.compile(
    r'\b(?:fl(?:oor)?\.?|flr\.?)\s*[-–]?\s*(\d+)\b'
    r'|\b(\d+)\s*(?:f|fl|flr|floor)\b'
    r'|\bf[-–](\d+)\b',
    re.IGNORECASE,
)

# A13 — slash house numbers (India): "12/A" → "12a"
_RE_SLASH_HOUSE = re.compile(r'(\d+)/([A-Za-z])\b')

# A18 — Cedex (France)
_RE_CEDEX = re.compile(r'\bCedex\s*\d*\b', re.IGNORECASE)

# A19 — Arrondissement (France) "Paris 1er", "Lyon 13ème"
_RE_ARROND = re.compile(
    r'\b(\d+)\s*(?:e|er|ème|eme|ième|ieme)?\b',
    re.IGNORECASE,
)

# A23 — URL in address
_RE_URL_IN_ADDR = re.compile(
    r'\b(?:https?://|www\.)\S+',
    re.IGNORECASE,
)

# A24 — phone in address
_RE_PHONE_IN_ADDR = re.compile(
    r'\b(?:ph|tel|mobile|mob|fax)\s*:?\s*[\d\-\+\(\)]{7,}',
    re.IGNORECASE,
)

# A26 — road number format "Road No.10", "Road #5"
_RE_ROAD_NO = re.compile(r'\broad\s*(?:no\.?|#)\s*(\d+)', re.IGNORECASE)

# A27 — Survey/Gat/Plot numbers
_RE_SURVEY = re.compile(
    r'\b(?:gat|s\.?\s*no|survey|plot|khasra)\s*\.?\s*(?:no\.?)?\s*([\d/]+)',
    re.IGNORECASE,
)

# A28 — Door No (South India)
_RE_DOOR_NO = re.compile(r'\bd(?:oor)?\s*\.?\s*no\.?\s*([\d\-/]+)', re.IGNORECASE)

# A29 — House No variations
_RE_HOUSE_NO = re.compile(
    r'\bh(?:ouse)?\s*\.?\s*n(?:o|um)?\.?\s*[-–]?\s*([\d\-/A-Za-z]+)',
    re.IGNORECASE,
)

# A30 — KH No (land records)
_RE_KH_NO = re.compile(r'\bkh\.?\s*no\.?\s*[-–.]?\s*', re.IGNORECASE)

# A32 — Private Mailbox (PMB)
_RE_PMB = re.compile(r'\bPMB\s*\d+\b', re.IGNORECASE)

# A33 — parenthetical old address
_RE_PAREN_ADDR = re.compile(r'\([^)]*\)')

# A34 — mixed separators between numeric components "1-2-3" → "1 2 3"
_RE_MIXED_SEP = re.compile(r'(?<=\d)[-/](?=\d)')

# A22 — empty comma segments
_RE_EMPTY_SEGMENTS = re.compile(r',\s*,|^,|,$')


def _extract_numeric_tokens(text: str) -> frozenset:
    """Extract all purely numeric tokens from an address."""
    return frozenset(tok for tok in text.split() if tok.isdigit())


def normalise_address(
    raw_address: Optional[str],
    country: str,
) -> dict:
    """
    Full address normalisation pipeline covering A1–A37.

    Returns dict with keys:
      addr_normalized, addr_tokens, addr_numeric_tokens,
      addr_is_null, has_landmark, has_bank_atm_landmark
    """
    # A1 — NaN / None
    if raw_address is None or (isinstance(raw_address, float)):
        return {
            "addr_normalized": "",
            "addr_tokens": frozenset(),
            "addr_numeric_tokens": frozenset(),
            "addr_is_null": True,
            "has_landmark": False,
            "has_bank_atm_landmark": False,
        }

    text = str(raw_address).strip()

    # A1 — empty string
    if not text or text.lower() in ("nan", "none", "null", "na", "n/a"):
        return {
            "addr_normalized": "",
            "addr_tokens": frozenset(),
            "addr_numeric_tokens": frozenset(),
            "addr_is_null": True,
            "has_landmark": False,
            "has_bank_atm_landmark": False,
        }

    addr_is_null = False

    # N16 equivalent — HTML entities
    text = html.unescape(text)

    # N17 equivalent — Unicode Cf chars
    text = _strip_unicode_format_chars(text)

    # NFC
    text = _normalise_nfc(text)

    # N24 equivalent — Indic digits
    text = _apply_devanagari_digits(text)

    # A2 — literal NULL tokens
    text = _RE_NULL_TOKEN.sub("", text)

    # A22 — empty comma segments
    text = _RE_EMPTY_SEGMENTS.sub(",", text)
    segments = [s.strip() for s in text.split(",")]
    segments = [s for s in segments if s and s.lower() not in ("null", "none", "nan", "")]
    text = ", ".join(segments)

    # A5 — double-hash
    text = _RE_DOUBLE_HASH.sub("", text)

    # A7 — C/O prefix (must be before PO Box strip, case-insensitive on original case)
    text = re.sub(r'(?i)\bc/o\b\s*[^,]*,?\s*', "", text)

    # A6 — PO Box
    text = _RE_PO_BOX.sub("", text)

    # A32 — PMB
    text = _RE_PMB.sub("", text)

    # A23 — URL in address
    text = _RE_URL_IN_ADDR.sub("", text)

    # A24 — phone in address
    text = _RE_PHONE_IN_ADDR.sub("", text)

    # A33 — parenthetical old address
    text = _RE_PAREN_ADDR.sub("", text)

    # A3 — lowercase (S2 is ALL CAPS)
    text = text.lower()

    # A10 — ordinal suffixes on numbers "2nd" → "2"
    text = _RE_ORDINAL.sub(r"\1", text)

    # A9 — fractional address numbers "19 1/2" → "19"
    text = _RE_FRACTION.sub(r"\1", text)

    # A12 — floor variants → "floor N"
    def _floor_replace(m: re.Match) -> str:
        num = m.group(1) or m.group(2) or m.group(3) or "0"
        return f"floor {num}"
    text = _RE_FLOOR_VARIANTS.sub(_floor_replace, text)

    # A11 — unit variants → "unit "
    text = _RE_UNIT_VARIANTS.sub("unit ", text)

    # A13 — slash house numbers India "12/A" → "12a"
    text = _RE_SLASH_HOUSE.sub(r"\1\2", text)

    # A26 — road number "Road No.10" → "road 10"
    text = _RE_ROAD_NO.sub(r"road \1", text)

    # A28 — door no → "door N"
    text = _RE_DOOR_NO.sub(r"door \1", text)

    # A29 — house no → "hno N"
    text = _RE_HOUSE_NO.sub(r"hno \1", text)

    # A30 — KH No strip prefix
    text = _RE_KH_NO.sub("", text)

    # A34 — mixed separators "1-2-3" → "1 2 3"
    text = _RE_MIXED_SEP.sub(" ", text)

    # A8 / A25 — landmark detection (before further stripping)
    has_landmark = bool(_LANDMARK_PREFIXES.search(text))
    has_bank_atm_landmark = bool(_BANK_ATM_PATTERN.search(text))

    # Country-specific handling
    if country == "US":
        # A15 — US ZIP codes (5-digit and 5+4)
        text = re.sub(r'\b\d{5}(?:-\d{4})?\b', "", text)
        # A17 — expand US state abbreviations (token sort handles reorder)
        text = _expand_us_states(text)

    elif country == "India":
        # A14 — Indian PIN codes (6-digit)
        text = re.sub(r'\b\d{6}\b', "", text)
        # A20/A21 — Indic script in address → transliterate for TF-IDF
        text = _transliterate_indic_to_roman(text)
        # A27 — Survey/Gat numbers → keep numeric key
        text = _RE_SURVEY.sub(r"survey \1", text)

    elif country == "France":
        # A16 — French postal codes (5-digit)
        text = re.sub(r'\b\d{5}\b', "", text)
        # A18 — Cedex
        text = _RE_CEDEX.sub("", text)
        # A19 — arrondissement markers
        # Already lowercased; just normalise the suffix
        text = re.sub(r'\b(\d+)(?:e|er|ème|eme|ième|ieme)\b', r'\1', text)

    # A4 / A22 — clean up empty segments and trailing commas introduced by stripping
    # Split on commas, drop empty/whitespace-only segments, rejoin
    segments = [s.strip() for s in text.split(",")]
    segments = [s for s in segments if s]
    text = ", ".join(segments)
    text = _RE_WHITESPACE.sub(" ", text).strip()

    # Street abbreviation expansion
    for abbrev_pat, expansion in _STREET_ABBREV.items():
        text = re.sub(abbrev_pat, expansion, text, flags=re.IGNORECASE)

    # Final whitespace collapse
    addr_normalized = _RE_WHITESPACE.sub(" ", text).strip()

    # Token set — sorted unique non-empty tokens (A4 token sort representation)
    addr_tokens = frozenset(
        tok for tok in addr_normalized.replace(",", " ").split()
        if len(tok) >= 2  # skip single-char noise tokens
    )

    addr_numeric_tokens = _extract_numeric_tokens(
        addr_normalized.replace(",", " ")
    )

    return {
        "addr_normalized": addr_normalized,
        "addr_tokens": addr_tokens,
        "addr_numeric_tokens": addr_numeric_tokens,
        "addr_is_null": addr_is_null,
        "has_landmark": has_landmark,
        "has_bank_atm_landmark": has_bank_atm_landmark,
    }


# ---- US state abbreviation expansion helper --------------------------------

_US_STATES: dict[str, str] = {
    r'\bal\b': 'alabama', r'\bak\b': 'alaska', r'\baz\b': 'arizona',
    r'\bar\b': 'arkansas', r'\bca\b': 'california', r'\bco\b': 'colorado',
    r'\bct\b': 'connecticut', r'\bde\b': 'delaware', r'\bfl\b': 'florida',
    r'\bga\b': 'georgia', r'\bhi\b': 'hawaii', r'\bid\b': 'idaho',
    r'\bil\b': 'illinois', r'\bin\b': 'indiana', r'\bia\b': 'iowa',
    r'\bks\b': 'kansas', r'\bky\b': 'kentucky', r'\bla\b': 'louisiana',
    r'\bme\b': 'maine', r'\bmd\b': 'maryland', r'\bma\b': 'massachusetts',
    r'\bmi\b': 'michigan', r'\bmn\b': 'minnesota', r'\bms\b': 'mississippi',
    r'\bmo\b': 'missouri', r'\bmt\b': 'montana', r'\bne\b': 'nebraska',
    r'\bnv\b': 'nevada', r'\bnh\b': 'new hampshire', r'\bnj\b': 'new jersey',
    r'\bnm\b': 'new mexico', r'\bny\b': 'new york', r'\bnc\b': 'north carolina',
    r'\bnd\b': 'north dakota', r'\boh\b': 'ohio', r'\bok\b': 'oklahoma',
    r'\bor\b': 'oregon', r'\bpa\b': 'pennsylvania', r'\bri\b': 'rhode island',
    r'\bsc\b': 'south carolina', r'\bsd\b': 'south dakota', r'\btn\b': 'tennessee',
    r'\btx\b': 'texas', r'\but\b': 'utah', r'\bvt\b': 'vermont',
    r'\bva\b': 'virginia', r'\bwa\b': 'washington', r'\bwv\b': 'west virginia',
    r'\bwi\b': 'wisconsin', r'\bwy\b': 'wyoming', r'\bdc\b': 'district of columbia',
}


def _expand_us_states(text: str) -> str:
    for pat, expansion in _US_STATES.items():
        text = re.sub(pat, expansion, text, flags=re.IGNORECASE)
    return text


# ===========================================================================
# 6. MAIN ENTRY POINT
# ===========================================================================

def normalise_record(
    entity_id: str,
    business_name: Optional[str],
    business_address: Optional[str],
    country: Optional[str],
) -> NormalisedRecord:
    """
    Normalise a single raw record into a NormalisedRecord.

    This is the single function called per row during both the
    blocking phase and the feature engineering phase.

    Parameters
    ----------
    entity_id        : e.g. "S1-00001"
    business_name    : raw name string (may be None/NaN)
    business_address : raw address string (may be None/NaN)
    country          : raw country string (may be None/NaN)

    Returns
    -------
    NormalisedRecord dataclass with all derived fields.
    """
    source = entity_id.split("-")[0] if "-" in entity_id else "S?"

    # --- Country ---
    country_canonical = canonical_country(country)

    # --- Name ---
    raw_name = str(business_name) if business_name is not None else ""
    dominant_script = _detect_dominant_script(raw_name)
    name_fields = normalise_name(raw_name, country_canonical)

    # --- Address ---
    addr_fields = normalise_address(business_address, country_canonical)

    return NormalisedRecord(
        entity_id=entity_id,
        source=source,
        country_raw=str(country) if country is not None else "",
        country_canonical=country_canonical,
        name_raw=raw_name,
        name_normalized=name_fields["name_normalized"],
        name_core=name_fields["name_core"],
        name_token_set=name_fields["name_token_set"],
        name_trade=name_fields["name_trade"],
        name_legal=name_fields["name_legal"],
        is_domain=name_fields["is_domain"],
        base_domain=name_fields["base_domain"],
        tld=name_fields["tld"],
        tld_country_signal=name_fields["tld_country_signal"],
        addr_raw=str(business_address) if business_address is not None else "",
        addr_normalized=addr_fields["addr_normalized"],
        addr_tokens=addr_fields["addr_tokens"],
        addr_numeric_tokens=addr_fields["addr_numeric_tokens"],
        addr_is_null=addr_fields["addr_is_null"],
        has_landmark=addr_fields["has_landmark"],
        has_bank_atm_landmark=addr_fields["has_bank_atm_landmark"],
        dominant_script=dominant_script,
    )


# ===========================================================================
# 7. BATCH HELPER  (used by embedding + feature scripts)
# ===========================================================================

def normalise_dataframe(df) -> list[NormalisedRecord]:
    """
    Normalise an entire pandas DataFrame of source records.

    Expected columns: entity_id, business_name, business_address, country
    Returns a list of NormalisedRecord in the same row order.
    """
    records = []
    for row in df.itertuples(index=False):
        rec = normalise_record(
            entity_id=str(row.entity_id),
            business_name=getattr(row, "business_name", None),
            business_address=getattr(row, "business_address", None),
            country=getattr(row, "country", None),
        )
        records.append(rec)
    return records


def build_embed_text(rec: NormalisedRecord) -> str:
    """
    Construct the text string fed to BGE-m3 for embedding.

    Format: "Business: {name_normalized}  Address: {addr_normalized}"
    For domain-name records: use base_domain as the name component.
    For null addresses: name only.
    """
    name_part = rec.base_domain if rec.is_domain else rec.name_normalized
    if rec.addr_is_null or not rec.addr_normalized:
        return f"Business: {name_part}"
    return f"Business: {name_part}  Address: {rec.addr_normalized}"


# ===========================================================================
# 8. QUICK SMOKE TEST  (python normaliser.py)
# ===========================================================================

if __name__ == "__main__":
    _TEST_CASES = [
        # (entity_id, name, address, country)
        ("S1-00001", "FOUNDATION EXCEL AGENCY",
         "8219 17RD STREET, OH, Columbus", "US"),
        ("S2-00002", "Pvt Pvt Ltd [INCORPORATED]",
         "Near SBI ATM, Opp HDFC Bank, Sector 42", "India"),
        ("S3-00003", "Crestline Crestline Clean Corp",
         "##2326 341ND PLACE, NULL, KY", "USA"),
        ("S1-00004", "heassociates.com",
         "67 Rue de la Paix, Paris Cedex 13, 75001", "France"),
        ("S2-00005", "Sun पावर Provision Pvt Ltd",
         "Gat No 34, Door No 183, हरयाना", "india"),
        ("S3-00006", "Ectolumdrex dba X+ Madison Inc",
         "PO Box 6009, C/O Manoj Kumar, Cincinnati", "United States"),
        ("S1-00007", "A &amp; B Corp™",
         "19 1/2 STARDUST TRAIL, PMB 2592, TX", "US"),
        ("S2-00008", "*** Yamuica Venus Ltd",
         "12/A, 63/2275/7, Door No 20-6-3", "IN"),
        ("S1-00009", None, None, None),
        ("S3-00010", "குளோபல் பிசினஸ் பிரைவேட்",
         "Chennai, Tamil Nadu", "India"),
    ]

    print(f"{'Entity':<12} {'Country':<8} {'Name Normalized':<45} "
          f"{'Name Core':<35} {'Domain?':<8} {'Addr Null?'}")
    print("-" * 130)
    for args in _TEST_CASES:
        r = normalise_record(*args)
        print(
            f"{r.entity_id:<12} {r.country_canonical:<8} "
            f"{r.name_normalized[:43]:<45} "
            f"{r.name_core[:33]:<35} "
            f"{str(r.is_domain):<8} "
            f"{r.addr_is_null}"
        )
    print("\n--- Embed texts ---")
    for args in _TEST_CASES:
        r = normalise_record(*args)
        print(f"  {r.entity_id}: {build_embed_text(r)}")
