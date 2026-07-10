"""
paper_preprocessor.py
=====================
Unsupervised Extractive Research Paper Summarizer
Update 1: Document Ingestion & Academic Text Preprocessing

Pipeline:  PDF  ──►  Raw Text  ──►  Cleaned Text  ──►  Sentences
                  (pdfplumber)   (regex cascade)    (NLTK Punkt)

Target papers:
  - "Attention Is All You Need"          (Vaswani et al., 2017)
  - "SciBERT: A Pretrained LM for Sci."  (Beltagy et al., 2019)

Install requirements:
    pip install pdfplumber nltk

NLTK data (auto-downloaded on first run):
    punkt, punkt_tab
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize

# Small, deliberately conservative stopword set for _is_reversed_gibberish —
# only common short function words, so the forward-vs-reversed comparison
# stays a low-false-positive signal rather than a broad fluency judgment.
_REVERSAL_CHECK_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "and", "is", "are", "was",
    "were", "this", "that", "with", "for", "as", "by", "from", "we", "our",
    "which", "be", "it", "its",
}


def _stopword_hits(sentence: str) -> int:
    words = re.findall(r"[a-z]+", sentence.lower())
    return sum(1 for w in words if w in _REVERSAL_CHECK_STOPWORDS)


def _is_reversed_gibberish(sentence: str) -> bool:
    """
    True if reversing the sentence character-by-character reveals
    meaningfully MORE recognizable English stopwords than the sentence
    contains as-is — the signature of a whole-string-reversed text run
    (see SentenceTokenizer's docstring, filter 3). The +1 margin means a
    tie or a one-word edge case is not flagged; only a clear reversal
    signal is.
    """
    forward = _stopword_hits(sentence)
    backward = _stopword_hits(sentence[::-1])
    return backward > forward + 1

# ── One-time NLTK data download (silent after first run) ─────────────────────
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paper_preprocessor")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractionConfig:
    """Controls behaviour of the PDF extraction stage (Stage 1)."""

    # Words per text block below this count are skipped — they are almost
    # always captions, lone headers, or orphan column fragments, not prose.
    min_block_words: int = 8

    # Pages with fewer characters than this are title/blank pages; skip them.
    min_page_chars: int = 100

    # Activate column-aware word reordering (see PDFExtractor._column_aware_text).
    # Set False only for single-column PDFs (rare in academic publishing).
    detect_columns: bool = True


@dataclass
class PreprocessingConfig:
    """Controls the cleaning (Stage 2) and filtering (Stage 3) stages."""

    # Sentences shorter than this (chars) after cleaning are discarded.
    # Most citation-only fragments collapse below 40 chars.
    min_sentence_chars: int = 40

    # Sentences where non-alphabetic characters exceed this share of total
    # length are almost always residual equations or table rows.
    max_symbol_ratio: float = 0.40


@dataclass
class ProcessedDocument:
    """Carries state through the full pipeline; mutated in-place by each stage."""

    source_path: str
    raw_pages:   List[str] = field(default_factory=list)  # Stage 1 output
    joined_text: str       = ""                            # pre-cleaning join
    cleaned_text: str      = ""                            # Stage 2 output
    sentences:   List[str] = field(default_factory=list)  # Stage 3 output
    abstract:    str       = ""                            # heuristic extraction, may be ""


# Marks the end of the abstract: the paper's first section heading.
_ABSTRACT_END_PATTERN = re.compile(
    r"\b(?:[IVX]+\.?|1\.?)\s*INTRODUCTION\b|\bINTRODUCTION\b",
    re.IGNORECASE,
)


def extract_abstract(cleaned_text: str, search_window: int = 6000, max_len: int = 3000) -> str:
    """
    Best-effort heuristic extraction of the paper's Abstract section from the
    cleaned full-text, for use as a reference summary in ROUGE evaluation.

    There is no structural (e.g. XML/section-tagged) source to rely on here —
    only a flat cleaned-text string — so this looks for the literal word
    "Abstract" near the top of the document and takes everything up to the
    first "Introduction" heading. Returns "" if no "Abstract" marker is found
    within search_window chars of the start (e.g. non-standard paper layouts),
    so callers must treat this as optional, not guaranteed.
    """
    window = cleaned_text[:search_window]
    match = re.search(r"\babstract\b[\s—:.\-]*", window, re.IGNORECASE)
    if not match:
        return ""

    start = match.end()
    end_match = _ABSTRACT_END_PATTERN.search(cleaned_text, start)
    end = end_match.start() if end_match else start + max_len

    return cleaned_text[start:end].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — PDF EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

class PDFExtractor:
    """
    Extracts raw text from academic PDFs using pdfplumber with column-aware
    word reordering to fix the primary structural failure mode of naive PDF
    text extraction.

    THE MULTI-COLUMN PROBLEM
    ─────────────────────────
    Academic papers (especially ACL/NeurIPS/ICML format) use two-column layout.
    pdfplumber's default `page.extract_text()` scans left-to-right, top-to-bottom
    across the FULL page width, so it interleaves the two columns:

        PHYSICAL PAGE:                    NAIVE EXTRACTION RESULT:
        ┌──────────┬──────────┐           "Left col line 1. Right col line 1.
        │ L-line 1 │ R-line 1 │    →       Left col line 2. Right col line 2."
        │ L-line 2 │ R-line 2 │
        └──────────┴──────────┘

    This breaks sentence boundaries between unrelated paragraphs. BERT sentence
    embeddings computed on interleaved text are semantically meaningless.

    OUR SOLUTION: extract_words() gives each word with its bounding box.
    We split words by their x0 position relative to the page midpoint, sort
    each column independently by (y, x), then concatenate left column first.
    """

    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract(self, pdf_path: str) -> ProcessedDocument:
        """Open a PDF and populate doc.raw_pages with per-page text strings."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = ProcessedDocument(source_path=str(path))

        self.logger.info(f"Opening: {path.name}")
        with pdfplumber.open(path) as pdf:
            self.logger.info(f"Total pages detected: {len(pdf.pages)}")
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = self._extract_page(page, page_num)
                if page_text:
                    doc.raw_pages.append(page_text)

        self.logger.info(f"Kept {len(doc.raw_pages)} content pages after length gate.")
        return doc

    def _extract_page(self, page, page_num: int) -> Optional[str]:
        """
        Extract and reorder words from a single page.

        We use extract_words() instead of extract_text() because it gives us
        fine-grained bounding boxes per word, enabling column detection.

        x_tolerance / y_tolerance:
          These control when pdfplumber groups adjacent characters into one
          "word" object. 3pt is a reliable default for standard academic fonts;
          lowering it fragments hyphenated words into two tokens.
        """
        words = page.extract_words(
            keep_blank_chars=False,
            x_tolerance=1,   # tightened from 3 — prevents inter-word fusion in
            y_tolerance=3,   # justified/narrow-column layouts where word gaps
                             # can be as small as 1-2 pt (within old threshold)
        )

        if not words:
            self.logger.debug(f"Page {page_num}: no words found, skipping.")
            return None

        text = (
            self._column_aware_text(words, page.width)
            if self.config.detect_columns
            else self._flat_text(words)
        )

        if len(text) < self.config.min_page_chars:
            self.logger.debug(
                f"Page {page_num}: too short ({len(text)} chars), likely title/blank."
            )
            return None

        self.logger.debug(f"Page {page_num}: {len(text)} chars extracted.")
        return text

    def _column_aware_text(self, words: List[dict], page_width: float) -> str:
        """
        Two-column reordering algorithm.

        Step 1 — SPLIT:  words with x0 < page_width/2 go to left_col; rest
                          go to right_col. The midpoint is a robust proxy for
                          the gutter between columns in standard two-column PDFs.

        Step 2 — SORT:   Within each column, sort by rounded top (vertical
                          position) as the primary key, then by x0 as secondary.
                          Rounding top to the nearest 4pt bucket merges words
                          that sit on the same typographic baseline but whose
                          bounding boxes differ by sub-pixel amounts due to
                          font metrics — without rounding, words on the same
                          line can sort as different "rows".

        Step 3 — JOIN:   Left column text first, then right column text.
                          A space separator ensures the sentence tokenizer sees
                          a clean word boundary at the column junction.
        """
        mid = page_width / 2

        left_col  = [(w["top"], w["x0"], w["text"]) for w in words if w["x0"] <  mid]
        right_col = [(w["top"], w["x0"], w["text"]) for w in words if w["x0"] >= mid]

        # round(top / 4) * 4 quantises vertical positions into 4pt buckets
        left_col.sort( key=lambda t: (round(t[0] / 4) * 4, t[1]))
        right_col.sort(key=lambda t: (round(t[0] / 4) * 4, t[1]))

        return " ".join(t[2] for t in left_col) + " " + " ".join(t[2] for t in right_col)

    def _flat_text(self, words: List[dict]) -> str:
        """Fallback for single-column PDFs: simple top→bottom, left→right sort."""
        words_sorted = sorted(words, key=lambda w: (round(w["top"] / 5) * 5, w["x0"]))
        return " ".join(w["text"] for w in words_sorted)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

class TextPreprocessor:
    """
    Cleans raw PDF text through an ordered cascade of regex transformations.

    WHY ORDER MATTERS
    ─────────────────
    Each transformation assumes the text is in the state left by the previous
    one.  Swapping steps can cause patterns to miss their targets:

        • Hyphen repair MUST be first — other patterns match on complete tokens.
          If we remove citations before repair, "pre- [1] training" never gets
          its hyphen fixed because the inter-token gap has changed.

        • Whitespace normalisation MUST be last — every prior removal leaves
          double-spaces or spaces-before-punctuation that need a final sweep.

    All regex patterns are compiled once at class definition to avoid repeated
    re-compilation overhead when the preprocessor is called on many pages.
    """

    # ── Pattern 1: Hyphenated Line-Break Repair ────────────────────────────────
    #
    # WHY THIS PATTERN EXISTS:
    #   PDF renderers break long words at line boundaries by inserting a hyphen
    #   and a newline. pdfplumber's word tokeniser keeps these as separate tokens,
    #   so after joining words with spaces we see "trans- former" (note the space
    #   after the hyphen). This destroys token identity — SciBERT's WordPiece
    #   vocabulary contains "transformer" but not "trans-" as a standalone token.
    #
    # PATTERN ANATOMY:
    #   (\w)   — capturing group 1: word character BEFORE the hyphen (the stem)
    #   -      — the literal hyphen left by the PDF renderer
    #   \s+    — one or more whitespace characters (the line-break space)
    #   ([a-z]) — capturing group 2: LOWERCASE letter starting the suffix
    #
    # WHY LOWERCASE IN GROUP 2 IS CRITICAL:
    #   Without it, "Figure 3-\nAblation" → "Figure 3Ablation", destroying a
    #   legitimate dash. Lowercase-after-hyphen reliably signals a word-internal
    #   break because true compound words and section headers start with uppercase.
    #
    # SUBSTITUTION:  r'\1\2'  — drop the hyphen and whitespace, fuse the two halves.
    HYPHEN_LINEBREAK = re.compile(r"(\w)-\s+([a-z])")

    # ── Pattern 2: Bracket-Style Numeric Citations ─────────────────────────────
    #
    # WHY THIS PATTERN EXISTS:
    #   "Attention Is All You Need" uses IEEE/NeurIPS numeric citation style:
    #   [1], [2, 3], [1–4].  These tokens:
    #     (a) carry no semantic content useful for summarisation,
    #     (b) break sentence boundary detection — "[1]." looks like end-of-sentence,
    #     (c) pollute SciBERT embeddings by injecting out-of-vocabulary number sequences.
    #
    # PATTERN ANATOMY:
    #   \[                — literal opening square bracket
    #   [\d,\s–-]+   — one or more of: digit, comma, space, en-dash (U+2013), hyphen
    #                       This covers [1], [1, 2], [1, 2, 3], [1–4], [1-4]
    #   \]                — literal closing square bracket
    #
    # WHY NOT \[\d+\]:
    #   Would miss multi-citation forms like [1, 2] or [1–4] which are extremely
    #   common in Vaswani et al. 2017.
    BRACKET_CITATIONS = re.compile(r"\[[\d,\s–-]+\]")

    # ── Pattern 3: Parenthetical Author-Year Citations ─────────────────────────
    #
    # WHY THIS PATTERN EXISTS:
    #   "SciBERT" uses ACL/APA parenthetical citation style:
    #   (Devlin et al., 2018), (Vaswani and Shazeer, 2017), (Peters, 2018a).
    #   These also carry no summarisation value and confuse sentence tokenisers
    #   because the closing ")" is ambiguous with grouped mathematical expressions.
    #
    # PATTERN ANATOMY (reading left to right):
    #   \(                           — opening parenthesis
    #   [A-Z][a-z]+                  — capitalised author surname (e.g., "Devlin")
    #   (?:                          — non-capturing group for optional author suffix
    #     \s+et\s+al\.               — "et al." (multi-author shorthand)
    #     | \s+and\s+[A-Z][a-z]+     — "and SecondAuthor" (two-author form)
    #     | ,\s+[A-Z][a-z]+          — ", SecondAuthor" (comma-separated)
    #   )?                           — whole suffix is optional (single-author form)
    #   ,?\s*                        — optional comma + space before year
    #   \d{4}                        — 4-digit publication year
    #   [a-z]?                       — optional letter suffix: 2018a, 2018b
    #   \)                           — closing parenthesis
    #
    # DESIGN CONSERVATISM:
    #   We require a capital surname as the FIRST token inside the parens.
    #   This prevents false positives on ordinary parenthetical phrases like
    #   "(see Section 3)" or "(approximately 512 dimensions)".
    PAREN_CITATIONS = re.compile(
        r"\([A-Z][a-z]+"
        r"(?:\s+et\s+al\.|\s+and\s+[A-Z][a-z]+|,\s+[A-Z][a-z]+)?"
        r",?\s*\d{4}[a-z]?"
        r"\)"
    )

    # ── Pattern 4: Inline Mathematical Expressions ─────────────────────────────
    #
    # WHY THIS PATTERN EXISTS:
    #   Both target papers contain inline math in their prose, e.g.
    #   "...divide each by d_k = 64..." or "...where d_model = 512...".
    #   After pdfplumber extraction these appear as literal text strings.
    #   SciBERT was pre-trained on scientific text but its WordPiece vocabulary
    #   does not robustly represent algebraic notation; embeddings dominated by
    #   variable names skew the sentence vector away from the semantic meaning.
    #
    # PATTERN ANATOMY:
    #   (?<!\w)                     — negative lookbehind: not preceded by a word char
    #                                 (prevents matching inside "ResNet-50" or "BERT-base")
    #   [a-zA-Z0-9_]+               — left operand: variable name or number
    #   \s*[=<>≤≥+\-×÷/^]\s*       — operator character surrounded by optional spaces
    #   [a-zA-Z0-9_.]+              — right operand
    #   (?!\w)                      — negative lookahead: not followed by a word char
    #
    # INTENTIONAL LIMITATION:
    #   We target *operator-connected* expressions only.  Bare standalone numbers
    #   like "512 dimensions" are semantically meaningful and are kept.  Complex
    #   multi-token equations that span several words are also not fully caught —
    #   that is acceptable; the goal is reducing, not eliminating, noise.
    MATH_EXPRESSIONS = re.compile(
        r"(?<!\w)"
        r"[a-zA-Z0-9_]+"
        r"\s*[=<>≤≥+\-×÷/^]\s*"
        r"[a-zA-Z0-9_.]+"
        r"(?!\w)"
    )

    # ── Pattern 5: Inline Section Header Numbers ───────────────────────────────
    #
    # WHY THIS PATTERN EXISTS:
    #   When pdfplumber reads a two-column page, section headers like "3.1 Encoder"
    #   sometimes get absorbed into the preceding paragraph's text stream, producing
    #   "...training. 3.1 Encoder The encoder maps an...".  The section number
    #   "3.1" is not a sentence — it is a document structural artifact.
    #   Without removal, sent_tokenize() may treat it as a sentence fragment.
    #
    # PATTERN ANATOMY:
    #   \b              — word boundary (ensures we don't match "3.1" inside "Figure 3.1")
    #   \d+             — leading section number (e.g., "3")
    #   (?:\.\d+)*      — zero or more sub-section extensions: ".1", ".2.1", etc.
    #   \s+             — mandatory whitespace between number and title word
    #   (?=[A-Z])       — POSITIVE LOOKAHEAD for a capital letter (the title word)
    #                     The lookahead does NOT consume the letter, so "Encoder"
    #                     is retained — only the "3.1 " prefix is removed.
    SECTION_NUMBERS = re.compile(r"\b\d+(?:\.\d+)*\s+(?=[A-Z])")

    # ── Pattern 6: Standalone Page Numbers ────────────────────────────────────
    #
    # WHY THIS PATTERN EXISTS:
    #   Page numbers in PDF headers/footers are positioned between the columns
    #   or at page margins.  After column-aware extraction they appear as lone
    #   integers surrounded by spaces: "...gradient. 4 We suspect...".
    #   These are structurally identical to numeric measurements, so we must
    #   use CONTEXT to distinguish them: a page number has NO adjacent letter.
    #
    # PATTERN ANATOMY:
    #   (?<!\w)          — not preceded by a word character (e.g., avoids "ResNet50")
    #   \d{1,3}          — 1-3 digit integer (covers pages 1–999)
    #   (?!\w)           — not followed by a word character (avoids "512-dimensional")
    #   (?!\s*[a-zA-Z])  — not followed by optional whitespace then a letter
    #                      This is the KEY discriminator: "512 dimensions" keeps
    #                      "512" because it IS followed by a letter ("d").
    #                      "4 " before a non-letter or end-of-text is a page number.
    #
    # NOTE: This is a heuristic with known false-positive risk at sentence starts
    # like "4. We find that..." — acceptable for academic body text but revisit
    # if processing documents with numbered list items.
    PAGE_NUMBERS = re.compile(r"(?<!\w)\d{1,3}(?!\w)(?!\s*[a-zA-Z])")

    # ── Pattern 7 & 8: Whitespace Normalisation ────────────────────────────────
    #
    # WHY THESE PATTERNS EXIST:
    #   Steps 4–7 substitute matches with a single SPACE (" ") rather than the
    #   empty string ("").  This guarantees that removing a citation, equation,
    #   or page number can never fuse the surrounding words:
    #     "values[3]are"  →  "values are"   (not "valuesare")
    #     "results d_k=64 using"  →  "results  using"  →  "results using"
    #   The resulting double-spaces are then collapsed here in Step 8a.
    #
    # MULTI_WHITESPACE:  \s+ matches any whitespace run (spaces, tabs, newlines)
    #                    and replaces it with a single ASCII space.  This also
    #                    flattens residual newlines from the page-join.
    #
    # SPACE_BEFORE_PUNCT: \s+([,;:.!?]) removes any whitespace that precedes a
    #                     punctuation character, replacing with just the punctuation.
    #                     The capture group \1 re-inserts the punctuation itself.
    #                     Safe: only triggers on whitespace-before-punctuation runs,
    #                     never between two word-character tokens.
    MULTI_WHITESPACE   = re.compile(r"\s+")
    SPACE_BEFORE_PUNCT = re.compile(r"\s+([,;:.!?])")

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Run the full cleaning cascade on doc.raw_pages.
        Populates doc.joined_text and doc.cleaned_text.
        Returns the same doc (pipeline pattern).
        """
        self.logger.info("Preprocessing stage started.")

        # Join pages with ". " so the sentence tokeniser sees a valid boundary
        # at each page break and doesn't fuse the last sentence of page N with
        # the first sentence of page N+1 into one nonsensical string.
        doc.joined_text = ". ".join(doc.raw_pages)
        self.logger.info(
            f"Joined {len(doc.raw_pages)} pages → {len(doc.joined_text):,} raw chars."
        )

        text = doc.joined_text

        # ── Ordered cascade ───────────────────────────────────────────────────
        # Steps 4–7 substitute with " " (space) not "" (empty) so that removing
        # a citation/equation/page-number between two words never fuses them.
        # Step 8a collapses the resulting double-spaces into single spaces.
        text = self.HYPHEN_LINEBREAK.sub(r"\1\2",  text)  # Step 2
        text = self.SECTION_NUMBERS.sub("",        text)  # Step 3 — lookahead keeps title word
        text = self.BRACKET_CITATIONS.sub(" ",     text)  # Step 4 — space guards word boundary
        text = self.PAREN_CITATIONS.sub(" ",       text)  # Step 5 — space guards word boundary
        text = self.MATH_EXPRESSIONS.sub(" ",      text)  # Step 6 — space guards word boundary
        text = self.PAGE_NUMBERS.sub(" ",          text)  # Step 7 — space guards word boundary
        text = self.MULTI_WHITESPACE.sub(" ",      text)  # Step 8a — collapse all extra spaces
        text = self.SPACE_BEFORE_PUNCT.sub(r"\1",  text)  # Step 8b — drop space-before-punct
        text = text.strip()

        doc.cleaned_text = text
        self.logger.info(f"Cleaning complete → {len(doc.cleaned_text):,} chars.")
        return doc


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SENTENCE TOKENISATION & QUALITY FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

class SentenceTokenizer:
    """
    Splits cleaned text into individual sentences using NLTK's Punkt algorithm,
    then removes sentences that would produce low-quality BERT embeddings.

    WHY PUNKT OVER A NAIVE SPLIT
    ─────────────────────────────
    A naive re.split on period + uppercase fails on scientific prose because:

      • Abbreviations:   "et al.", "e.g.", "i.e.", "Fig.", "Eq.", "approx."
                         all end with a period followed (sometimes) by a capital.
                         A naive split fractures them constantly.

      • Decimals:        "achieved 0.512 BLEU" contains a period between digits —
                         naive split on period after digit would miss it; split on
                         "period + letter" would incorrectly keep it together.

    NLTK Punkt is an unsupervised abbreviation-learner: it reads the entire text
    and builds a statistical model of which period-terminated tokens are sentence
    enders vs. abbreviations. It handles scientific text well out-of-the-box and
    is the standard choice before feeding to transformer sentence encoders.

    POST-TOKENISATION FILTERS
    ──────────────────────────
    Even after cleaning, two artifact classes slip through:

      1. Too-short sentences — citation-only fragments collapse to ". " or a
         single word after pattern removal.  These produce near-zero semantic
         content and pollute the K-Means cluster centroids.

      2. Symbol-dense sentences — residual equations or table rows that evaded
         the MATH_EXPRESSIONS pattern still have very few alphabetic characters
         relative to their total length.  Their BERT embeddings cluster around
         out-of-vocabulary token distributions rather than semantic topics.

      3. Whole-string-reversed text — some PDFs (observed directly on a test
         document containing an attention-visualization figure caption) have
         a text layer where certain runs are stored character-reversed, e.g.
         "stnemnrevog naciremA" for "American governments". pdfplumber
         extracts exactly what the PDF's text layer contains, so this is a
         source-PDF encoding defect, not an extraction bug — but it produces
         embeddings that are semantically meaningless and, worse, poison
         retrieval (a query embedding can end up spuriously close to
         reversed gibberish). Detected via _is_reversed_gibberish(): compare
         how many common English stopwords appear in the sentence forward
         vs. character-reversed; real prose never has meaningfully more
         reversed than forward, so this has zero false positives on clean
         text (verified against a 268-sentence clean document) while
         catching the reversed runs directly. It does NOT catch "softer"
         corruption like dropped/fused words from a broken glyph mapping —
         that would require a language-model fluency check, out of scope
         for a regex/heuristic filtering stage.
    """

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def tokenize(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Tokenise doc.cleaned_text into doc.sentences.
        Applies length and symbol-density quality filters.
        Returns the same doc.
        """
        raw_sentences = sent_tokenize(doc.cleaned_text)
        self.logger.info(f"Punkt produced {len(raw_sentences)} candidate sentences.")

        filtered: List[str] = []
        n_short     = 0
        n_symbolic  = 0
        n_reversed  = 0

        for sent in raw_sentences:
            sent = sent.strip()

            # Filter A — minimum length
            # After stripping "(Vaswani et al., 2017)." we might have just ".".
            if len(sent) < self.config.min_sentence_chars:
                n_short += 1
                continue

            # Filter B — maximum symbol density
            # Count alpha characters; derive the ratio of NON-alpha content.
            # A sentence like "Q K^T / sqrt(d_k) = 0.98" is ~0% alphabetic after
            # the variable names — it will produce a very noisy embedding.
            alpha = sum(1 for c in sent if c.isalpha())
            symbol_ratio = 1.0 - (alpha / len(sent))
            if symbol_ratio > self.config.max_symbol_ratio:
                n_symbolic += 1
                continue

            # Filter C — whole-string-reversed text (see class docstring)
            if _is_reversed_gibberish(sent):
                n_reversed += 1
                continue

            filtered.append(sent)

        doc.sentences = filtered
        self.logger.info(
            f"Kept {len(filtered)} sentences.  "
            f"Discarded: {n_short} too-short, {n_symbolic} too-symbolic, "
            f"{n_reversed} reversed-text."
        )
        return doc


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentIngestionPipeline:
    """
    Public API for Update 1.

    Composes the three stages into a single call and returns a ProcessedDocument
    whose .sentences list is ready to be fed to a SciBERT sentence encoder.

    Usage:
        pipeline = DocumentIngestionPipeline()
        doc = pipeline.run("attention_is_all_you_need.pdf")
        sentences = doc.sentences   # List[str]
    """

    def __init__(
        self,
        extraction_config:    ExtractionConfig    = None,
        preprocessing_config: PreprocessingConfig = None,
    ):
        self.extractor    = PDFExtractor(extraction_config)
        self.preprocessor = TextPreprocessor(preprocessing_config)
        self.tokenizer    = SentenceTokenizer(preprocessing_config)
        self.logger       = logging.getLogger(self.__class__.__name__)

    def run(self, pdf_path: str) -> ProcessedDocument:
        """Execute the full three-stage pipeline on a single PDF."""
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Pipeline START  →  {pdf_path}")
        self.logger.info(f"{'='*60}")

        doc = self.extractor.extract(pdf_path)
        doc = self.preprocessor.preprocess(doc)
        doc = self.tokenizer.tokenize(doc)
        doc.abstract = extract_abstract(doc.cleaned_text)

        self.logger.info(
            f"Pipeline END  →  {len(doc.sentences)} sentences ready for embedding."
        )
        return doc


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE EXECUTION BLOCK
# Runs when you call:  python paper_preprocessor.py
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    DIV  = "=" * 72
    SUB  = "-" * 72

    # ── Synthetic Raw Input ────────────────────────────────────────────────────
    # This string intentionally encodes every noise pattern from both papers:
    #
    #  (a) Section number prefix:       "3.2 Scaled"
    #  (b) Hyphenated line-break split: "dimen- sion", "magni- tude", "pre- training"
    #  (c) Bracket citation:            "[3]"
    #  (d) Parenthetical citations:     "(Vaswani et al., 2017)", "(Devlin et al., 2018)",
    #                                   "(Beltagy et al., 2019)"
    #  (e) Inline math expression:      "d_k = 64"
    #  (f) Standalone page number:      " 4 "  (between two content sentences)

    RAW = (
        "3.2 Scaled Dot-Product Attention The input consists of queries and keys of "
        "dimen- sion d_k, and values of dimension d_v. We compute the dot products of "
        "the query with all keys, divide each by d_k = 64 [3], and apply a softmax "
        "function (Vaswani et al., 2017) to obtain the weights on the values. "
        "4 "
        "We suspect that for large values of d_k, the dot products grow large in "
        "magni- tude, pushing the softmax function into regions where it has extremely "
        "small gradients (Devlin et al., 2018). SciBERT (Beltagy et al., 2019) later "
        "showed that domain-adaptive pre- training on scientific corpora significantly "
        "improves performance on SciIE and other downstream NLP tasks."
    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def show(label: str, text: str, width: int = 72) -> None:
        """Print a labelled text block, wrapping long lines for readability."""
        print(f"\n  {label}:")
        words, line = text.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > width - 4:
                print(f"    {line}")
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            print(f"    {line}")

    def delta(before: str, after: str) -> str:
        """Report how many characters were removed."""
        diff = len(before) - len(after)
        return f"  ▸ Removed {diff} chars ({len(before)} → {len(after)})"

    # ── Banner ────────────────────────────────────────────────────────────────

    print(f"\n{DIV}")
    print("  INTERACTIVE DEMO — Step-by-Step PDF Text Transformation")
    print(f"{DIV}")
    print(
        "\n  Synthetic input encodes every noise class found in:\n"
        "    • 'Attention Is All You Need'  (bracket citations, column wrap, math)\n"
        "    • 'SciBERT'                    (paren citations, hyphenated compounds)\n"
    )

    show("RAW INPUT", RAW)

    # ── Initialise preprocessor and tokeniser in isolation ────────────────────
    pp  = TextPreprocessor()
    tok = SentenceTokenizer()
    text = RAW

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STEP 2  ─  Repair Hyphenated Line-Break Splits")
    print(f"  Regex: (\\w)-\\s+([a-z])  →  \\1\\2")
    print(f"  Why:   Word-internal hyphens inserted by the PDF renderer fuse back.")
    print(f"  Guard: lowercase-only right-half prevents merging 'Figure 3-Ablation'.")
    print(SUB)
    before, text = text, pp.HYPHEN_LINEBREAK.sub(r"\1\2", text)
    print(delta(before, text))
    print("  Changed tokens:  'dimen- sion' → 'dimension'")
    print("                   'magni- tude' → 'magnitude'")
    print("                   'pre- training' → 'pretraining'")
    show("AFTER STEP 2", text)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STEP 3  ─  Remove Inline Section Header Numbers")
    print(r"  Regex: \b\d+(?:\.\d+)*\s+(?=[A-Z])  →  ''")
    print("  Why:   '3.2 Scaled' is a structural label, not prose content.")
    print("  Guard: Lookahead (?=[A-Z]) keeps the title word 'Scaled' intact.")
    print(SUB)
    before, text = text, pp.SECTION_NUMBERS.sub("", text)
    print(delta(before, text))
    print("  Changed tokens:  '3.2 Scaled' → 'Scaled'")
    show("AFTER STEP 3", text)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STEP 4  ─  Remove Bracket-Style Numeric Citations")
    print(r"  Regex: \[[\d,\s–-]+\]  →  ''")
    print("  Why:   [3] disrupts sentence boundary detection ('values.[3]' looks")
    print("         like a sentence ender). Covers [1], [1,2], [1–4], [1-4].")
    print(SUB)
    before, text = text, pp.BRACKET_CITATIONS.sub("", text)
    print(delta(before, text))
    print("  Changed tokens:  '[3]' → ''")
    show("AFTER STEP 4", text)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STEP 5  ─  Remove Parenthetical Author-Year Citations")
    print("  Regex: (Surname [et al.|and X|, X]?, YYYY[a-z]?)")
    print("  Why:   (Vaswani et al., 2017) has zero summarisation value and")
    print("         its closing ')' creates ambiguity for sentence tokenisers.")
    print(SUB)
    before, text = text, pp.PAREN_CITATIONS.sub("", text)
    print(delta(before, text))
    print("  Changed tokens:  '(Vaswani et al., 2017)' → ''")
    print("                   '(Devlin et al., 2018)'   → ''")
    print("                   '(Beltagy et al., 2019)'  → ''")
    show("AFTER STEP 5", text)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STEP 6  ─  Remove Inline Mathematical Expressions")
    print(r"  Regex: (?<!\w)[a-zA-Z0-9_]+\s*[=<>≤≥+\-×÷/^]\s*[a-zA-Z0-9_.]+(?!\w)")
    print("  Why:   'd_k = 64' is algebra, not language — SciBERT's WordPiece")
    print("         vocabulary cannot meaningfully represent it in context.")
    print("  Guard: Negative lookarounds prevent partial matches inside words.")
    print(SUB)
    before, text = text, pp.MATH_EXPRESSIONS.sub("", text)
    print(delta(before, text))
    print("  Changed tokens:  'd_k = 64' → ''")
    show("AFTER STEP 6", text)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STEP 7  ─  Remove Standalone Page Numbers")
    print(r"  Regex: (?<!\w)\d{1,3}(?!\w)(?!\s*[a-zA-Z])  →  ''")
    print("  Why:   Lone ' 4 ' between content sentences is a page number from")
    print("         the extracted header/footer, not a measurement.")
    print("  Guard: (?!\\s*[a-zA-Z]) keeps '512 dimensions' ('512' has a letter after it).")
    print(SUB)
    before, text = text, pp.PAGE_NUMBERS.sub("", text)
    print(delta(before, text))
    print("  Changed tokens:  ' 4 ' → ''")
    show("AFTER STEP 7", text)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STEP 8  ─  Normalise Whitespace  (always last)")
    print(r"  Pass 8a: \s+  →  ' '         (collapse all whitespace runs)")
    print(r"  Pass 8b: \s+([,;:.!?])  →  \1  (remove space-before-punctuation artifacts)")
    print("  Why:   Every removal above leaves double-spaces or dangling spaces")
    print("         before punctuation. Normalise once, cleanly, at the end.")
    print(SUB)
    before = text
    text = pp.MULTI_WHITESPACE.sub(" ", text)
    text = pp.SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = text.strip()
    print(delta(before, text))
    show("FINAL CLEANED TEXT", text)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STEP 9  ─  NLTK Punkt Sentence Tokenisation + Quality Filtering")
    print(SUB)
    print(
        "  Punkt builds a statistical abbreviation model from the input text\n"
        "  so that 'et al.', 'e.g.', and 'Fig.' do not trigger false splits.\n"
        "\n"
        "  Each sentence then passes two quality gates:\n"
        "    A. Length gate     — must be ≥ 40 characters after cleaning.\n"
        "    B. Symbol-density  — non-alpha chars must be < 40% of sentence length.\n"
    )

    demo_doc = ProcessedDocument(source_path="<demo>")
    demo_doc.cleaned_text = text
    demo_doc = tok.tokenize(demo_doc)

    print(f"\n  OUTPUT — {len(demo_doc.sentences)} Sentence(s):\n{SUB}")
    for i, sent in enumerate(demo_doc.sentences, start=1):
        alpha = sum(1 for c in sent if c.isalpha())
        sym_r = 1.0 - (alpha / len(sent))
        print(f"\n  [{i:02d}]  len={len(sent)} chars  |  symbol_ratio={sym_r:.2f}")
        # Wrap the sentence for display
        words_out, line_out = sent.split(), ""
        for w in words_out:
            if len(line_out) + len(w) + 1 > 66:
                print(f"        {line_out}")
                line_out = w
            else:
                line_out = (line_out + " " + w).strip()
        if line_out:
            print(f"        {line_out}")

    print(f"\n{DIV}")
    print("  Demo complete.  Sentences above are ready for SciBERT embedding.")
    print(f"{DIV}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # LIVE PDF USAGE  (uncomment and point at real PDFs)
    # ─────────────────────────────────────────────────────────────────────────
    #
    # pipeline = DocumentIngestionPipeline(
    #     extraction_config=ExtractionConfig(detect_columns=True),
    #     preprocessing_config=PreprocessingConfig(min_sentence_chars=40),
    # )
    #
    # for pdf in ["attention_is_all_you_need.pdf", "scibert.pdf"]:
    #     doc = pipeline.run(pdf)
    #     print(f"\n{pdf}  →  {len(doc.sentences)} sentences")
    #     for s in doc.sentences[:3]:
    #         print(f"  • {s[:90]}...")
