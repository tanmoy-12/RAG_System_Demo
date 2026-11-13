from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retriever import RetrievedChunk

# Accept the required form `[Paper, p. 12]` and harmless typography variants
# such as `[Paper, p.12]` or `[Paper, page 12]`. The paper label and page must
# still match a retrieved chunk exactly (case/whitespace-insensitively).
CITATION_RE = re.compile(
    r"\[\s*(?P<paper>[^,\[\]]+?)\s*,?\s*(?:p(?:age)?|pp?)\.?\s*(?P<page>\d+)\s*\]",
    re.IGNORECASE,
)
CITATION_LIKE_RE = re.compile(
    r"\[[^\]]*(?:\bp{1,2}\.?\s*\d+|\bpage\s*\d+)[^\]]*\]",
    re.IGNORECASE,
)

# This deliberately recognises a refusal statement, not any sentence that
# happens to contain words such as "does not" or "not available".
REFUSAL_RE = re.compile(
    r"(?:"
    r"\b(?:i|we)\s+(?:cannot|can't)\s+(?:reliably\s+)?(?:answer|determine|provide)\b"
    r"|\b(?:the\s+)?(?:provided\s+)?(?:papers|sources|retrieved\s+(?:passages|excerpts)|available\s+evidence)\s+"
    r"(?:do\s+not|does\s+not|don't|doesn't)\s+(?:state|specify|establish|cover|contain|provide)\b"
    r"|\b(?:not\s+(?:explicitly\s+)?(?:stated|specified|covered|established)\s+in\s+(?:the\s+)?(?:paper|papers|provided\s+sources|retrieved\s+excerpts))\b"
    r"|\b(?:the\s+)?(?:evidence|retrieved\s+evidence)\s+is\s+insufficient\b"
    r"|\binsufficient\s+(?:evidence|information)\s+(?:to\s+(?:answer|determine|establish))\b"
    r")",
    re.IGNORECASE,
)

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


@dataclass
class CitationCheck:
    valid: bool
    found: list[str]
    invalid: list[str]
    reason: str
    uncited_blocks: list[str] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)


def _normalise_label(value: str) -> str:
    return " ".join(value.casefold().split())


def _answer_blocks(answer: str) -> list[str]:
    """Return prose paragraphs and individual list items as validation blocks.

    Newlines inside ordinary prose are treated as visual wrapping, not as new
    paragraphs. Markdown list items remain separate factual blocks.
    """
    blocks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", answer.strip()):
        current: list[str] = []
        for raw_line in paragraph.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if HEADING_RE.match(line):
                if current:
                    blocks.append(" ".join(current))
                    current = []
                # Preserve the heading as a boundary marker. The validator
                # later treats it as a section label, not a factual claim.
                blocks.append(line)
                continue
            if LIST_ITEM_RE.match(line):
                if current:
                    blocks.append(" ".join(current))
                current = [line]
            else:
                # Join hard-wrapped lines and continuation lines to their
                # paragraph/list item, so a citation at the end still counts.
                current.append(line)
        if current:
            blocks.append(" ".join(current))
    return [b for b in blocks if b.strip()]


def _is_nonfactual_lead_in(block: str) -> bool:
    """Ignore short structural labels such as 'The main components are:'"""
    cleaned = re.sub(r"[*_`#]", "", block).strip()
    return cleaned.endswith(":") and len(cleaned.split()) <= 14


def _is_display_math_block(block: str) -> bool:
    r"""Return True for a standalone Markdown/LaTeX display-math block.

    Supported forms include ``\[ ... \]``, ``$$ ... $$`` and common
    numbered/aligned LaTeX environments. Inline math remains part of the
    surrounding prose paragraph and is validated with that paragraph.
    """
    value = block.strip()
    if not value:
        return False

    if value.startswith(r"\[") and value.endswith(r"\]"):
        return True
    if value.startswith("$$") and value.endswith("$$"):
        return True

    begin = re.match(
        r"^\\begin\{(?P<environment>equation\*?|align\*?|alignat\*?|"
        r"flalign\*?|gather\*?|multline\*?|eqnarray\*?|displaymath)\}",
        value,
        flags=re.IGNORECASE,
    )
    if begin:
        end_marker = rf"\end{{{begin.group('environment')}}}"
        return end_marker in value
    return False


def _is_citation_only_block(block: str) -> bool:
    """Recognise a citation-only paragraph that immediately follows an equation.

    The citation is still validated globally against retrieved paper/page pairs;
    this helper only associates its location with the immediately preceding
    display equation. It does not make arbitrary uncited prose acceptable.
    """
    if not CITATION_RE.search(block):
        return False
    remainder = CITATION_RE.sub("", block)
    remainder = re.sub(r"[*_`~\s.,;:()]+", "", remainder)
    return remainder == ""


def validate_citations(answer: str, retrieved: list[RetrievedChunk]) -> CitationCheck:
    allowed = {
        (_normalise_label(c.citation_name), int(c.page))
        for c in retrieved
    }

    parsed = list(CITATION_RE.finditer(answer))
    found: list[str] = []
    invalid: list[str] = []
    parsed_spans = {(m.start(), m.end()) for m in parsed}

    for match in parsed:
        name = match.group("paper").strip()
        page = int(match.group("page"))
        canonical = f"[{name}, p. {page}]"
        found.append(canonical)
        if (_normalise_label(name), page) not in allowed:
            invalid.append(canonical)

    malformed: list[str] = []
    for match in CITATION_LIKE_RE.finditer(answer):
        if not any(start <= match.start() and match.end() <= end for start, end in parsed_spans):
            malformed.append(match.group(0))

    if malformed:
        return CitationCheck(
            False, found, invalid, "A citation-like reference has malformed syntax.",
            malformed=malformed,
        )
    if invalid:
        return CitationCheck(
            False, found, invalid,
            "One or more citations do not map to a retrieved paper/page pair.",
        )

    blocks = _answer_blocks(answer)
    uncited: list[str] = []

    # Relaxed policy: citations are validated strictly for correctness, but
    # coverage is checked per logical section rather than every paragraph.
    # A section may contain an introductory sentence, bullets, inline/display
    # equations and explanations, followed by one or more citations.
    #
    # Markdown headings and short bold/italic labels start a new section.
    # In answers without headings, the whole answer is one section.
    sections: list[list[str]] = []
    current: list[str] = []

    def is_section_heading(block: str) -> bool:
        value = block.strip()
        if HEADING_RE.match(value):
            return True
        # Bold/italic-only labels such as **Scaled dot-product attention**
        if re.fullmatch(r"\s*(?:\*\*[^*]{1,100}\*\*|__[^_]{1,100}__|\*[^*]{1,100}\*|_[^_]{1,100}_)\s*:?\s*", value):
            return True
        return _is_nonfactual_lead_in(value)

    for block in blocks:
        if is_section_heading(block):
            if current:
                sections.append(current)
                current = []
            # Heading itself is a label, not a factual claim.
            continue
        current.append(block)
    if current:
        sections.append(current)

    substantive_sections = 0
    for section in sections:
        factual_blocks = [
            b for b in section
            if not REFUSAL_RE.search(b)
            and not _is_citation_only_block(b)
        ]
        if not factual_blocks:
            continue
        substantive_sections += 1

        # One or more valid citations must support each logical section.
        # We still reject invented paper/page pairs above. This avoids false
        # negatives when the model places a citation at the end of a section
        # rather than repeating it on every paragraph, bullet or equation.
        if not any(CITATION_RE.search(b) for b in section):
            # Also allow a citation-only paragraph immediately after a math
            # block; it is in the same section if no heading intervenes.
            uncited.extend(b[:180] for b in factual_blocks[:3])

    if uncited:
        return CitationCheck(
            False, found, [],
            "A substantive answer section has no valid citation. Add at least one citation to each section.",
            uncited_blocks=uncited,
        )

    if not found and substantive_sections:
        return CitationCheck(
            False, found, [],
            "No verifiable citation was found in a substantive answer.",
        )


    return CitationCheck(
        True, found, [],
        "All citations map to retrieved paper/page pairs and factual blocks are cited." if found
        else "A clear evidence-gap refusal was accepted.",
    )
