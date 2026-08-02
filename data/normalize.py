"""Answer extraction and equivalence, per answer *type* rather than per dataset.

Three answer types cover the five benchmarks:

``letter``   multiple choice -- MMLU-Pro, GPQA, TruthfulQA (MC1).
``number``   a single numeric value -- GSM8K.
``math``     a LaTeX expression -- MATH-500.

Keeping these here rather than inside each task means a new benchmark of an
existing type inherits a parser that has already been tested, and that two
benchmarks of the same type cannot silently disagree about what counts as
correct.

Every parser must satisfy the contract in :mod:`data.tasks`: it takes free-form
model output, tolerates ``""``, and returns ``""`` when it can extract nothing.
Returning ``""`` is meaningful -- the runner counts it as a parse failure --
so no parser may invent a plausible-looking answer.
"""

from __future__ import annotations

import re
from typing import List, Optional

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ------------------------------------------------------------ multiple choice --
#: Ordered by how explicit the statement is. The first pattern that matches
#: wins, so "the answer is (B)" is never overruled by a stray "A" earlier in
#: the reasoning.
_LETTER_PATTERNS = (
    r"answer\s*(?:is|:|=)\s*\(?\s*([A-Z])\s*\)?",
    r"final\s+answer\s*(?:is|:|=)?\s*\(?\s*([A-Z])\s*\)?",
    r"\\boxed\{\s*\(?\s*([A-Z])\s*\)?\s*\}",
    r"option\s*\(?\s*([A-Z])\s*\)?",
    r"\*\*\s*\(?\s*([A-Z])\s*\)?\s*\*\*",
)


def parse_choice_letter(text: str, n_options: int = 26) -> str:
    """Extract an option letter from model output.

    Searches explicit answer statements first, then falls back to a lone letter
    standing on its own (``"C"``, ``"(C)"``, ``"C."``). Letters beyond the
    option count are rejected rather than clamped: a model answering ``"J"`` to
    a four-option question has not answered it, and scoring that as a wrong
    option would hide a prompting bug.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    valid = set(LETTERS[: max(0, min(int(n_options), len(LETTERS)))])

    for pattern in _LETTER_PATTERNS:
        matches = re.findall(pattern, raw, flags=re.IGNORECASE)
        for candidate in reversed(matches):
            letter = candidate.upper()
            if letter in valid:
                return letter

    # A bare option, possibly parenthesised or followed by punctuation.
    bare = re.fullmatch(r"\(?\s*([A-Za-z])\s*[).:]?", raw)
    if bare and bare.group(1).upper() in valid:
        return bare.group(1).upper()

    # Last resort: a standalone capital letter token anywhere in the text.
    for candidate in reversed(re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", raw)):
        if candidate in valid:
            return candidate
    return ""


def letter_for_index(index: int) -> str:
    """``0 -> "A"``. Raises rather than wrapping past Z."""
    if not 0 <= index < len(LETTERS):
        raise ValueError(f"option index {index} is outside A-Z")
    return LETTERS[index]


def index_for_letter(letter: str) -> Optional[int]:
    letter = str(letter or "").strip().upper()
    return LETTERS.index(letter) if len(letter) == 1 and letter in LETTERS else None


# --------------------------------------------------------------------- number --
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_number(text: str) -> str:
    """Extract the final numeric value from free-form output.

    Takes the *last* number: chain-of-thought restates the result at the end.
    Thousands separators, currency symbols and trailing units are stripped
    before matching, so ``"$1,234.00 per day"`` yields ``"1234.00"``.
    """
    raw = str(text or "")
    if not raw.strip():
        return ""
    cleaned = raw.replace(",", "").replace("$", "").replace("%", "")
    # Drop LaTeX wrappers that would otherwise split a number.
    cleaned = cleaned.replace("\\!", "").replace("\\,", "").replace("\\$", "")
    matches = _NUMBER.findall(cleaned)
    return matches[-1] if matches else ""


def numbers_equal(prediction: str, gold: str, *, tolerance: float = 1e-6) -> bool:
    """Numeric equality with a tolerance; falls back to string equality.

    ``"18"``, ``"18.0"`` and ``"18.000001"`` are the same answer; ``"18"`` and
    ``"eighteen"`` are not, because a benchmark whose gold answers are numeric
    should not be silently scored on prose.
    """
    left, right = str(prediction or "").strip(), str(gold or "").strip()
    if not left:
        return False
    try:
        return abs(float(left.replace(",", "")) - float(right.replace(",", ""))) <= tolerance
    except (TypeError, ValueError):
        return left == right


# ----------------------------------------------------------------------- math --
def extract_boxed(text: str) -> str:
    """Contents of the last ``\\boxed{...}``, with brace matching.

    A regex cannot do this: MATH answers nest braces
    (``\\boxed{\\frac{1}{2}}``). Returns ``""`` when there is no complete box.
    """
    raw = str(text or "")
    marker = raw.rfind("\\boxed")
    if marker < 0:
        return ""
    index = raw.find("{", marker)
    if index < 0:
        # \boxed 5 (no braces) -- take the rest of the token.
        tail = raw[marker + len("\\boxed") :].strip()
        return tail.split()[0] if tail else ""
    depth = 0
    for position in range(index, len(raw)):
        if raw[position] == "{":
            depth += 1
        elif raw[position] == "}":
            depth -= 1
            if depth == 0:
                return raw[index + 1 : position].strip()
    return ""  # unbalanced


def _fix_fractions(text: str) -> str:
    """``\\frac12`` -> ``\\frac{1}{2}``, ``\\frac{1}2`` -> ``\\frac{1}{2}``."""
    parts = text.split("\\frac")
    out = parts[0]
    for part in parts[1:]:
        if not part:
            out += "\\frac"
            continue
        if part[0] == "{":
            out += "\\frac" + part
            continue
        if len(part) >= 2:
            numerator, denominator, rest = part[0], part[1], part[2:]
            if denominator == "{":
                out += "\\frac{" + numerator + "}" + denominator + rest
            else:
                out += "\\frac{" + numerator + "}{" + denominator + "}" + rest
        else:
            out += "\\frac" + part
    return out


def _fix_slash_fraction(text: str) -> str:
    """``a/b`` -> ``\\frac{a}{b}`` for simple integer fractions only."""
    match = re.fullmatch(r"(-?\d+)/(-?\d+)", text)
    if not match:
        return text
    return f"\\frac{{{match.group(1)}}}{{{match.group(2)}}}"


def _fix_sqrt(text: str) -> str:
    """``\\sqrt3`` -> ``\\sqrt{3}``."""
    return re.sub(r"\\sqrt(?!\s*\{)\s*([A-Za-z0-9])", r"\\sqrt{\1}", text)


def _strip_text_wrappers(text: str) -> str:
    """Drop ``\\text{...}`` / ``\\mbox{...}`` wrappers and their unit content."""
    for command in ("\\text", "\\mbox", "\\textbf", "\\mathrm"):
        while command + "{" in text:
            start = text.find(command + "{")
            open_brace = text.find("{", start)
            depth = 0
            end = -1
            for position in range(open_brace, len(text)):
                if text[position] == "{":
                    depth += 1
                elif text[position] == "}":
                    depth -= 1
                    if depth == 0:
                        end = position
                        break
            if end < 0:
                break
            text = text[:start] + text[open_brace + 1 : end] + text[end + 1 :]
    return text


def normalize_math(text: str) -> str:
    """Canonical form for comparing LaTeX answers.

    Follows the conventions the MATH literature settled on: presentation-only
    differences (``\\dfrac`` vs ``\\frac``, ``\\left(``, ``0.5`` vs ``.5``,
    units, ``x =`` prefixes, thousands separators) are removed; anything that
    could change the value is left alone.

    This is deliberately conservative. It normalises formatting, it does not
    evaluate expressions: ``\\frac{1}{2}`` and ``0.5`` stay different strings
    here and are reconciled, if at all, by the numeric fallback in
    :func:`math_equal`.
    """
    value = str(text or "").strip()
    if not value:
        return ""

    value = value.replace("\n", " ").replace("\\!", "").replace("\\ ", " ")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("^{\\circ}", "").replace("^\\circ", "")
    value = value.replace("\\$", "").replace("$", "")
    value = value.replace("\\%", "").replace("%", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = _strip_text_wrappers(value)
    value = value.replace("\\cdot", "*").replace("\\times", "*")
    value = value.rstrip(".").strip()

    # "x = 5" -> "5", but only when there is exactly one equals sign and the
    # left side is a bare variable: "x + y = 5" must keep its meaning.
    if value.count("=") == 1:
        left, right = value.split("=")
        if re.fullmatch(r"\s*[A-Za-z]\w*\s*", left):
            value = right.strip()

    value = value.replace(" ", "")
    value = re.sub(r"(\d),(\d\d\d)", r"\1\2", value)  # 1,000 -> 1000
    if value.startswith("."):
        value = "0" + value
    value = value.replace("{,}", "")
    value = _fix_sqrt(_fix_fractions(value))
    value = _fix_slash_fraction(value)

    # Trailing zeros in a decimal are formatting, not precision.
    if re.fullmatch(r"-?\d+\.\d+", value):
        value = value.rstrip("0").rstrip(".")
    return value


def parse_math_answer(text: str) -> str:
    """Extract a MATH-style answer from model output.

    A ``\\boxed{}`` wins when present -- it is what the prompt asks for and what
    the gold answers use. Otherwise the last line is taken, after stripping
    common lead-ins, which is the best available guess at the stated answer.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    boxed = extract_boxed(raw)
    if boxed:
        return boxed.strip()

    tail = [line.strip() for line in raw.splitlines() if line.strip()]
    if not tail:
        return ""
    candidate = tail[-1]
    candidate = re.sub(
        r"^(the\s+)?(final\s+)?answer\s*(is|:|=)\s*", "", candidate, flags=re.IGNORECASE
    ).strip()
    return candidate.rstrip(".").strip()


def math_equal(prediction: str, gold: str) -> bool:
    """Equivalence for MATH-500 answers.

    Normalised string match first, then a numeric comparison so ``0.5`` and
    ``0.50`` agree. Symbolic equivalence (``\\frac{1}{2}`` vs ``0.5``) is *not*
    claimed: doing it properly needs a CAS, and pretending otherwise would
    inflate accuracy in a way nobody could audit from the logs.
    """
    left, right = normalize_math(prediction), normalize_math(gold)
    if not left:
        return False
    if left == right:
        return True
    try:
        return abs(float(left) - float(right)) <= 1e-6
    except (TypeError, ValueError):
        return False


def format_options(choices: List[str]) -> str:
    """Render options as ``A. text`` lines, matching the letter semantics."""
    return "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices))


__all__ = [
    "LETTERS",
    "extract_boxed",
    "format_options",
    "index_for_letter",
    "letter_for_index",
    "math_equal",
    "normalize_math",
    "numbers_equal",
    "parse_choice_letter",
    "parse_math_answer",
    "parse_number",
]
