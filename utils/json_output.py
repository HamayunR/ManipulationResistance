"""Parsing model output that is *meant* to be JSON.

Agents are asked to answer in JSON. Models mostly comply with the structure and
routinely break the string escaping, because the fields they are filling in
contain mathematics::

    {"answer": "B", "reasoning": "The expectation value of \\sigma{z} is ..."}

A backslash inside a JSON string starts an escape sequence, so LaTeX lands in
one of two states, and the second is worse than the first:

* ``\\sigma``, ``\\(``, ``\\uparrow`` are **invalid** escapes -- the parse
  fails outright and every field is lost, including the answer and the
  confidence;
* ``\\frac``, ``\\times``, ``\\theta``, ``\\rangle``, ``\\begin`` are **valid**
  escapes for formfeed, tab, tab, carriage return and backspace -- the parse
  succeeds and quietly hands back mangled prose.

A third break comes from the same place: a model that lays its reasoning out
over several lines puts a raw newline inside the string, which JSON forbids.

This module repairs all three, and reports which repair it made, so that a
rewritten model output is never mistaken for one the model wrote that way.
Recovery is ordered from safest to least safe and stops at the first reading
that works.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

#: Characters that legitimately follow a backslash inside a JSON string.
#: Anything else is not an escape sequence at all.
_JSON_ESCAPE_CHARS = frozenset('"\\/bfnrtu')

#: Control characters JSON spells with a short escape rather than ``\uXXXX``.
_CONTROL_ESCAPES = {
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}

#: LaTeX commands that collide with a JSON escape character. ``\theta`` is a
#: tab followed by ``heta``; ``\frac`` a formfeed followed by ``rac``. Both
#: readings are legal JSON, so the text alone cannot say which was meant, and
#: ``"first line\nStep two"`` -- a real newline -- is just as plausible as
#: ``\nabla``.
#:
#: Naming the commands is what resolves it. Only the five escape letters can
#: collide, so the list is small and closed, and anything not on it keeps its
#: JSON meaning: an unrecognised command is left as the control character it
#: decodes to rather than guessed at.
_LATEX_COMMANDS = frozenset(
    """
    bar because begin beta big bigg binom bmod boldsymbol bot bullet
    fbox flat footnotesize forall frac frown
    nabla ne neq newline nonumber not notin nrightarrow nsubseteq nu
    rangle rbrace rceil ref restriction rfloor rho right rightarrow rm rvert
    tan tanh tau text textbf textit textrm textstyle tfrac therefore theta
    tilde times to top trace triangle
    """.split()
)

#: A backslash, one of the colliding escape letters, and the run of letters
#: that follows. Whether that run names a command is decided by
#: :func:`_is_latex_command`.
_ESCAPED_LATEX = re.compile(r"\\([bfnrt][A-Za-z]+)")


def _is_latex_command(text: str, index: int) -> bool:
    """Whether the backslash at ``index`` opens a LaTeX command, not an escape."""
    match = _ESCAPED_LATEX.match(text, index)
    return bool(match) and match.group(1) in _LATEX_COMMANDS


def _contains_latex_command(text: str) -> bool:
    """Whether any ``\\theta``-shaped command appears, decodable but mangled."""
    return any(
        match.group(1) in _LATEX_COMMANDS for match in _ESCAPED_LATEX.finditer(text)
    )


#: How a payload was recovered. Ordered from untouched to most repaired.
REPAIR_NONE = "none"
REPAIR_TRAILING_COMMA = "trailing_comma"
REPAIR_STRING_LITERALS = "string_literals"
REPAIR_FAILED = "failed"


@dataclass(frozen=True)
class JsonParse:
    """The outcome of reading one model output.

    Attributes
    ----------
    payload:
        The decoded object, or ``{}`` when nothing could be read.
    repair:
        Which recovery step produced :attr:`payload`; one of the ``REPAIR_*``
        constants. Anything but :data:`REPAIR_NONE` means the text handed to
        the decoder is not byte-for-byte what the model emitted.
    error:
        The decoder's complaint when nothing could be read, for logs.
    """

    payload: Dict[str, Any] = field(default_factory=dict)
    repair: str = REPAIR_FAILED
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.payload)

    @property
    def repaired(self) -> bool:
        return self.ok and self.repair != REPAIR_NONE


def parse_json_object(text: str) -> JsonParse:
    """Read the JSON object out of ``text``, repairing broken string literals.

    Recovery order, stopping at the first reading that works:

    1. the text between the outermost braces, as-is;
    2. the same with trailing commas removed, a common local-model slip;
    3. the same with its string literals repaired -- stray backslashes escaped
       and raw control characters spelled out.

    Step 1 succeeding is not the end of it: a payload that decoded cleanly but
    contains ``\\theta``-shaped escapes is re-read through step 3, because that
    payload is intact JSON carrying broken text.
    """
    if not text:
        return JsonParse(error="empty output")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return JsonParse(error="no JSON object found")
    candidate = text[start : end + 1]

    payload, error = _load(candidate)
    if payload is not None:
        # Decoded, but LaTeX that happens to spell a valid escape has already
        # been swallowed into control characters. Prefer the repaired reading.
        if _contains_latex_command(candidate):
            repaired, _ = _load(repair_string_literals(candidate))
            if repaired is not None:
                return JsonParse(repaired, REPAIR_STRING_LITERALS)
        return JsonParse(payload, REPAIR_NONE)

    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", candidate)
    payload, error = _load(without_trailing_commas)
    if payload is not None:
        return JsonParse(payload, REPAIR_TRAILING_COMMA)

    payload, error = _load(repair_string_literals(without_trailing_commas))
    if payload is not None:
        return JsonParse(payload, REPAIR_STRING_LITERALS)

    return JsonParse(error=error)


def repair_string_literals(text: str) -> str:
    """Make every string literal in ``text`` decodable, leaving structure alone.

    One pass, tracking whether it is inside a string, because both repairs need
    that context -- a brace or a backslash between strings must not be touched:

    * a backslash that does not open a real escape is doubled, so ``\\sigma``
      survives as text. Real escapes are left paired, including ``\\"``, which
      would otherwise appear to end the string; and
    * a raw control character is replaced by its escape, so a reasoning field
      written across several lines still parses.

    ``\\uXXXX`` counts as a real escape only when four hexadecimal digits
    follow, so ``\\uparrow`` is repaired rather than read as a broken code
    point.
    """
    out: list[str] = []
    in_string = False
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if not in_string:
            in_string = char == '"'
            out.append(char)
            index += 1
            continue

        if char == '"':
            in_string = False
            out.append(char)
            index += 1
            continue

        if char < " ":
            # A literal control character is never valid inside a JSON string,
            # so spelling it out cannot change the meaning of anything that
            # already parsed.
            out.append(_CONTROL_ESCAPES.get(char, f"\\u{ord(char):04x}"))
            index += 1
            continue

        if char != "\\":
            out.append(char)
            index += 1
            continue

        following = text[index + 1] if index + 1 < length else ""
        if following == "u":
            if _is_hex4(text[index + 2 : index + 6]):
                out.append(text[index : index + 6])
                index += 6
            else:
                out.append("\\\\")
                index += 1
            continue

        if following in _JSON_ESCAPE_CHARS and not _is_latex_command(text, index):
            out.append(char)
            out.append(following)
            index += 2
            continue

        out.append("\\\\")
        index += 1

    return "".join(out)


def _load(candidate: str) -> Tuple[Dict[str, Any] | None, str]:
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, exc.msg
    return (payload, "") if isinstance(payload, dict) else (None, "not a JSON object")


def _is_hex4(chunk: str) -> bool:
    return len(chunk) == 4 and all(c in "0123456789abcdefABCDEF" for c in chunk)


__all__ = [
    "REPAIR_FAILED",
    "REPAIR_NONE",
    "REPAIR_STRING_LITERALS",
    "REPAIR_TRAILING_COMMA",
    "JsonParse",
    "parse_json_object",
    "repair_string_literals",
]
