"""Tests for reading model output that is meant to be JSON.

Agents answer in JSON and write mathematics in the string fields, which breaks
JSON escaping. The cases below are the ones a real GPQA run against
Qwen2.5-7B produced: of 385 model outputs, 115 failed to decode and every one
of those failures was a LaTeX backslash.

Two properties matter and are tested in both directions:

* anything that was already valid JSON must decode to exactly what it decoded
  to before -- the repair must never rewrite a payload the model got right; and
* a payload that decodes but carries mangled text must be caught, because that
  failure is silent and therefore worse than the loud one.
"""

from __future__ import annotations

import json

import pytest

from utils.json_output import (
    REPAIR_NONE,
    REPAIR_STRING_LITERALS,
    REPAIR_TRAILING_COMMA,
    parse_json_object,
    repair_string_literals,
)


# ------------------------------------------------------- valid JSON is safe --
@pytest.mark.parametrize(
    "payload",
    [
        {"answer": "B", "confidence": 3, "reasoning": "plain prose"},
        {"answer": "18", "confidence": 5, "reasoning": 'quoted "words" and \\ slashes'},
        {"answer": "A", "reasoning": "a\tb"},
        {"answer": "A", "reasoning": "line one\nline two"},
        {"answer": "C", "reasoning": "unicode \u00e9\u00e8 and emoji \U0001f600"},
        {"answer": "D", "reasoning": "path C:\\Users\\x"},
        {"reviews": [{"target": 1, "assessment": "Strong", "correction": ""}]},
    ],
)
def test_valid_json_is_returned_untouched(payload):
    parse = parse_json_object(json.dumps(payload))
    assert parse.payload == payload
    assert parse.repair == REPAIR_NONE


def test_prose_around_the_object_is_ignored():
    parse = parse_json_object('Sure!\n```json\n{"answer": "B"}\n```\nHope that helps.')
    assert parse.payload == {"answer": "B"}


def test_trailing_commas_are_still_recovered():
    parse = parse_json_object('{"answer": "B", "confidence": 3,}')
    assert parse.payload == {"answer": "B", "confidence": 3}
    assert parse.repair == REPAIR_TRAILING_COMMA


# --------------------------------------------------- LaTeX that fails to parse --
def test_backslash_command_no_longer_loses_the_payload():
    # Verbatim from the run: this decoded to nothing, so the answer and the
    # confidence were both lost and the router saw a fabricated default.
    text = (
        '{\n  "answer": "B",\n  "confidence": 3,\n  "reasoning": "For \\sigma{z}, '
        'the eigenvalues are +1 for spin-up and -1 for spin-down."\n}'
    )
    parse = parse_json_object(text)
    assert parse.payload["answer"] == "B"
    assert parse.payload["confidence"] == 3
    assert "\\sigma{z}" in parse.payload["reasoning"]
    assert parse.repair == REPAIR_STRING_LITERALS


@pytest.mark.parametrize(
    "snippet",
    [
        r"\sigma_z",
        r"\(\sigma_z\)",
        r"\uparrow",
        r"\downarrow",
        r"\langle\psi|",
        r"\[E = mc^2\]",
        r"\alpha + \delta",
        r"10^{-3}\,\mathrm{m}",
    ],
)
def test_common_latex_survives_as_written(snippet):
    parse = parse_json_object(json_with_reasoning(snippet))
    assert parse.ok
    assert parse.payload["reasoning"] == f"the value is {snippet} exactly"


def json_with_reasoning(snippet: str) -> str:
    """Hand-build the broken JSON a model emits; json.dumps would escape it."""
    return '{"answer": "B", "reasoning": "the value is ' + snippet + ' exactly"}'


# ------------------------------------------- LaTeX that parses but is mangled --
@pytest.mark.parametrize(
    "snippet",
    [
        r"\theta",   # \t -> tab
        r"\times",   # \t -> tab
        r"\frac{1}{2}",  # \f -> formfeed
        r"\rangle",  # \r -> carriage return
        r"\begin{bmatrix}",  # \b -> backspace
        r"\nabla",   # \n -> newline
    ],
)
def test_latex_that_spells_a_valid_escape_is_caught(snippet):
    # These decode without complaint, which is why they are the dangerous ones:
    # the payload looks fine and the prose has a control character in it.
    text = json_with_reasoning(snippet)
    assert json.loads(text)["reasoning"] != f"the value is {snippet} exactly"

    parse = parse_json_object(text)
    assert parse.payload["reasoning"] == f"the value is {snippet} exactly"
    assert parse.repair == REPAIR_STRING_LITERALS


@pytest.mark.parametrize(
    "escaped, decoded",
    [
        (r"a\tb", "a\tb"),
        (r"first line\nStep two", "first line\nStep two"),
        (r"col\tvalue", "col\tvalue"),
        (r"one\nrepeat", "one\nrepeat"),
        (r"\ttotally not latex", "\ttotally not latex"),
    ],
)
def test_escapes_that_do_not_name_a_command_keep_their_json_meaning(escaped, decoded):
    # The dangerous direction: prose after a real newline or tab looks exactly
    # like a LaTeX command, so only names on the list are re-read.
    parse = parse_json_object('{"reasoning": "' + escaped + '"}')
    assert parse.payload["reasoning"] == decoded
    assert parse.repair == REPAIR_NONE


# ------------------------------------------------------- other broken strings --
def test_raw_newlines_inside_a_string_are_recovered():
    # A model laying its reasoning out over lines emits a literal newline,
    # which JSON forbids. Unambiguous: valid JSON can never contain one.
    text = '{"answer": "B", "reasoning": "first line\nsecond line"}'
    parse = parse_json_object(text)
    assert parse.payload["reasoning"] == "first line\nsecond line"
    assert parse.repair == REPAIR_STRING_LITERALS


def test_a_repair_pass_leaves_the_good_escapes_in_the_same_string_alone():
    # The combination that matters: one field forces the repair, and another
    # holds a real newline followed by prose. Repairing the first must not
    # rewrite the second.
    text = (
        '{"reasoning": "the value of \\sigma is 1", '
        '"neighbor_assessment": "first line\\nStep two follows"}'
    )
    parse = parse_json_object(text)
    assert parse.payload["reasoning"] == r"the value of \sigma is 1"
    assert parse.payload["neighbor_assessment"] == "first line\nStep two follows"


def test_quotes_stay_paired_through_a_repair():
    text = r'{"answer": "B", "reasoning": "he said \"\sigma\" loudly"}'
    parse = parse_json_object(text)
    assert parse.payload["reasoning"] == r'he said "\sigma" loudly'


def test_a_lone_trailing_backslash_does_not_crash():
    parse = parse_json_object('{"answer": "B", "reasoning": "ends with \\')
    assert not parse.ok
    assert parse.repair == "failed"


@pytest.mark.parametrize("text", ["", "   ", "no braces here", "}{"])
def test_output_without_an_object_reports_failure(text):
    parse = parse_json_object(text)
    assert not parse.ok
    assert parse.payload == {}


def test_an_object_wrapped_in_an_array_is_still_recovered():
    # The scan runs between the outermost braces, so a model that wraps its
    # answer in a list still gets read rather than scored as a parse failure.
    assert parse_json_object('[{"answer": "B"}]').payload == {"answer": "B"}


# ------------------------------------------------------------ the repair pass --
def test_repair_leaves_structure_alone():
    text = '{"a": "x", "b": {"c": [1, 2]}}'
    assert repair_string_literals(text) == text


def test_repair_keeps_real_unicode_escapes():
    assert repair_string_literals(r'{"a": "caf\u00e9"}') == r'{"a": "caf\u00e9"}'


def test_repair_fixes_a_unicode_escape_that_is_really_latex():
    # \uparrow is not four hex digits, so it is text, not a code point.
    assert json.loads(repair_string_literals(r'{"a": "\uparrow"}'))["a"] == r"\uparrow"


def test_repair_is_idempotent():
    once = repair_string_literals(json_with_reasoning(r"\sigma \frac{1}{2}"))
    assert repair_string_literals(once) == once
