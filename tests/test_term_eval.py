
import csv
from pathlib import Path

import pytest

from src.term_eval import (
    load_terms,
    atomicize,
    score_term_corpus
)
from src.normalize import mixed_tokens


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "terms_fixture.csv"
)


@pytest.fixture
def terms():
    return load_terms(FIXTURE_PATH)


def test_basic_match(terms):
    tokens = mixed_tokens("we use attention today")

    assert atomicize(tokens, terms) == [
        "we", "use", "TERM:004", "today"
    ]


def test_alias_match(terms):
    tokens = mixed_tokens("we use attn today")

    assert atomicize(tokens, terms) == [
        "we", "use", "TERM:004", "today"
    ]


def test_longest_match(terms):
    tokens = mixed_tokens("beam search algorithm")

    assert atomicize(tokens, terms) == [
        "TERM:002", "algorithm"
    ]


def test_english_word_boundary(terms):
    tokens = mixed_tokens("transformer")

    assert atomicize(tokens, terms) == [
        "transformer"
    ]


def test_repeated_terms(terms):
    tokens = mixed_tokens(
        "attention attention today"
    )

    assert atomicize(tokens, terms) == [
        "TERM:004", "TERM:004", "today"
    ]


def test_ordinary_tokens(terms):
    tokens = mixed_tokens("hello world")

    assert atomicize(tokens, terms) == [
        "hello", "world"
    ]


def write_terms_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = [
            "canonical_id",
            "canonical",
            "aliases"
        ]

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def test_duplicate_canonical_id(tmp_path):
    path = tmp_path / "terms.csv"

    write_terms_csv(path, [
        {
            "canonical_id": "001",
            "canonical": "attention",
            "aliases": "[]"
        },
        {
            "canonical_id": "001",
            "canonical": "conformer",
            "aliases": "[]"
        }
    ])

    with pytest.raises(ValueError, match="重复 canonical_id"):
        load_terms(path)


def test_alias_conflict(tmp_path):
    path = tmp_path / "terms.csv"

    write_terms_csv(path, [
        {
            "canonical_id": "001",
            "canonical": "attention",
            "aliases": '["attn"]'
        },
        {
            "canonical_id": "002",
            "canonical": "focus",
            "aliases": '["attn"]'
        }
    ])

    with pytest.raises(ValueError, match="别名冲突"):
        load_terms(path)


def test_invalid_alias_json(tmp_path):
    path = tmp_path / "terms.csv"

    write_terms_csv(path, [
        {
            "canonical_id": "001",
            "canonical": "attention",
            "aliases": "not-json"
        }
    ])

    with pytest.raises(ValueError, match="不是合法 JSON"):
        load_terms(path)


def test_empty_alias(tmp_path):
    path = tmp_path / "terms.csv"

    write_terms_csv(path, [
        {
            "canonical_id": "001",
            "canonical": "attention",
            "aliases": '[""]'
        }
    ])

    with pytest.raises(ValueError, match="存在空别名"):
        load_terms(path)


def test_missing_required_columns(tmp_path):
    path = tmp_path / "terms.csv"

    write_terms_csv(
        path,
        [{"canonical_id": "001"}],
        fieldnames=["canonical_id"]
    )

    with pytest.raises(ValueError, match="缺少必要字段"):
        load_terms(path)


def test_term_corpus_weighted():
    result = score_term_corpus([
        (
            ["TERM:001"],
            ["TERM:002"]
        ),
        (
            ["TERM:001"] * 9,
            ["TERM:001"] * 9
        )
    ])

    assert result["S_t"] == 1
    assert result["D_t"] == 0
    assert result["I_t"] == 0
    assert result["N_t"] == 10
    assert result["term_errors"] == 1
    assert result["term_wer"] == 0.1


def test_term_corpus_zero_reference_terms():
    result = score_term_corpus([
        (
            [],
            ["TERM:001", "TERM:002"]
        )
    ])

    assert result["S_t"] == 0
    assert result["D_t"] == 0
    assert result["I_t"] == 2
    assert result["N_t"] == 0
    assert result["term_errors"] == 2
    assert result["term_wer"] is None


def test_term_corpus_insertion_with_other_samples():
    result = score_term_corpus([
        (
            ["TERM:001"],
            ["TERM:001"]
        ),
        (
            [],
            ["TERM:002"]
        )
    ])

    assert result["I_t"] == 1
    assert result["N_t"] == 1
    assert result["term_wer"] == 1.0


def test_term_corpus_repeated_terms():
    result = score_term_corpus([
        (
            ["TERM:001", "TERM:001"],
            ["TERM:001"]
        ),
        (
            ["TERM:002", "TERM:002"],
            ["TERM:002", "TERM:002"]
        )
    ])

    assert result["D_t"] == 1
    assert result["N_t"] == 4
    assert result["term_wer"] == 0.25


def test_term_corpus_empty_dataset():
    result = score_term_corpus([])

    assert result["S_t"] == 0
    assert result["D_t"] == 0
    assert result["I_t"] == 0
    assert result["N_t"] == 0
    assert result["term_errors"] == 0
    assert result["term_wer"] is None
    assert result["sample_results"] == []
