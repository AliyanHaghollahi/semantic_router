"""Focused tests for free-inference Viterbi BIO decoding (Experiment A)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tiergraph.planner.align import BIO_B, BIO_I, BIO_O, TokenCharSpan
from tiergraph.planner.model import (
    decode_bio_spans,
    select_bio_labels,
    viterbi_bio_labels,
)


def _tokens_special_two_content() -> tuple[TokenCharSpan, ...]:
    return (
        TokenCharSpan(None, None, True, False),
        TokenCharSpan(0, 3, False, False),
        TokenCharSpan(4, 8, False, False),
        TokenCharSpan(None, None, True, False),
    )


def _assert_legal_bio_path(labels: list[int], tokens: tuple[TokenCharSpan, ...]) -> None:
    prev: int | None = None
    for label, token in zip(labels, tokens, strict=True):
        if not token.is_content:
            assert label == BIO_O
            prev = BIO_O
            continue
        if prev is None:
            assert label in (BIO_O, BIO_B)
        elif prev == BIO_O:
            assert label in (BIO_O, BIO_B)
        else:
            assert label in (BIO_O, BIO_B, BIO_I)
        prev = label


def test_decode_bio_spans_unchanged_orphan_i_as_b():
    tokens = _tokens_special_two_content()
    labels = [BIO_O, BIO_I, BIO_I, BIO_O]
    assert decode_bio_spans(labels, tokens) == ((0, 8),)


def test_argmax_select_matches_historical_path():
    tokens = _tokens_special_two_content()
    # Prefer I on all positions; historical path forces non-content → O.
    logits = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
        ]
    )
    historical = logits.argmax(dim=-1).tolist()
    historical = [
        BIO_O if not token.is_content else int(label)
        for label, token in zip(historical, tokens, strict=True)
    ]
    assert select_bio_labels(logits, tokens, bio_decode_mode="argmax") == historical
    assert historical == [BIO_O, BIO_I, BIO_I, BIO_O]


def test_viterbi_forbids_illegal_i_start_and_o_to_i():
    tokens = _tokens_special_two_content()
    # Argmax would pick I on content tokens; Viterbi cannot start with I.
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],  # non-content → forced O
            [0.0, 1.0, 8.0],  # content: I >> B > O
            [0.0, 1.0, 8.0],  # content: I >> B > O
            [5.0, 0.0, 0.0],  # non-content → forced O
        ]
    )
    argmax_labels = select_bio_labels(logits, tokens, bio_decode_mode="argmax")
    viterbi_labels = viterbi_bio_labels(logits, tokens)
    assert argmax_labels == [BIO_O, BIO_I, BIO_I, BIO_O]
    assert viterbi_labels[0] == BIO_O
    assert viterbi_labels[-1] == BIO_O
    assert viterbi_labels[1] != BIO_I
    _assert_legal_bio_path(viterbi_labels, tokens)
    # Best legal content path under these emissions is B then I.
    assert viterbi_labels == [BIO_O, BIO_B, BIO_I, BIO_O]


def test_viterbi_maximizes_sequence_log_score_vs_local_trap():
    tokens = (
        TokenCharSpan(0, 1, False, False),
        TokenCharSpan(1, 2, False, False),
        TokenCharSpan(2, 3, False, False),
    )
    # Locally, token0 prefers I (illegal start). Viterbi must pick a legal path.
    logits = torch.tensor(
        [
            [2.0, 1.5, 10.0],  # trap: I huge but illegal at start
            [0.0, 3.0, 0.5],
            [0.0, 0.5, 3.0],
        ]
    )
    labels = viterbi_bio_labels(logits, tokens)
    _assert_legal_bio_path(labels, tokens)
    assert labels[0] != BIO_I

    log_emit = torch.log_softmax(logits.float(), dim=-1)
    legal_from = {
        BIO_O: (BIO_O, BIO_B),
        BIO_B: (BIO_O, BIO_B, BIO_I),
        BIO_I: (BIO_O, BIO_B, BIO_I),
    }

    def _score(path: list[int]) -> float:
        total = float(log_emit[0, path[0]].item())
        for index in range(1, len(path)):
            total += float(log_emit[index, path[index]].item())
        return total

    # Enumerate all legal length-3 paths; Viterbi must match the max score.
    best_score = float("-inf")
    for s0 in (BIO_O, BIO_B):
        for s1 in legal_from[s0]:
            for s2 in legal_from[s1]:
                best_score = max(best_score, _score([s0, s1, s2]))
    assert abs(_score(labels) - best_score) < 1e-5



def test_viterbi_forces_non_content_to_o():
    tokens = _tokens_special_two_content()
    logits = torch.full((4, 3), -5.0)
    logits[:, BIO_I] = 10.0
    labels = viterbi_bio_labels(logits, tokens)
    assert labels[0] == BIO_O
    assert labels[-1] == BIO_O
    _assert_legal_bio_path(labels, tokens)


def test_select_bio_labels_rejects_unknown_mode():
    tokens = _tokens_special_two_content()
    logits = torch.zeros(4, 3)
    try:
        select_bio_labels(logits, tokens, bio_decode_mode="greedy")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "bio_decode_mode" in str(exc)


def test_predict_batch_forwards_bio_decode_mode(monkeypatch):
    from tiergraph.planner import free_eval as free_eval_mod

    captured: dict[str, str] = {}

    class _FakeModel:
        def encode(self, texts):
            return SimpleNamespace(texts=tuple(texts))

        def predict_structures(self, features, *, bio_decode_mode="argmax"):
            captured["bio_decode_mode"] = bio_decode_mode
            return SimpleNamespace(items=(SimpleNamespace(),))

    example = SimpleNamespace(query="Where is my gate?")
    free_eval_mod.predict_batch(
        _FakeModel(),
        (example,),
        bio_decode_mode="viterbi",
    )
    assert captured["bio_decode_mode"] == "viterbi"
