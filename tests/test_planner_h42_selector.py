"""Focused checks for the H4.2 multi-label candidate selector.

These tests do not load a checkpoint, MiniLM, DEV, or TEST.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from tiergraph.planner.align import TokenCharSpan
from tiergraph.planner.h42_selector import (
    H42_MAX_SPAN_TOKENS,
    H42CandidateSelector,
    WordSpan,
    candidate_targets,
    candidate_token_mask,
    enumerate_word_spans,
    h42_loss,
    select_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def _content(start: int, end: int) -> TokenCharSpan:
    return TokenCharSpan(
        char_start=start,
        char_end=end,
        is_special=False,
        is_padding=False,
    )


def test_candidates_come_from_query_text_only_and_cover_short_spans():
    signature = inspect.signature(enumerate_word_spans)
    assert "gold" not in signature.parameters
    query = "Am I allergic to it on my menu?"
    spans = enumerate_word_spans(query)
    offsets = {(span.start, span.end) for span in spans}
    words = ["I", "it", "my", "menu", "my menu"]
    for word in words:
        start = query.index(word)
        assert (start, start + len(word)) in offsets
    assert all(1 <= _word_count(query, span) <= H42_MAX_SPAN_TOKENS for span in spans)
    assert len(offsets) == len(spans)
    longer = query
    nine = "one two three four five six seven eight nine"
    nine_spans = enumerate_word_spans(nine)
    assert all(_word_count(nine, span) <= 8 for span in nine_spans)
    assert not any(_word_count(nine, span) == 9 for span in nine_spans)
    booked = "What time is my taxi booked for?"
    booked_spans = enumerate_word_spans(booked)
    start = booked.index("my taxi booked for")
    assert (start, start + len("my taxi booked for")) in {
        (span.start, span.end) for span in booked_spans
    }


def test_labels_are_multi_label_and_none_when_unowned():
    query = "Does this menu match my allergy?"
    spans = enumerate_word_spans(query)
    menu = _find(query, "this menu")
    allergy = _find(query, "my allergy")
    targets, none_targets = candidate_targets(
        spans,
        2,
        (
            (menu.start, menu.end, 0),
            (allergy.start, allergy.end, 0),
        ),
    )
    menu_index = _index(spans, menu)
    allergy_index = _index(spans, allergy)
    assert targets[menu_index, 0] == 1
    assert targets[allergy_index, 0] == 1
    assert targets[:, 1].sum() == 0
    assert none_targets.tolist() == [0.0, 1.0]
    assert int(targets[:, 0].sum()) == 2


def test_selection_allows_zero_one_or_many_and_never_forces_one():
    candidate_logits = torch.tensor(
        [
            [3.0, -2.0, 0.2],
            [1.5, -2.0, -3.0],
            [-4.0, -2.0, -3.0],
        ]
    )
    none_logits = torch.tensor([0.0, 5.0, 0.0])
    chosen = select_candidates(candidate_logits, none_logits)
    assert chosen == ((0, 1), (), (0,))
    assert any(len(item) != 1 for item in chosen)


def test_same_span_can_be_scored_independently_without_a_sharing_rule():
    candidate_logits = torch.tensor([[2.0, 2.0], [-1.0, 3.0]])
    none_logits = torch.tensor([0.0, 0.0])
    assert select_candidates(candidate_logits, none_logits) == ((0,), (0, 1))


def test_scorer_uses_one_embedding_tensor_and_rejects_unaligned_spans():
    query = "my gate"
    tokens = (_content(0, 2), _content(3, 7))
    spans = enumerate_word_spans(query)
    candidate_mask = candidate_token_mask(spans, tokens)
    assert candidate_mask.shape == (len(spans), 2)
    operation_mask = candidate_token_mask((WordSpan(0, 2),), tokens)
    embeddings = torch.randn(2, 4)
    selector = H42CandidateSelector(4)
    candidate_logits, none_logits = selector.score(
        embeddings,
        operation_mask,
        candidate_mask,
    )
    assert candidate_logits.shape == (len(spans), 1)
    assert none_logits.shape == (1,)
    broken = candidate_token_mask((WordSpan(100, 104),), tokens)
    broken_logits, broken_none = selector.score(embeddings, operation_mask, broken)
    lowest = torch.finfo(broken_logits.dtype).min
    assert torch.equal(broken_logits, torch.full_like(broken_logits, lowest))
    assert torch.isfinite(broken_none).all()
    assert select_candidates(broken_logits.detach(), broken_none.detach()) == ((),)


def test_loss_trains_the_selector_and_leaves_the_bio_head_alone():
    query = "Locate my gate"
    spans = enumerate_word_spans(query)
    gold = _find(query, "my gate")
    targets, none_targets = candidate_targets(spans, 1, ((gold.start, gold.end, 0),))
    tokens = tuple(
        _content(span.start, span.end)
        for span in enumerate_word_spans(query, max_tokens=1)
    )
    mask = candidate_token_mask(spans, tokens)
    representable = mask.sum(dim=1) > 0
    selector = H42CandidateSelector(4)
    embeddings = torch.randn(len(tokens), 4)
    logits, none_logits = selector.score(embeddings, mask[:1], mask)
    loss = h42_loss(
        logits,
        none_logits,
        targets,
        none_targets,
        representable,
        torch.tensor([True]),
    )
    loss.backward()
    assert selector.anchor_proj.weight.grad is not None
    assert float(selector.anchor_proj.weight.grad.abs().sum()) > 0
    assert selector.none_vector.grad is not None
    model_source = (ROOT / "tiergraph" / "planner" / "model.py").read_text(encoding="utf-8")
    assert "self.anc_bio_head" in model_source
    assert "h42" not in model_source.casefold()
    script = (ROOT / "scripts" / "build_planner_v31_correction_manifest.py").read_text(
        encoding="utf-8"
    )
    assert "h42" not in script.casefold()


def _word_count(query: str, span: WordSpan) -> int:
    import re

    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", span.text(query)))


def _find(query: str, text: str) -> WordSpan:
    start = query.index(text)
    return WordSpan(start, start + len(text))


def _index(spans: tuple[WordSpan, ...], span: WordSpan) -> int:
    return next(
        index
        for index, item in enumerate(spans)
        if (item.start, item.end) == (span.start, span.end)
    )
