"""Tests for planner head ablation loss wiring."""

from __future__ import annotations

import torch

from tiergraph.planner.loss import HEAD_KEYS, planner_loss


class _FakeGold:
    query_type_labels = torch.tensor([0])
    op_bio_labels = torch.zeros(1, 1, dtype=torch.long)
    anc_bio_labels = torch.zeros(1, 1, dtype=torch.long)
    token_loss_mask = torch.zeros(1, 1, dtype=torch.bool)
    op_type_labels = torch.zeros(1, 1, dtype=torch.long)
    op_valid = torch.zeros(1, 1, dtype=torch.bool)
    impl_labels = torch.zeros(1, 1, dtype=torch.long)
    anc_valid = torch.zeros(1, 1, dtype=torch.bool)
    own_labels = torch.zeros(1, 1, dtype=torch.long)
    own_mask = torch.zeros(1, 1, 1, dtype=torch.bool)
    dep_labels = torch.zeros(1, 1, 1)
    dep_mask = torch.zeros(1, 1, 1, dtype=torch.bool)


class _FakeOutputs:
    query_type_logits = torch.zeros(1, 3)
    op_bio_logits = torch.zeros(1, 1, 3)
    anc_bio_logits = torch.zeros(1, 1, 3)
    op_type_logits = torch.zeros(1, 1, 6)
    impl_logits = torch.zeros(1, 1, 2)
    own_logits = torch.zeros(1, 1, 1)
    dep_logits = torch.zeros(1, 1, 1)


def test_planner_loss_active_heads_excludes_disabled():
    gold = _FakeGold()
    outputs = _FakeOutputs()
    outputs.query_type_logits = torch.tensor([[0.0, 1.0, 2.0]])
    outputs.impl_logits = torch.tensor([[[0.0, 2.0]]])
    gold.impl_labels = torch.tensor([[1]])
    gold.anc_valid = torch.tensor([[True]])
    h1_h5 = planner_loss(outputs, gold, active_heads=frozenset({"h1", "h5"}))
    h1_only = planner_loss(outputs, gold, active_heads=frozenset({"h1"}))
    assert h1_h5.total.item() > h1_only.total.item()


def test_train_config_active_heads():
    from tiergraph.planner.train import TrainConfig

    config = TrainConfig(disabled_heads=("h5", "h7"))
    assert config.active_heads() == frozenset(HEAD_KEYS) - {"h5", "h7"}
