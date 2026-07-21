"""Tests for the opt-in, one-pass frozen planner feature encoder."""

from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

from tiergraph.planner.encoder import (
    DEFAULT_MINILM_MODEL,
    EncoderBatch,
    FeatureEncoder,
    MiniLMFeatureEncoder,
)


class _FakeBatchEncoding(dict):
    """Small Mapping double with the relevant BatchEncoding behavior."""


class _FakeTokenizer:
    pad_token_id = 0
    all_special_ids = (0, 101, 102)
    model_input_names = ("input_ids", "attention_mask", "token_type_ids")

    def __init__(self, output_transform=None) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
        self.output_transform = output_transform
        self.last_output = None

    def __call__(self, texts, **kwargs):
        self.calls.append((tuple(texts), kwargs))
        input_ids = []
        token_type_ids = []
        for text in texts:
            word_ids = [10 + len(word) for word in text.split()]
            ids = [101, *word_ids, 102]
            input_ids.append(ids)
            token_type_ids.append(list(range(len(ids))))
        output = _FakeBatchEncoding({
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
            "token_type_ids": token_type_ids,
            "scalar_metadata": 99,
        })
        if self.output_transform is not None:
            output = self.output_transform(output)
        self.last_output = output
        return output


class _FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(hidden_size=3)
        self.device_calls: list[torch.device] = []
        self.eval_calls = 0
        self.forward_calls = 0
        self.inference_mode_flags: list[bool] = []
        self.received_token_type_ids: list[torch.Tensor | None] = []

    def to(self, device):
        target = torch.device(device)
        self.device_calls.append(target)
        return super().to(target)

    def eval(self):
        self.eval_calls += 1
        return super().eval()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        self.forward_calls += 1
        self.inference_mode_flags.append(torch.is_inference_mode_enabled())
        self.received_token_type_ids.append(
            None if token_type_ids is None else token_type_ids.detach().clone()
        )
        values = input_ids.to(dtype=torch.float32)
        token_embeddings = torch.stack(
            (values, values * 2, values + 1),
            dim=-1,
        )
        return SimpleNamespace(last_hidden_state=token_embeddings)


class _BadShapeTransformer(_FakeTransformer):
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        self.forward_calls += 1
        return SimpleNamespace(last_hidden_state=input_ids.to(torch.float32))


class _MissingOutputTransformer(_FakeTransformer):
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        self.forward_calls += 1
        return SimpleNamespace()


class _NonTensorOutputTransformer(_FakeTransformer):
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        self.forward_calls += 1
        return SimpleNamespace(last_hidden_state=[[1.0]])


class _MismatchedOutputTransformer(_FakeTransformer):
    def __init__(self, *, batch_delta=0, token_delta=0, hidden_size=3) -> None:
        super().__init__()
        self.batch_delta = batch_delta
        self.token_delta = token_delta
        self.output_hidden_size = hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        self.forward_calls += 1
        batch = input_ids.shape[0] + self.batch_delta
        tokens = input_ids.shape[1] + self.token_delta
        return SimpleNamespace(
            last_hidden_state=torch.ones(
                (batch, tokens, self.output_hidden_size),
                device=input_ids.device,
            )
        )


class _LoaderHarness:
    def __init__(self, model=None, tokenizer=None) -> None:
        self.tokenizer = tokenizer if tokenizer is not None else _FakeTokenizer()
        self.model = model if model is not None else _FakeTransformer()
        self.tokenizer_loads: list[str] = []
        self.model_loads: list[str] = []

    def load_tokenizer(self, model_name: str):
        self.tokenizer_loads.append(model_name)
        return self.tokenizer

    def load_model(self, model_name: str):
        self.model_loads.append(model_name)
        return self.model

    def encoder(self, **kwargs) -> MiniLMFeatureEncoder:
        return MiniLMFeatureEncoder(
            tokenizer_loader=self.load_tokenizer,
            model_loader=self.load_model,
            **kwargs,
        )


def test_imports_remain_isolated_from_encoder_and_runtime_dependencies():
    script = """
import json
import sys
import tiergraph
after_tiergraph = {
    "encoder": "tiergraph.planner.encoder" in sys.modules,
    "torch": "torch" in sys.modules,
    "transformers": "transformers" in sys.modules,
}
import tiergraph.planner.annotations
forbidden = (
    "tiergraph.planner.encoder",
    "torch",
    "transformers",
    "sentence_transformers",
    "router",
    "edge",
    "fog",
    "context_store",
    "config",
    "sqlite3",
    "faiss",
    "httpx",
    "aiohttp",
)
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
print(json.dumps({"after_tiergraph": after_tiergraph, "loaded": loaded}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "after_tiergraph": {
            "encoder": False,
            "torch": False,
            "transformers": False,
        },
        "loaded": [],
    }


def test_construction_is_lazy_and_exposes_configuration():
    harness = _LoaderHarness()
    encoder = harness.encoder()

    assert encoder.model_name == DEFAULT_MINILM_MODEL
    assert encoder.max_length == 128
    assert encoder.device == torch.device("cpu")
    assert not encoder.is_loaded
    assert harness.tokenizer_loads == []
    assert harness.model_loads == []
    assert isinstance(encoder, FeatureEncoder)
    with pytest.raises(RuntimeError, match="before model loading"):
        _ = encoder.hidden_size


def test_first_encode_loads_once_and_repeated_calls_reuse_dependencies():
    harness = _LoaderHarness()
    encoder = harness.encoder()

    first = encoder.encode("first query")
    second = encoder.encode(["second query", "third query"])

    assert encoder.is_loaded
    assert encoder.hidden_size == 3
    assert harness.tokenizer_loads == [DEFAULT_MINILM_MODEL]
    assert harness.model_loads == [DEFAULT_MINILM_MODEL]
    assert len(harness.tokenizer.calls) == 2
    assert harness.model.forward_calls == 2
    assert first.token_embeddings.shape[:2] == first.input_ids.shape
    assert first.pooled_embeddings.shape == (1, 3)
    assert second.batch_size == 2


def test_failed_partial_load_exposes_no_state_and_can_retry_cleanly():
    harness = _LoaderHarness()
    model_attempts = 0

    def flaky_model_loader(model_name):
        nonlocal model_attempts
        model_attempts += 1
        harness.model_loads.append(model_name)
        if model_attempts == 1:
            raise RuntimeError("simulated model load failure")
        return harness.model

    encoder = MiniLMFeatureEncoder(
        tokenizer_loader=harness.load_tokenizer,
        model_loader=flaky_model_loader,
    )

    with pytest.raises(RuntimeError, match="simulated model load failure"):
        encoder.encode("first attempt")
    assert not encoder.is_loaded
    assert encoder._tokenizer is None
    assert encoder._model is None

    batch = encoder.encode("retry")

    assert batch.batch_size == 1
    assert encoder.is_loaded
    assert harness.tokenizer_loads == [DEFAULT_MINILM_MODEL] * 2
    assert harness.model_loads == [DEFAULT_MINILM_MODEL] * 2


def test_each_encode_tokenizes_complete_batch_and_runs_model_exactly_once():
    harness = _LoaderHarness()
    encoder = harness.encoder()

    batch = encoder.encode(("alpha beta", "gamma"))

    assert batch.texts == ("alpha beta", "gamma")
    assert len(harness.tokenizer.calls) == 1
    tokenized_texts, options = harness.tokenizer.calls[0]
    assert tokenized_texts == batch.texts
    assert options == {
        "add_special_tokens": True,
        "padding": False,
        "truncation": False,
        "return_attention_mask": True,
    }
    assert harness.model.forward_calls == 1


def test_batch_encoding_mapping_and_tokenizer_attention_mask_are_used():
    def mask_one_token(output):
        output["attention_mask"][0] = [1, 1, 0, 1]
        return output

    tokenizer = _FakeTokenizer(output_transform=mask_one_token)
    harness = _LoaderHarness(tokenizer=tokenizer)

    batch = harness.encoder().encode("one two")

    assert isinstance(tokenizer.last_output, _FakeBatchEncoding)
    assert batch.input_ids.tolist() == [[101, 13, 102]]
    assert batch.attention_mask.tolist() == [[1, 1, 1]]
    assert harness.model.forward_calls == 1


def test_token_type_ids_use_retained_positions_and_reach_model():
    harness = _LoaderHarness()

    batch = harness.encoder(max_length=4).encode(
        ["one two three", "one"]
    )

    assert batch.input_ids.shape == (2, 4)
    assert len(harness.model.received_token_type_ids) == 1
    received = harness.model.received_token_type_ids[0]
    assert received is not None
    assert received.dtype is torch.int64
    assert received.device == batch.input_ids.device
    assert received.tolist() == [[0, 1, 2, 4], [0, 1, 2, 0]]


@pytest.mark.parametrize(
    "transform,texts,error",
    [
        (
            lambda output: {key: value for key, value in output.items() if key != "attention_mask"},
            ["one"],
            "missing required fields.*attention_mask",
        ),
        (
            lambda output: output | {"attention_mask": []},
            ["one"],
            "batch size must match input_ids",
        ),
        (
            lambda output: output
            | {"attention_mask": [output["attention_mask"][0][:-1]]},
            ["one"],
            "token lengths must match input_ids",
        ),
        (
            lambda output: output | {"attention_mask": [[1, 2, 1]]},
            ["one"],
            "only Boolean or integer 0/1",
        ),
    ],
)
def test_malformed_or_missing_tokenizer_attention_mask_is_rejected(
    transform,
    texts,
    error,
):
    tokenizer = _FakeTokenizer(output_transform=transform)

    with pytest.raises((TypeError, ValueError), match=error):
        _LoaderHarness(tokenizer=tokenizer).encoder().encode(texts)


def test_boolean_tokenizer_attention_mask_preserves_boolean_dtype():
    def boolean_mask(output):
        output["attention_mask"] = [
            [bool(value) for value in sequence]
            for sequence in output["attention_mask"]
        ]
        return output

    batch = _LoaderHarness(
        tokenizer=_FakeTokenizer(output_transform=boolean_mask)
    ).encoder().encode(["one", "one two"])

    assert batch.attention_mask.dtype is torch.bool


def test_variable_length_batch_without_pad_token_is_rejected_clearly():
    tokenizer = _FakeTokenizer()
    tokenizer.pad_token_id = None

    with pytest.raises(ValueError, match="must define pad_token_id"):
        _LoaderHarness(tokenizer=tokenizer).encoder().encode(
            ["one", "one two"]
        )


def test_single_string_has_batch_size_one_and_preserves_exact_text():
    harness = _LoaderHarness()
    encoder = harness.encoder()

    batch = encoder.encode("  Exact Original Text  ")

    assert batch.batch_size == 1
    assert batch.texts == ("  Exact Original Text  ",)
    assert batch.input_ids.shape == (1, 5)
    assert batch.attention_mask.shape == (1, 5)
    assert batch.token_embeddings.shape == (1, 5, 3)
    assert batch.pooled_embeddings.shape == (1, 3)


def test_batched_strings_preserve_order_and_tensor_shapes():
    harness = _LoaderHarness()
    encoder = harness.encoder()

    batch = encoder.encode(["a", "alpha beta gamma"])

    assert batch.texts == ("a", "alpha beta gamma")
    assert batch.input_ids.shape == (2, 5)
    assert batch.attention_mask.shape == (2, 5)
    assert batch.token_embeddings.shape == (2, 5, 3)
    assert batch.pooled_embeddings.shape == (2, 3)
    assert batch.truncated == (False, False)


def test_masked_mean_pooling_is_numerically_correct():
    harness = _LoaderHarness()
    batch = harness.encoder().encode("a")

    expected = torch.tensor([[214 / 3, 428 / 3, 217 / 3]])

    torch.testing.assert_close(batch.pooled_embeddings, expected)


def test_padding_does_not_affect_pooled_embeddings():
    single_harness = _LoaderHarness()
    batch_harness = _LoaderHarness()

    single = single_harness.encoder().encode("a")
    padded = batch_harness.encoder().encode(["a", "alpha beta"])

    assert padded.attention_mask[0].tolist() == [1, 1, 1, 0]
    assert padded.token_embeddings[0, 3].tolist() == [0.0, 0.0, 1.0]
    torch.testing.assert_close(
        padded.pooled_embeddings[0],
        single.pooled_embeddings[0],
    )


def test_model_is_moved_to_device_evaluated_frozen_and_inference_only():
    harness = _LoaderHarness()
    encoder = harness.encoder(device="cpu")

    batch = encoder.encode("frozen encoder")

    assert harness.model.device_calls == [torch.device("cpu")]
    assert harness.model.eval_calls == 1
    assert not harness.model.training
    assert all(not parameter.requires_grad for parameter in harness.model.parameters())
    assert harness.model.inference_mode_flags == [True]
    assert batch.input_ids.device == encoder.device
    assert batch.attention_mask.device == encoder.device
    assert batch.token_embeddings.device == encoder.device
    assert batch.pooled_embeddings.device == encoder.device
    assert batch.input_ids.dtype is torch.int64
    assert batch.attention_mask.dtype is torch.int64
    assert batch.token_embeddings.dtype is torch.float32
    assert batch.pooled_embeddings.dtype is torch.float32
    assert not batch.token_embeddings.requires_grad
    assert not batch.pooled_embeddings.requires_grad


def test_one_model_output_supplies_token_and_pooled_states():
    harness = _LoaderHarness()

    batch = harness.encoder().encode("shared representation")

    assert harness.model.forward_calls == 1
    mask = batch.attention_mask.unsqueeze(-1).to(batch.token_embeddings.dtype)
    expected = (batch.token_embeddings * mask).sum(1) / mask.sum(1).clamp_min(1)
    torch.testing.assert_close(batch.pooled_embeddings, expected)


def test_truncation_flags_and_special_tokens_for_mixed_batch():
    harness = _LoaderHarness()
    encoder = harness.encoder(max_length=4)

    batch = encoder.encode(["one two", "one two three"])

    assert batch.truncated == (False, True)
    assert batch.input_ids.shape == (2, 4)
    assert batch.input_ids[0].tolist() == [101, 13, 13, 102]
    assert batch.input_ids[1].tolist() == [101, 13, 13, 102]
    assert len(harness.tokenizer.calls) == 1
    assert harness.model.forward_calls == 1


def test_untruncated_inputs_report_false():
    batch = _LoaderHarness().encoder(max_length=8).encode(["one", "one two"])

    assert batch.truncated == (False, False)


def test_max_length_one_is_rejected_for_minilm_special_tokens():
    with pytest.raises(ValueError, match="at least 2"):
        MiniLMFeatureEncoder(max_length=1)


def test_max_length_two_preserves_leading_and_terminal_special_tokens():
    harness = _LoaderHarness()

    batch = harness.encoder(max_length=2).encode("one two")

    assert batch.input_ids.tolist() == [[101, 102]]
    assert batch.attention_mask.tolist() == [[1, 1]]
    assert batch.truncated == (True,)
    assert harness.model.received_token_type_ids[0].tolist() == [[0, 3]]
    assert harness.model.forward_calls == 1


@pytest.mark.parametrize(
    "texts,error_type",
    [
        ([], ValueError),
        ((), ValueError),
        ("", ValueError),
        ("   ", ValueError),
        (["valid", ""], ValueError),
        (b"bytes", TypeError),
        (bytearray(b"bytes"), TypeError),
        (42, TypeError),
        (True, TypeError),
        (["valid", b"bytes"], TypeError),
        (["valid", 3], TypeError),
        (["valid", True], TypeError),
    ],
)
def test_invalid_encode_inputs_are_rejected_before_loading(texts, error_type):
    harness = _LoaderHarness()
    encoder = harness.encoder()

    with pytest.raises(error_type):
        encoder.encode(texts)

    assert harness.tokenizer_loads == []
    assert harness.model_loads == []


@pytest.mark.parametrize("max_length", [0, -1, True, False, "4", 4.0, None])
def test_invalid_max_length_values_are_rejected(max_length):
    with pytest.raises((TypeError, ValueError)):
        MiniLMFeatureEncoder(max_length=max_length)


def test_generator_input_is_explicitly_rejected_before_loading():
    harness = _LoaderHarness()
    encoder = harness.encoder()

    with pytest.raises(TypeError, match="sequence of strings"):
        encoder.encode(text for text in ["one", "two"])

    assert not encoder.is_loaded


@pytest.mark.parametrize(
    "field,replacement,error",
    [
        ("input_ids", torch.ones(3), "input_ids"),
        ("attention_mask", torch.ones((2, 2)), "attention_mask"),
        ("token_embeddings", torch.ones((1, 3)), "token_embeddings"),
        ("pooled_embeddings", torch.ones((2, 3)), "pooled_embeddings"),
        ("truncated", (), "truncated"),
        ("texts", (), "texts"),
    ],
)
def test_encoder_batch_validates_dimensions_and_batch_lengths(
    field,
    replacement,
    error,
):
    values = {
        "input_ids": torch.ones((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "token_embeddings": torch.ones((1, 3, 4)),
        "pooled_embeddings": torch.ones((1, 4)),
        "truncated": (False,),
        "texts": ("query",),
    }
    values[field] = replacement

    with pytest.raises((TypeError, ValueError), match=error):
        EncoderBatch(**values)


@pytest.mark.parametrize(
    "field,replacement,error",
    [
        ("input_ids", torch.ones((1, 3), dtype=torch.float32), "input_ids"),
        ("input_ids", torch.ones((1, 3), dtype=torch.bool), "input_ids"),
        (
            "attention_mask",
            torch.ones((1, 3), dtype=torch.float32),
            "attention_mask",
        ),
        (
            "token_embeddings",
            torch.ones((1, 3, 4), dtype=torch.int64),
            "token_embeddings",
        ),
        (
            "pooled_embeddings",
            torch.ones((1, 4), dtype=torch.int64),
            "pooled_embeddings",
        ),
        (
            "token_embeddings",
            torch.ones((1, 3, 4), dtype=torch.complex64),
            "token_embeddings",
        ),
        (
            "pooled_embeddings",
            torch.ones((1, 4), dtype=torch.complex64),
            "pooled_embeddings",
        ),
    ],
)
def test_encoder_batch_rejects_invalid_tensor_dtypes(field, replacement, error):
    values = {
        "input_ids": torch.ones((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.bool),
        "token_embeddings": torch.ones((1, 3, 4)),
        "pooled_embeddings": torch.ones((1, 4)),
        "truncated": (False,),
        "texts": ("query",),
    }
    values[field] = replacement

    with pytest.raises(TypeError, match=error):
        EncoderBatch(**values)


def test_dataclass_replace_reruns_encoder_batch_validation():
    batch = _LoaderHarness().encoder().encode("query")

    with pytest.raises(TypeError, match="input_ids"):
        replace(batch, input_ids=batch.input_ids.to(torch.float32))


def test_invalid_model_output_dimensions_are_rejected():
    harness = _LoaderHarness(model=_BadShapeTransformer())

    with pytest.raises(ValueError, match="last_hidden_state"):
        harness.encoder().encode("query")


def test_missing_model_last_hidden_state_is_rejected():
    harness = _LoaderHarness(model=_MissingOutputTransformer())

    with pytest.raises(ValueError, match="missing last_hidden_state"):
        harness.encoder().encode("query")


def test_nontensor_model_last_hidden_state_is_rejected():
    harness = _LoaderHarness(model=_NonTensorOutputTransformer())

    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        harness.encoder().encode("query")


@pytest.mark.parametrize(
    "model",
    [
        _MismatchedOutputTransformer(batch_delta=1),
        _MismatchedOutputTransformer(token_delta=1),
    ],
)
def test_mismatched_model_batch_or_token_dimensions_are_rejected(model):
    harness = _LoaderHarness(model=model)

    with pytest.raises(ValueError, match="dimensions must match"):
        harness.encoder().encode("query")


def test_model_output_hidden_size_must_match_configuration():
    harness = _LoaderHarness(
        model=_MismatchedOutputTransformer(hidden_size=4)
    )

    with pytest.raises(ValueError, match="hidden dimension"):
        harness.encoder().encode("query")


def test_encoder_batch_is_frozen_but_has_no_serialization_contract():
    batch = _LoaderHarness().encoder().encode("query")

    with pytest.raises(FrozenInstanceError):
        batch.texts = ("changed",)
    assert not hasattr(batch, "model_dump")
    assert not hasattr(batch, "model_dump_json")
    with pytest.raises(TypeError):
        json.dumps(batch)


def test_injected_loaders_avoid_transformers_and_network_runtime(monkeypatch):
    harness = _LoaderHarness()

    def fail_network(*args, **kwargs):
        raise AssertionError("network access is forbidden in encoder unit tests")

    monkeypatch.setattr("socket.create_connection", fail_network)
    batch = harness.encoder().encode("offline fake")

    assert batch.batch_size == 1
    assert harness.tokenizer_loads == [DEFAULT_MINILM_MODEL]
    assert harness.model_loads == [DEFAULT_MINILM_MODEL]
