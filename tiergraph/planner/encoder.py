"""Lazy, one-pass frozen transformer features for the TierGraph planner.

Importing this module imports PyTorch, but never imports Transformers, loads a
model, or performs network access. The default Hugging Face loaders are
invoked only by the first :meth:`MiniLMFeatureEncoder.encode` call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


DEFAULT_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_Loader = Callable[[str], Any]
_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        getattr(torch, "uint16", torch.uint8),
        getattr(torch, "uint32", torch.uint8),
        getattr(torch, "uint64", torch.uint8),
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)
_INTEGER_OR_BOOL_DTYPES = _INTEGER_DTYPES | {torch.bool}


@dataclass(frozen=True)
class EncoderBatch:
    """Token and pooled features produced by one frozen encoder pass.

    The container prevents field reassignment, but PyTorch tensors remain
    mutable objects and must be treated as immutable by downstream planner
    components.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_embeddings: torch.Tensor
    pooled_embeddings: torch.Tensor
    truncated: tuple[bool, ...]
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        tensors = (
            self.input_ids,
            self.attention_mask,
            self.token_embeddings,
            self.pooled_embeddings,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("EncoderBatch features must be torch.Tensor objects")
        if self.input_ids.dtype not in _INTEGER_DTYPES:
            raise TypeError("input_ids must use an integer tensor dtype")
        if self.attention_mask.dtype not in _INTEGER_OR_BOOL_DTYPES:
            raise TypeError(
                "attention_mask must use an integer or Boolean tensor dtype"
            )
        if not self.token_embeddings.is_floating_point():
            raise TypeError("token_embeddings must use a floating-point dtype")
        if not self.pooled_embeddings.is_floating_point():
            raise TypeError("pooled_embeddings must use a floating-point dtype")
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids")
        if self.token_embeddings.ndim != 3:
            raise ValueError(
                "token_embeddings must have shape [batch, tokens, hidden]"
            )
        if self.token_embeddings.shape[:2] != self.input_ids.shape:
            raise ValueError(
                "token_embeddings batch and token dimensions must match input_ids"
            )
        if self.pooled_embeddings.ndim != 2:
            raise ValueError(
                "pooled_embeddings must have shape [batch, hidden]"
            )
        expected_pooled_shape = (
            self.input_ids.shape[0],
            self.token_embeddings.shape[2],
        )
        if self.pooled_embeddings.shape != expected_pooled_shape:
            raise ValueError(
                "pooled_embeddings batch and hidden dimensions must match "
                "token_embeddings"
            )
        if len(self.truncated) != self.input_ids.shape[0]:
            raise ValueError("truncated must contain one flag per input")
        if any(type(value) is not bool for value in self.truncated):
            raise TypeError("truncated values must be booleans")
        if len(self.texts) != self.input_ids.shape[0]:
            raise ValueError("texts must contain one string per input")
        if not (
            self.input_ids.device
            == self.attention_mask.device
            == self.token_embeddings.device
            == self.pooled_embeddings.device
        ):
            raise ValueError("all EncoderBatch tensors must use the same device")

    @property
    def batch_size(self) -> int:
        """Number of original texts represented by this batch."""
        return self.input_ids.shape[0]


@runtime_checkable
class FeatureEncoder(Protocol):
    """Structural interface implemented by planner feature encoders."""

    def encode(self, texts: str | Sequence[str]) -> EncoderBatch:
        """Encode one string or a nonempty sequence of strings."""
        ...


class MiniLMFeatureEncoder:
    """Lazy frozen MiniLM encoder returning token and pooled representations.

    Manual truncation targets MiniLM/BERT-style tokenizers with a leading CLS
    token and one terminal SEP token. Tokenizer output is requested without
    padding, so original lengths are derived from its supplied attention mask.

    Concurrent first calls are not thread-safe in this phase. Callers must
    initialize an encoder from one thread before sharing it across threads.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MINILM_MODEL,
        max_length: int = 128,
        device: str | torch.device = "cpu",
        *,
        tokenizer_loader: _Loader | None = None,
        model_loader: _Loader | None = None,
    ) -> None:
        if type(model_name) is not str:
            raise TypeError("model_name must be a string")
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        if type(max_length) is not int:
            raise TypeError("max_length must be a strict integer")
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if tokenizer_loader is not None and not callable(tokenizer_loader):
            raise TypeError("tokenizer_loader must be callable")
        if model_loader is not None and not callable(model_loader):
            raise TypeError("model_loader must be callable")

        self._model_name = model_name
        self._max_length = max_length
        self._device = torch.device(device)
        self._tokenizer_loader = tokenizer_loader
        self._model_loader = model_loader
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._hidden_size: int | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether both tokenizer and model have been initialized."""
        return self._tokenizer is not None and self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def hidden_size(self) -> int:
        """Loaded model hidden size, available after the first encode call."""
        if self._hidden_size is None:
            raise RuntimeError("hidden_size is unavailable before model loading")
        return self._hidden_size

    def encode(self, texts: str | Sequence[str]) -> EncoderBatch:
        """Produce token and masked-mean features with one transformer call."""
        normalized_texts = _validate_texts(texts)
        self._ensure_loaded()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:  # Defensive type narrowing.
            raise RuntimeError("encoder loading did not initialize its dependencies")

        encoded = tokenizer(
            list(normalized_texts),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
        )
        model_inputs, input_ids, attention_mask, truncated = self._prepare_inputs(
            encoded,
            tokenizer,
            len(normalized_texts),
        )

        with torch.inference_mode():
            model_output = model(**model_inputs)

        token_embeddings = _last_hidden_state(model_output).detach()
        if token_embeddings.device != self._device:
            raise ValueError(
                "model output device does not match configured encoder device"
            )
        if token_embeddings.ndim != 3:
            raise ValueError(
                "model last_hidden_state must have shape [batch, tokens, hidden]"
            )
        if not token_embeddings.is_floating_point():
            raise TypeError("model last_hidden_state must use a floating-point dtype")
        if token_embeddings.shape[:2] != input_ids.shape:
            raise ValueError(
                "model token-state dimensions must match tokenized inputs"
            )

        output_hidden_size = token_embeddings.shape[2]
        if self._hidden_size is None:
            self._hidden_size = output_hidden_size
        elif self._hidden_size != output_hidden_size:
            raise ValueError(
                "model output hidden dimension does not match model configuration"
            )

        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        sum_embeddings = (token_embeddings * mask).sum(dim=1)
        token_count = mask.sum(dim=1).clamp_min(1)
        pooled_embeddings = (sum_embeddings / token_count).detach()

        return EncoderBatch(
            input_ids=input_ids.detach(),
            attention_mask=attention_mask.detach(),
            token_embeddings=token_embeddings,
            pooled_embeddings=pooled_embeddings,
            truncated=truncated,
            texts=normalized_texts,
        )

    def token_char_spans_for_batch(
        self,
        batch: EncoderBatch,
    ) -> tuple[tuple["TokenCharSpan", ...], ...]:
        """Build TokenCharSpan views aligned to ``batch`` token positions.

        Reuses this encoder's loaded tokenizer and the same truncation policy as
        :meth:`encode`. Token IDs are re-derived and must exactly match
        ``batch.input_ids`` on non-padding positions so planner alignment cannot
        drift from the embeddings that were actually computed.
        """
        from tiergraph.planner.align import TokenCharSpan

        if not self.is_loaded:
            raise RuntimeError(
                "token_char_spans_for_batch requires a loaded encoder; call encode first"
            )
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("encoder tokenizer is unavailable")
        if len(batch.texts) != batch.batch_size:
            raise ValueError("EncoderBatch texts length must match batch size")

        encoded = tokenizer(
            list(batch.texts),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        input_sequences = _batch_sequences(
            encoded["input_ids"],
            field_name="input_ids",
            batch_size=batch.batch_size,
        )
        attention_sequences = _attention_sequences(
            encoded["attention_mask"],
            batch_size=batch.batch_size,
        )
        if "offset_mapping" not in encoded:
            raise ValueError(
                "tokenizer must provide offset_mapping for planner alignment"
            )
        offset_sequences = _offset_sequences(
            encoded["offset_mapping"],
            batch_size=batch.batch_size,
        )
        special_ids = frozenset(getattr(tokenizer, "all_special_ids", ()))
        positions = tuple(
            _truncation_positions(
                sequence,
                attention_mask=mask,
                max_length=self._max_length,
                special_ids=special_ids,
            )
            for sequence, mask in zip(
                input_sequences,
                attention_sequences,
                strict=True,
            )
        )

        batch_ids = batch.input_ids.detach().cpu().tolist()
        batch_mask = batch.attention_mask.detach().cpu().tolist()
        views: list[tuple[TokenCharSpan, ...]] = []
        for example_index, (
            ids,
            offsets,
            selected,
            padded_ids,
            padded_mask,
        ) in enumerate(
            zip(
                input_sequences,
                offset_sequences,
                positions,
                batch_ids,
                batch_mask,
                strict=True,
            )
        ):
            retained_ids = [ids[index] for index in selected]
            retained_offsets = [offsets[index] for index in selected]
            attended = sum(1 for value in padded_mask if bool(value))
            if attended != len(retained_ids):
                raise ValueError(
                    "aligned token count does not match EncoderBatch attention_mask "
                    f"for example {example_index}"
                )
            if padded_ids[:attended] != retained_ids:
                raise ValueError(
                    "tokenizer offsets do not match EncoderBatch input_ids for "
                    f"example {example_index}; refuse to align mismatched tokens"
                )
            tokens: list[TokenCharSpan] = []
            for position, (token_id, offset) in enumerate(
                zip(retained_ids, retained_offsets, strict=True)
            ):
                is_special = token_id in special_ids
                start, end = offset
                if is_special or (start == 0 and end == 0):
                    tokens.append(
                        TokenCharSpan(
                            char_start=None,
                            char_end=None,
                            is_special=True,
                            is_padding=False,
                        )
                    )
                else:
                    tokens.append(
                        TokenCharSpan(
                            char_start=int(start),
                            char_end=int(end),
                            is_special=False,
                            is_padding=False,
                        )
                    )
            for _ in range(len(padded_ids) - attended):
                tokens.append(
                    TokenCharSpan(
                        char_start=None,
                        char_end=None,
                        is_special=False,
                        is_padding=True,
                    )
                )
            if len(tokens) != len(padded_ids):
                raise ValueError(
                    "token char span length must match EncoderBatch token length"
                )
            views.append(tuple(tokens))
        return tuple(views)

    def _ensure_loaded(self) -> None:
        """Load and freeze a complete pair before publishing instance state."""
        if self.is_loaded:
            return

        tokenizer_loader = self._tokenizer_loader or _load_huggingface_tokenizer
        model_loader = self._model_loader or _load_huggingface_model
        tokenizer = tokenizer_loader(self._model_name)
        model = model_loader(self._model_name)
        model.to(self._device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        configured_hidden_size = getattr(
            getattr(model, "config", None),
            "hidden_size",
            None,
        )
        if configured_hidden_size is not None:
            if type(configured_hidden_size) is not int or configured_hidden_size <= 0:
                raise ValueError("model config hidden_size must be a positive integer")
            self._hidden_size = configured_hidden_size

        self._tokenizer = tokenizer
        self._model = model

    def _prepare_inputs(
        self,
        encoded: Mapping[str, Any],
        tokenizer: Any,
        batch_size: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, tuple[bool, ...]]:
        """Validate and align token fields; ignore unsupported metadata keys.

        Only fields named by ``tokenizer.model_input_names`` are forwarded.
        Such optional fields must be token-aligned; unrelated scalar or batch
        metadata is ignored.
        """
        if not isinstance(encoded, Mapping):
            raise TypeError("tokenizer output must be a mapping")
        missing_fields = {
            field_name
            for field_name in ("input_ids", "attention_mask")
            if field_name not in encoded
        }
        if missing_fields:
            raise ValueError(
                "tokenizer output is missing required fields: "
                f"{sorted(missing_fields)}"
            )

        input_sequences = _batch_sequences(
            encoded["input_ids"],
            field_name="input_ids",
            batch_size=batch_size,
        )
        attention_sequences = _attention_sequences(
            encoded["attention_mask"],
            batch_size=batch_size,
        )
        for input_values, mask_values in zip(
            input_sequences,
            attention_sequences,
            strict=True,
        ):
            if len(input_values) != len(mask_values):
                raise ValueError(
                    "tokenizer attention_mask token lengths must match input_ids"
                )

        positions = tuple(
            _truncation_positions(
                sequence,
                attention_mask=mask,
                max_length=self._max_length,
                special_ids=frozenset(getattr(tokenizer, "all_special_ids", ())),
            )
            for sequence, mask in zip(
                input_sequences,
                attention_sequences,
                strict=True,
            )
        )
        truncated = tuple(
            sum(int(value) for value in mask) > self._max_length
            for mask in attention_sequences
        )
        clipped_ids = tuple(
            [sequence[index] for index in selected]
            for sequence, selected in zip(input_sequences, positions, strict=True)
        )
        clipped_attention = tuple(
            [mask[index] for index in selected]
            for mask, selected in zip(
                attention_sequences,
                positions,
                strict=True,
            )
        )
        if any(not sequence for sequence in clipped_ids):
            raise ValueError("tokenizer produced an empty attended token sequence")

        padded_length = max(len(sequence) for sequence in clipped_ids)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None and any(
            len(sequence) != padded_length for sequence in clipped_ids
        ):
            raise ValueError("tokenizer must define pad_token_id for batched padding")
        pad_token_id = 0 if pad_token_id is None else pad_token_id

        input_ids = torch.tensor(
            [
                sequence + [pad_token_id] * (padded_length - len(sequence))
                for sequence in clipped_ids
            ],
            dtype=torch.long,
            device=self._device,
        )
        attention_dtype = (
            torch.bool
            if all(
                type(value) is bool
                for sequence in attention_sequences
                for value in sequence
            )
            else torch.long
        )
        attention_mask = torch.tensor(
            [
                sequence + [0] * (padded_length - len(sequence))
                for sequence in clipped_attention
            ],
            dtype=attention_dtype,
            device=self._device,
        )
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        model_input_names = tuple(
            getattr(tokenizer, "model_input_names", ())
        )
        for field_name in model_input_names:
            if field_name in {"input_ids", "attention_mask"}:
                continue
            if field_name not in encoded:
                continue
            field_sequences = _batch_sequences(
                encoded[field_name],
                field_name=field_name,
                batch_size=batch_size,
            )
            clipped_field = []
            for values, selected, input_sequence in zip(
                field_sequences,
                positions,
                input_sequences,
                strict=True,
            ):
                if len(values) != len(input_sequence):
                    raise ValueError(
                        f"tokenizer {field_name} lengths must match input_ids"
                    )
                clipped_field.append([values[index] for index in selected])
            model_inputs[field_name] = torch.tensor(
                [
                    values + [0] * (padded_length - len(values))
                    for values in clipped_field
                ],
                dtype=torch.long,
                device=self._device,
            )

        return model_inputs, input_ids, attention_mask, truncated


def _validate_texts(texts: str | Sequence[str]) -> tuple[str, ...]:
    if type(texts) is str:
        values = (texts,)
    else:
        if isinstance(texts, (bytes, bytearray)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a string or a sequence of strings")
        values = tuple(texts)
    if not values:
        raise ValueError("texts must not be empty")
    for text in values:
        if type(text) is not str:
            raise TypeError("every text must be a Python string")
        if not text.strip():
            raise ValueError("texts must not contain blank strings")
    return values


def _batch_sequences(
    value: Any,
    *,
    field_name: str,
    batch_size: int,
) -> tuple[list[int], ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"tokenizer {field_name} must be a batch of sequences")
    if len(value) != batch_size:
        raise ValueError(f"tokenizer {field_name} batch size is incorrect")

    sequences: list[list[int]] = []
    for sequence in value:
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.detach().cpu().tolist()
        if not isinstance(sequence, Sequence) or isinstance(
            sequence,
            (str, bytes, bytearray),
        ):
            raise TypeError(f"tokenizer {field_name} entries must be sequences")
        values = list(sequence)
        if any(type(item) is not int for item in values):
            raise TypeError(f"tokenizer {field_name} entries must contain integers")
        sequences.append(values)
    return tuple(sequences)


def _attention_sequences(
    value: Any,
    *,
    batch_size: int,
) -> tuple[list[int | bool], ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("tokenizer attention_mask must be a batch of sequences")
    if len(value) != batch_size:
        raise ValueError("tokenizer attention_mask batch size must match input_ids")

    sequences: list[list[int | bool]] = []
    for sequence in value:
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.detach().cpu().tolist()
        if not isinstance(sequence, Sequence) or isinstance(
            sequence,
            (str, bytes, bytearray),
        ):
            raise TypeError("tokenizer attention_mask entries must be sequences")
        values = list(sequence)
        if any(
            type(item) not in {bool, int} or item not in {0, 1}
            for item in values
        ):
            raise ValueError(
                "tokenizer attention_mask entries must contain only Boolean "
                "or integer 0/1 values"
            )
        sequences.append(values)
    return tuple(sequences)


def _truncation_positions(
    input_ids: list[int],
    *,
    attention_mask: list[int | bool],
    max_length: int,
    special_ids: frozenset[int],
) -> tuple[int, ...]:
    attended_positions = tuple(
        index for index, value in enumerate(attention_mask) if bool(value)
    )
    if len(attended_positions) <= max_length:
        return attended_positions
    if (
        max_length >= 2
        and input_ids[attended_positions[0]] in special_ids
        and input_ids[attended_positions[-1]] in special_ids
    ):
        return (*attended_positions[: max_length - 1], attended_positions[-1])
    return attended_positions[:max_length]


def _offset_sequences(
    value: Any,
    *,
    batch_size: int,
) -> tuple[list[tuple[int, int]], ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("tokenizer offset_mapping must be a batch of sequences")
    if len(value) != batch_size:
        raise ValueError("tokenizer offset_mapping batch size must match input_ids")

    sequences: list[list[tuple[int, int]]] = []
    for sequence in value:
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.detach().cpu().tolist()
        if not isinstance(sequence, Sequence) or isinstance(
            sequence,
            (str, bytes, bytearray),
        ):
            raise TypeError("tokenizer offset_mapping entries must be sequences")
        pairs: list[tuple[int, int]] = []
        for item in sequence:
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().tolist()
            if not isinstance(item, Sequence) or isinstance(
                item,
                (str, bytes, bytearray),
            ):
                raise TypeError("offset_mapping entries must be (start, end) pairs")
            if len(item) != 2:
                raise ValueError("offset_mapping pairs must have length 2")
            start, end = item
            if type(start) is not int or type(end) is not int:
                raise TypeError("offset_mapping values must be integers")
            pairs.append((start, end))
        sequences.append(pairs)
    return tuple(sequences)


def _last_hidden_state(model_output: Any) -> torch.Tensor:
    missing = object()
    value = getattr(model_output, "last_hidden_state", missing)
    if value is missing and isinstance(model_output, Mapping):
        value = model_output.get("last_hidden_state", missing)
    if value is missing:
        raise ValueError("model output is missing last_hidden_state")
    if not isinstance(value, torch.Tensor):
        raise TypeError("model last_hidden_state must be a torch.Tensor")
    return value


def _load_huggingface_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _load_huggingface_model(model_name: str) -> Any:
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_name)


__all__ = [
    "DEFAULT_MINILM_MODEL",
    "EncoderBatch",
    "FeatureEncoder",
    "MiniLMFeatureEncoder",
]
