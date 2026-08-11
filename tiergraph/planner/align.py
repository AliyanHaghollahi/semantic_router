"""Deterministic character-span alignment onto retained MiniLM content tokens.

Alignment never clips a partially truncated gold span into a different target.
Unrepresentable spans are marked/masked and counted for evaluation.

Representability is based on retained content-token offset boundaries, not
full character coverage. Hugging Face / BERT-family offset mappings commonly
omit whitespace between tokens; whitespace gaps alone must not mark a
multi-word span as partially truncated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

# Token-level BIO labels for retained content tokens.
BIO_O = 0
BIO_B = 1
BIO_I = 2
# Loss ignore index for specials, padding, and masked supervision positions.
BIO_IGNORE = -100


class TruncationKind(str, Enum):
    """Why a gold span cannot supervise token labels."""

    NONE = "none"
    FULL = "full"
    PARTIAL = "partial"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class TokenCharSpan:
    """Character offsets for one tokenizer position in the original text.

    Special and padding tokens use ``char_start is None`` / ``char_end is None``.
    """

    char_start: int | None
    char_end: int | None
    is_special: bool
    is_padding: bool

    @property
    def is_content(self) -> bool:
        return (
            not self.is_special
            and not self.is_padding
            and self.char_start is not None
            and self.char_end is not None
            and self.char_end > self.char_start
        )


@dataclass(frozen=True, slots=True)
class SpanAlignment:
    """Alignment of one ``[start, end)`` character span onto a token sequence."""

    start: int
    end: int
    token_indices: tuple[int, ...]
    representable: bool
    truncation_kind: TruncationKind


@dataclass(frozen=True, slots=True)
class AlignmentStats:
    """Aggregate alignment / truncation counters for later evaluation."""

    n_spans: int = 0
    n_representable: int = 0
    n_fully_truncated: int = 0
    n_partially_truncated: int = 0
    n_empty: int = 0
    n_special_tokens: int = 0
    n_padding_tokens: int = 0
    n_content_tokens: int = 0

    def merge(self, other: "AlignmentStats") -> "AlignmentStats":
        return AlignmentStats(
            n_spans=self.n_spans + other.n_spans,
            n_representable=self.n_representable + other.n_representable,
            n_fully_truncated=self.n_fully_truncated + other.n_fully_truncated,
            n_partially_truncated=self.n_partially_truncated + other.n_partially_truncated,
            n_empty=self.n_empty + other.n_empty,
            n_special_tokens=self.n_special_tokens + other.n_special_tokens,
            n_padding_tokens=self.n_padding_tokens + other.n_padding_tokens,
            n_content_tokens=self.n_content_tokens + other.n_content_tokens,
        )


@dataclass(frozen=True, slots=True)
class BioEncoding:
    """Token BIO labels plus per-span alignment metadata."""

    labels: tuple[int, ...]
    spans: tuple[SpanAlignment, ...]
    stats: AlignmentStats


def content_token_indices(tokens: Sequence[TokenCharSpan]) -> tuple[int, ...]:
    """Indices of retained content tokens (excludes specials and padding)."""
    return tuple(
        index for index, token in enumerate(tokens) if token.is_content
    )


def covered_char_indices(tokens: Sequence[TokenCharSpan]) -> frozenset[int]:
    """Character offsets covered by retained content tokens."""
    covered: set[int] = set()
    for token in tokens:
        if not token.is_content:
            continue
        assert token.char_start is not None and token.char_end is not None
        covered.update(range(token.char_start, token.char_end))
    return frozenset(covered)


def align_char_span(
    start: int,
    end: int,
    tokens: Sequence[TokenCharSpan],
) -> SpanAlignment:
    """Align one half-open character span onto retained content tokens.

    A span is representable when every retained content token that intersects
    the span is kept for supervision and the span's semantic content lies
    inside the retained content-token frontier (min start .. max end over
    content tokens). Whitespace characters that fall between token offsets
    are ignored for coverage — they do not imply truncation.

    Partially or fully truncated spans are marked unrepresentable. Their
    ``token_indices`` are empty so callers cannot accidentally supervise a
    clipped subset.
    """
    if type(start) is not int or type(end) is not int:
        raise TypeError("span offsets must be strict integers")
    if start < 0 or end < 0:
        raise ValueError("span offsets must be non-negative")
    if end < start:
        raise ValueError("span end must be >= start")
    if end == start:
        return SpanAlignment(
            start=start,
            end=end,
            token_indices=(),
            representable=False,
            truncation_kind=TruncationKind.EMPTY,
        )

    content = []
    for index, token in enumerate(tokens):
        if not token.is_content:
            continue
        assert token.char_start is not None and token.char_end is not None
        content.append((index, token.char_start, token.char_end))

    if not content:
        return SpanAlignment(
            start=start,
            end=end,
            token_indices=(),
            representable=False,
            truncation_kind=TruncationKind.FULL,
        )

    frontier_start = min(char_start for _, char_start, _ in content)
    frontier_end = max(char_end for _, _, char_end in content)

    overlapping = [
        index
        for index, char_start, char_end in content
        if char_start < end and char_end > start
    ]

    # Span lies entirely after (or before) the retained content frontier.
    if end <= frontier_start or start >= frontier_end:
        return SpanAlignment(
            start=start,
            end=end,
            token_indices=(),
            representable=False,
            truncation_kind=TruncationKind.FULL,
        )

    if not overlapping:
        # Interior whitespace-only / uncovered gap with no overlapping token.
        # Treat as unrepresentable without inventing clipped supervision.
        return SpanAlignment(
            start=start,
            end=end,
            token_indices=(),
            representable=False,
            truncation_kind=TruncationKind.FULL,
        )

    # Truncation removes semantic content when the span extends past the
    # retained frontier while still overlapping some kept tokens.
    if start < frontier_start or end > frontier_end:
        return SpanAlignment(
            start=start,
            end=end,
            token_indices=(),
            representable=False,
            truncation_kind=TruncationKind.PARTIAL,
        )

    return SpanAlignment(
        start=start,
        end=end,
        token_indices=tuple(overlapping),
        representable=True,
        truncation_kind=TruncationKind.NONE,
    )


def align_char_spans(
    spans: Sequence[tuple[int, int]],
    tokens: Sequence[TokenCharSpan],
) -> tuple[tuple[SpanAlignment, ...], AlignmentStats]:
    """Align many character spans and aggregate truncation statistics."""
    aligned = tuple(align_char_span(start, end, tokens) for start, end in spans)
    stats = AlignmentStats(
        n_spans=len(aligned),
        n_representable=sum(1 for item in aligned if item.representable),
        n_fully_truncated=sum(
            1 for item in aligned if item.truncation_kind is TruncationKind.FULL
        ),
        n_partially_truncated=sum(
            1
            for item in aligned
            if item.truncation_kind is TruncationKind.PARTIAL
        ),
        n_empty=sum(
            1 for item in aligned if item.truncation_kind is TruncationKind.EMPTY
        ),
        n_special_tokens=sum(1 for token in tokens if token.is_special),
        n_padding_tokens=sum(1 for token in tokens if token.is_padding),
        n_content_tokens=sum(1 for token in tokens if token.is_content),
    )
    return aligned, stats


def encode_bio_labels(
    spans: Sequence[SpanAlignment],
    tokens: Sequence[TokenCharSpan],
    *,
    allow_token_conflicts: bool = False,
) -> BioEncoding:
    """Assign deterministic B/I/O labels over content tokens.

    Special and padding positions are ``BIO_IGNORE``. Unrepresentable spans are
    omitted from label supervision (their truncated status remains in stats).
    """
    labels = [
        BIO_IGNORE
        if (token.is_special or token.is_padding or not token.is_content)
        else BIO_O
        for token in tokens
    ]
    claimed: dict[int, int] = {}
    span_tuple = tuple(spans)

    for span_index, span in enumerate(span_tuple):
        if not span.representable or not span.token_indices:
            continue
        for offset, token_index in enumerate(span.token_indices):
            label = BIO_B if offset == 0 else BIO_I
            previous = claimed.get(token_index)
            if previous is not None and previous != span_index:
                if not allow_token_conflicts:
                    raise ValueError(
                        "BIO span token conflict at position "
                        f"{token_index}: spans {previous} and {span_index}"
                    )
                continue
            claimed[token_index] = span_index
            labels[token_index] = label

    stats = AlignmentStats(
        n_spans=len(span_tuple),
        n_representable=sum(1 for item in span_tuple if item.representable),
        n_fully_truncated=sum(
            1 for item in span_tuple if item.truncation_kind is TruncationKind.FULL
        ),
        n_partially_truncated=sum(
            1
            for item in span_tuple
            if item.truncation_kind is TruncationKind.PARTIAL
        ),
        n_empty=sum(
            1 for item in span_tuple if item.truncation_kind is TruncationKind.EMPTY
        ),
        n_special_tokens=sum(1 for token in tokens if token.is_special),
        n_padding_tokens=sum(1 for token in tokens if token.is_padding),
        n_content_tokens=sum(1 for token in tokens if token.is_content),
    )
    return BioEncoding(labels=tuple(labels), spans=span_tuple, stats=stats)


def build_token_char_spans(
    *,
    offset_mapping: Sequence[tuple[int, int] | None],
    special_mask: Sequence[bool],
    padding_mask: Sequence[bool],
) -> tuple[TokenCharSpan, ...]:
    """Construct token views from tokenizer offset mapping and masks."""
    if not (len(offset_mapping) == len(special_mask) == len(padding_mask)):
        raise ValueError("offset_mapping and masks must have equal length")
    tokens: list[TokenCharSpan] = []
    for offsets, is_special, is_padding in zip(
        offset_mapping,
        special_mask,
        padding_mask,
        strict=True,
    ):
        if is_padding or is_special or offsets is None:
            tokens.append(
                TokenCharSpan(
                    char_start=None,
                    char_end=None,
                    is_special=bool(is_special),
                    is_padding=bool(is_padding),
                )
            )
            continue
        start, end = offsets
        if type(start) is not int or type(end) is not int:
            raise TypeError("offset_mapping entries must be integer pairs")
        if end < start:
            raise ValueError("offset_mapping end must be >= start")
        # HF uses (0, 0) for specials even when special_mask is incomplete.
        if start == 0 and end == 0:
            tokens.append(
                TokenCharSpan(
                    char_start=None,
                    char_end=None,
                    is_special=True,
                    is_padding=False,
                )
            )
            continue
        tokens.append(
            TokenCharSpan(
                char_start=start,
                char_end=end,
                is_special=False,
                is_padding=False,
            )
        )
    return tuple(tokens)


__all__ = [
    "BIO_B",
    "BIO_I",
    "BIO_IGNORE",
    "BIO_O",
    "AlignmentStats",
    "BioEncoding",
    "SpanAlignment",
    "TokenCharSpan",
    "TruncationKind",
    "align_char_span",
    "align_char_spans",
    "build_token_char_spans",
    "content_token_indices",
    "covered_char_indices",
    "encode_bio_labels",
]
