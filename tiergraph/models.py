"""Strict, serializable execution result schemas for TierGraph."""

from typing import ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from tiergraph.enums import ExecutionStatus, SlotType, Tier


class TierGraphSchema(BaseModel):
    """Shared strict configuration for all TierGraph Pydantic models.

    Frozen models prevent attribute reassignment but do not deeply freeze
    nested mappings. Callers must treat required_inputs, produced_outputs,
    outputs, FusionPlan.required_slots, FusionPlan.metadata, and
    FusionOutput.metadata as immutable after model construction.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class EvidenceItem(TierGraphSchema):
    """Evidence attributable to a typed output slot of one execution node."""

    evidence_id: str
    graph_id: str
    node_id: str
    slot_name: str
    slot_type: SlotType
    tier: Tier
    value: JsonValue
    source: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("evidence_id", "graph_id", "node_id", "slot_name")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("source must not be blank when provided")
        return value


class TierResult(TierGraphSchema):
    """Serializable result emitted by one Edge or Fog execution node."""

    schema_version: Literal["1.0"] = "1.0"
    result_id: str
    graph_id: str
    node_id: str
    tier: Tier
    status: ExecutionStatus
    outputs: dict[str, JsonValue]
    evidence: tuple[EvidenceItem, ...] = ()
    latency_ms: float = Field(ge=0.0)
    error: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("result_id", "graph_id", "node_id")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("outputs")
    @classmethod
    def _validate_output_names(
        cls, outputs: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        if any(not name.strip() for name in outputs):
            raise ValueError("output slot names must not be blank")
        return outputs

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_json_evidence_array(cls, value: object) -> object:
        if type(value) is list:
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_status_and_error(self) -> "TierResult":
        if self.status is ExecutionStatus.SUCCEEDED and self.error is not None:
            raise ValueError("a succeeded result must not contain an error")
        if self.status is ExecutionStatus.FAILED:
            if self.error is None or not self.error.strip():
                raise ValueError("a failed result requires a nonblank error")
        return self
