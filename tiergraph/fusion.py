"""Typed plans and results for the future Edge fusion component."""

from copy import deepcopy
from typing import Any, Literal, Mapping, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from tiergraph.enums import (
    ExecutionStatus,
    FusionStrategy,
    NodeSemanticType,
    OperatorType,
    SlotType,
    Tier,
)
from tiergraph.graph import ExecutionGraph
from tiergraph.models import TierGraphSchema


class _ValidatedFusionSchema(TierGraphSchema):
    """Revalidate fusion schema updates rather than copying unchecked state."""

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Copy the model, fully validating any supplied field updates."""
        if update is None:
            return super().model_copy(deep=deep)

        model_data = self.model_dump(mode="python", round_trip=True)
        update_data = dict(update)
        if deep:
            model_data = deepcopy(model_data)
            update_data = deepcopy(update_data)
        model_data.update(update_data)
        return type(self).model_validate(model_data)

    def _revalidated(self) -> Self:
        """Reconstruct this instance through normal Pydantic validation."""
        return type(self).model_validate(
            self.model_dump(mode="python", round_trip=True)
        )


class FusionPlan(_ValidatedFusionSchema):
    """Question-agnostic instructions for fusing typed slots on Edge."""

    schema_version: Literal["1.0"] = "1.0"
    plan_id: str
    graph_id: str
    fusion_node_id: str
    strategy: FusionStrategy
    required_slots: dict[str, SlotType]
    ordered_slots: tuple[str, ...]
    max_sentences: int = Field(gt=0)
    spoken_style: bool
    instructions: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("plan_id", "graph_id", "fusion_node_id", "instructions")
    @classmethod
    def _validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("required_slots")
    @classmethod
    def _validate_required_slots(
        cls, required_slots: dict[str, SlotType]
    ) -> dict[str, SlotType]:
        if not required_slots:
            raise ValueError("required_slots must not be empty")
        if any(not slot_name.strip() for slot_name in required_slots):
            raise ValueError("slot names must not be blank")
        return required_slots

    @field_validator("ordered_slots", mode="before")
    @classmethod
    def _normalize_json_ordered_slots(cls, value: object) -> object:
        if type(value) is list:
            return tuple(value)
        return value

    @field_validator("ordered_slots")
    @classmethod
    def _validate_ordered_slot_names(
        cls, ordered_slots: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not slot_name.strip() for slot_name in ordered_slots):
            raise ValueError("slot names must not be blank")
        if len(ordered_slots) != len(set(ordered_slots)):
            raise ValueError("ordered_slots must not contain duplicates")
        return ordered_slots

    @model_validator(mode="after")
    def _validate_slot_order(self) -> "FusionPlan":
        if set(self.ordered_slots) != set(self.required_slots):
            raise ValueError(
                "ordered_slots must be an exact permutation of required_slots"
            )
        return self

    @property
    def execution_tier(self) -> Tier:
        """The fixed execution tier for response fusion."""
        return Tier.EDGE

    def validate_against_graph(self, graph: ExecutionGraph) -> Self:
        """Validate this plan against its declared execution graph."""
        validated_plan = self._revalidated()

        if validated_plan.graph_id != graph.graph_id:
            raise ValueError("FusionPlan graph_id does not match ExecutionGraph")

        try:
            fusion_node = graph.node_by_id(validated_plan.fusion_node_id)
        except KeyError:
            raise ValueError(
                f"fusion node does not exist: {validated_plan.fusion_node_id}"
            ) from None

        if fusion_node.operator is not OperatorType.FUSE:
            raise ValueError("fusion_node_id must reference a FUSE node")
        if fusion_node.semantic_type is not NodeSemanticType.CONTROL:
            raise ValueError("the FUSE node must have control semantics")
        if fusion_node.tier is not Tier.EDGE:
            raise ValueError("the FUSE node must execute on Edge")
        if graph.successors(fusion_node.node_id):
            raise ValueError("the FUSE node must be terminal")
        if validated_plan.required_slots != fusion_node.required_inputs:
            raise ValueError(
                "required_slots must exactly match the FUSE node required_inputs"
            )
        return self


class FusionOutput(_ValidatedFusionSchema):
    """Serializable result emitted by the future Edge fusion component."""

    schema_version: Literal["1.0"] = "1.0"
    output_id: str
    plan_id: str
    graph_id: str
    fusion_node_id: str
    strategy: FusionStrategy
    status: ExecutionStatus
    text: str
    method: str
    evidence_ids: tuple[str, ...] = ()
    latency_ms: float = Field(ge=0.0)
    error: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("output_id", "plan_id", "graph_id", "fusion_node_id", "method")
    @classmethod
    def _validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _normalize_json_evidence_ids(cls, value: object) -> object:
        if type(value) is list:
            return tuple(value)
        return value

    @field_validator("evidence_ids")
    @classmethod
    def _validate_evidence_ids(
        cls, evidence_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not evidence_id.strip() for evidence_id in evidence_ids):
            raise ValueError("evidence IDs must not be blank")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must not contain duplicates")
        return evidence_ids

    @model_validator(mode="after")
    def _validate_status_contract(self) -> "FusionOutput":
        if self.status not in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
        }:
            raise ValueError(
                "FusionOutput status must be succeeded or failed"
            )
        if self.status is ExecutionStatus.SUCCEEDED:
            if not self.text.strip():
                raise ValueError("a succeeded fusion output requires nonblank text")
            if self.error is not None:
                raise ValueError("a succeeded fusion output must not contain an error")
        if self.status is ExecutionStatus.FAILED:
            if self.error is None or not self.error.strip():
                raise ValueError("a failed fusion output requires a nonblank error")
        return self

    @property
    def execution_tier(self) -> Tier:
        """The fixed execution tier for response fusion."""
        return Tier.EDGE

    def validate_against(
        self,
        plan: FusionPlan,
        graph: ExecutionGraph,
    ) -> Self:
        """Validate this output against its plan and execution graph."""
        validated_output = self._revalidated()
        validated_plan = plan._revalidated()
        validated_plan.validate_against_graph(graph)

        if validated_output.plan_id != validated_plan.plan_id:
            raise ValueError("FusionOutput plan_id does not match FusionPlan")
        if validated_output.graph_id != validated_plan.graph_id:
            raise ValueError("FusionOutput graph_id does not match FusionPlan")
        if validated_output.fusion_node_id != validated_plan.fusion_node_id:
            raise ValueError(
                "FusionOutput fusion_node_id does not match FusionPlan"
            )
        if validated_output.strategy is not validated_plan.strategy:
            raise ValueError("FusionOutput strategy does not match FusionPlan")
        return self
