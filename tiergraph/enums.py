"""Core enum types for TierGraph schemas."""

from enum import Enum
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class _CanonicalWireEnum:
    """Normalize only exact enum wire strings before strict validation."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        strict_enum_schema = handler(source_type)

        def normalize_wire_value(value: Any) -> Any:
            if type(value) is str:
                try:
                    return cls(value)
                except ValueError:
                    return value
            return value

        return core_schema.no_info_before_validator_function(
            normalize_wire_value,
            strict_enum_schema,
        )


class QueryType(_CanonicalWireEnum, str, Enum):
    """Classification of the original user query."""

    PERSONAL = "Personal"
    ENVIRONMENTAL = "Environmental"
    MIXED = "Mixed"


class NodeSemanticType(_CanonicalWireEnum, str, Enum):
    """Semantic domain of an executable graph node."""

    PERSONAL = "personal"
    ENVIRONMENTAL = "environmental"
    CONTROL = "control"


class SlotType(_CanonicalWireEnum, str, Enum):
    """Semantic type of a value produced or consumed by a graph node."""

    RESOLVED_REFERENCE = "RESOLVED_REFERENCE"
    PERSONAL_FACT = "PERSONAL_FACT"
    PERSONAL_RECORD = "PERSONAL_RECORD"
    ENVIRONMENTAL_FACT = "ENVIRONMENTAL_FACT"
    LOCATION = "LOCATION"
    NAVIGATION_INSTRUCTION = "NAVIGATION_INSTRUCTION"
    SCENE_DESCRIPTION = "SCENE_DESCRIPTION"
    FINAL_RESPONSE = "FINAL_RESPONSE"


class Tier(_CanonicalWireEnum, str, Enum):
    """Execution tiers supported by the first TierGraph implementation."""

    EDGE = "edge"
    FOG = "fog"


class OperatorType(_CanonicalWireEnum, str, Enum):
    """Semantic answer-producing operations represented in TierGraph."""

    RESOLVE_PERSONAL = "RESOLVE_PERSONAL"
    RETRIEVE_PERSONAL = "RETRIEVE_PERSONAL"
    IDENTIFY_ENVIRONMENTAL = "IDENTIFY_ENVIRONMENTAL"
    LOCATE_ENVIRONMENTAL = "LOCATE_ENVIRONMENTAL"
    NAVIGATE_TO = "NAVIGATE_TO"
    DESCRIBE_ENVIRONMENT = "DESCRIBE_ENVIRONMENT"
    FUSE = "FUSE"


class ExecutionStatus(_CanonicalWireEnum, str, Enum):
    """Lifecycle state of an execution result."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TransferPolicy(_CanonicalWireEnum, str, Enum):
    """Policy controlling how a dependency value crosses tiers."""

    DIRECT = "direct"
    MINIMAL_REFERENCE = "minimal_reference"


class FusionStrategy(_CanonicalWireEnum, str, Enum):
    """Strategy selected for Edge response fusion."""

    CONCATENATE = "concatenate"
    TEMPLATE = "template"
    SLM = "slm"
    VALIDATED_SLM = "validated_slm"
