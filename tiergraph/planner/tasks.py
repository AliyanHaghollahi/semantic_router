"""Versioned deterministic task-string templates for planner-generated nodes.

Task text is intentionally excluded from semantic graph exact-match. These
templates exist so generated ``ExecutionGraph`` nodes remain deterministic.
"""

from __future__ import annotations

from tiergraph.enums import OperatorType
from tiergraph.planner.naming import SlotNamingError, normalize_base_name


TASK_TEMPLATE_VERSION = "TASK_TEMPLATE_V1"


class TaskTemplateError(ValueError):
    """Raised when TASK_TEMPLATE_V1 cannot render a node task string."""


_ANSWER_TEMPLATES: dict[OperatorType, str] = {
    OperatorType.RESOLVE_PERSONAL: "Resolve the user's {name} identifier",
    OperatorType.RETRIEVE_PERSONAL: "Retrieve the personal {name} fact",
    OperatorType.IDENTIFY_ENVIRONMENTAL: "Identify the {name}",
    OperatorType.LOCATE_ENVIRONMENTAL: "Locate the resolved {name}",
    OperatorType.NAVIGATE_TO: "Navigate to the {name}",
    OperatorType.DESCRIBE_ENVIRONMENT: "Describe the {name}",
}


def render_answer_task(*, operator: OperatorType, base_name: str) -> str:
    """Render a deterministic task string for an answer operation."""
    try:
        template = _ANSWER_TEMPLATES[operator]
    except KeyError as exc:
        raise TaskTemplateError(
            f"TASK_TEMPLATE_V1 has no answer template for {operator.value}"
        ) from exc
    try:
        name = normalize_base_name(base_name)
    except SlotNamingError as exc:
        raise TaskTemplateError(str(exc)) from exc
    return template.format(name=name)


def render_fuse_task() -> str:
    """Render the deterministic FUSE task string."""
    return "Fuse the terminal answers"


__all__ = [
    "TASK_TEMPLATE_VERSION",
    "TaskTemplateError",
    "render_answer_task",
    "render_fuse_task",
]
