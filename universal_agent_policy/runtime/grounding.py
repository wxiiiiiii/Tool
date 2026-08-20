from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from universal_agent_policy.policy.action_space import AbstractAction


@dataclass(frozen=True)
class ToolSpec:
    slot: int
    name: str
    description: str
    parameters: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class GroundedAction:
    abstract_action: AbstractAction
    tool: ToolSpec | None
    reason: str = ""

    @property
    def calls_tool(self) -> bool:
        return self.tool is not None


def normalize_tool(tool: dict[str, Any], slot: int) -> ToolSpec:
    return ToolSpec(
        slot=slot,
        name=str(tool.get("name", f"tool_{slot}")),
        description=str(tool.get("description", "")),
        parameters=dict(tool.get("parameters", {})),
        raw=tool,
    )


class ActionGrounder:
    def ground(self, action: AbstractAction, tools: list[dict[str, Any]]) -> GroundedAction:
        if not action.calls_tool:
            return GroundedAction(action, None, "policy_selected_no_tool")
        assert action.tool_slot is not None
        if action.tool_slot >= len(tools):
            return GroundedAction(action, None, "selected_slot_missing_from_environment")
        return GroundedAction(action, normalize_tool(tools[action.tool_slot], action.tool_slot))
