from __future__ import annotations

from dataclasses import dataclass


NO_TOOL = "NO_TOOL"
TOOL_PREFIX = "SELECT_TOOL_"
ANSWER = "ANSWER"
ACT_PREFIX = "ACT_"


@dataclass(frozen=True)
class AbstractAction:
    action_id: int
    name: str
    tool_slot: int | None
    intent: str | None = None

    @property
    def calls_tool(self) -> bool:
        return self.tool_slot is not None or self.name.startswith(ACT_PREFIX)


class ActionSpace:
    """Named action vocabulary used by frozen policy checkpoints."""

    def __init__(self, names: list[str]) -> None:
        if not names:
            raise ValueError("action space must contain at least one action")
        self.names = list(names)

    def __len__(self) -> int:
        return len(self.names)

    def action_from_id(self, action_id: int) -> AbstractAction:
        if action_id < 0 or action_id >= len(self.names):
            raise ValueError(f"action_id {action_id} is outside action space")
        name = self.names[action_id]
        if name == NO_TOOL:
            return AbstractAction(action_id, name, None, None)
        if name.startswith(TOOL_PREFIX):
            return AbstractAction(action_id, name, int(name.removeprefix(TOOL_PREFIX)), None)
        return AbstractAction(action_id, name, None, name)

    def action_from_name(self, name: str) -> AbstractAction:
        if name not in self.names:
            raise ValueError(f"Unknown action name: {name}")
        return self.action_from_id(self.names.index(name))


class SlotActionSpace:
    """Universal slot-level action space shared by all backbones.

    BFCL provides a fresh tool list per prompt, so the reusable policy predicts
    abstract slots rather than global function names. Runtime grounding maps
    SELECT_TOOL_i to the i-th concrete tool in the current environment.
    """

    def __init__(self, max_tool_slots: int) -> None:
        if max_tool_slots < 1:
            raise ValueError("max_tool_slots must be >= 1")
        self.max_tool_slots = max_tool_slots
        self.names = [NO_TOOL] + [f"{TOOL_PREFIX}{idx}" for idx in range(max_tool_slots)]

    def __len__(self) -> int:
        return len(self.names)

    def action_from_id(self, action_id: int) -> AbstractAction:
        if action_id < 0 or action_id >= len(self.names):
            raise ValueError(f"action_id {action_id} is outside action space")
        if action_id == 0:
            return AbstractAction(0, NO_TOOL, None, None)
        return AbstractAction(action_id, self.names[action_id], action_id - 1, None)

    def action_from_name(self, name: str) -> AbstractAction:
        if name == NO_TOOL:
            return self.action_from_id(0)
        if not name.startswith(TOOL_PREFIX):
            raise ValueError(f"Unknown action name: {name}")
        slot = int(name.removeprefix(TOOL_PREFIX))
        return self.action_from_id(slot + 1)

    @classmethod
    def from_num_actions(cls, num_actions: int) -> "SlotActionSpace":
        if num_actions < 2:
            raise ValueError("num_actions must include NO_TOOL and at least one tool slot")
        return cls(max_tool_slots=num_actions - 1)
