from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from universal_agent_policy.runtime.argument_generation import ArgumentGenerator
from universal_agent_policy.runtime.constrained_decode import DecodeReport, JsonSchemaConstrainedDecoder
from universal_agent_policy.runtime.grounding import ActionGrounder, GroundedAction
from universal_agent_policy.runtime.policy_runtime import FrozenPolicyRuntime, PolicyPrediction


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    slot: int


@dataclass
class AgentDecision:
    prediction: PolicyPrediction
    grounded: GroundedAction
    argument_report: DecodeReport | None
    tool_call: ToolCall | None

    @property
    def calls_tool(self) -> bool:
        return self.tool_call is not None


class UniversalAgent:
    def __init__(
        self,
        policy: FrozenPolicyRuntime,
        argument_generator: ArgumentGenerator,
        grounder: ActionGrounder | None = None,
        decoder: JsonSchemaConstrainedDecoder | None = None,
    ) -> None:
        self.policy = policy
        self.argument_generator = argument_generator
        self.grounder = grounder or ActionGrounder()
        self.decoder = decoder or JsonSchemaConstrainedDecoder()

    def decide_from_hidden(self, row: dict[str, Any], h: torch.Tensor) -> AgentDecision:
        prediction = self.policy.predict_from_hidden(h)
        grounded = self.grounder.ground(prediction.action, row.get("tools", []))
        if grounded.tool is None:
            return AgentDecision(prediction, grounded, None, None)

        raw_args = self.argument_generator.generate(row, grounded.tool)
        report = self.decoder.decode(raw_args, grounded.tool.parameters)
        if not report.valid:
            return AgentDecision(prediction, grounded, report, None)

        call = ToolCall(grounded.tool.name, report.constrained, grounded.tool.slot)
        return AgentDecision(prediction, grounded, report, call)
