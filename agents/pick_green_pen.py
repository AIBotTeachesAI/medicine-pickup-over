from typing import List
from brain_client.agent_types import Agent

class PickGreenPen(Agent):

    @property
    def id(self) -> str:
        return "pick_green_pen"

    @property
    def display_name(self) -> str:
        return "Pick Green Pen"

    def get_skills(self) -> List[str]:
        return [
            "navigate_to_position",
            "pick green pen",  # exactly matches "name" in metadata.json
        ]

    def get_inputs(self) -> List[str]:
        return ["micro"]

    def get_prompt(self) -> str:
        return """You are a manipulation robot. Your only job is to pick up the green pen.

## Procedure
1. Rotate slowly in place to scan the area around you.
2. Find the green pen.
3. Navigate toward it and stop directly in front of it.
4. Use the pick green pen skill to pick it up.
5. Say out loud: "I have picked up the green pen."

## Rules
- Only pick up the green pen, nothing else.
- Do NOT do anything after picking it up. Just stop and wait.
"""
