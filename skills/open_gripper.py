#!/usr/bin/env python3
import time
from brain_client.skill_types import Skill, SkillResult, Interface, InterfaceType

class OpenGripper(Skill):

    manipulation = Interface(InterfaceType.MANIPULATION)

    def __init__(self, logger):
        super().__init__(logger)
        self._cancelled = False

    @property
    def name(self):
        return "open_gripper"

    def guidelines(self):
        return (
            "Open the robot gripper to release an object. Call this after a "
            "pick skill to hand over the object. Waits dwell_before seconds "
            "before opening (default 4s), then holds open for dwell_after "
            "seconds (default 2s). percent=100 is fully open (default)."
        )

    def execute(self, percent: float = 100.0, dwell_before: float = 4.0, dwell_after: float = 2.0):
        if self.manipulation is None:
            return "Manipulation interface not available", SkillResult.FAILURE

        if dwell_before > 0:
            time.sleep(dwell_before)

        self.manipulation.open_gripper(percent)
        time.sleep(0.5)
        self.manipulation.open_gripper(percent)

        if dwell_after > 0:
            time.sleep(dwell_after)

        return f"Gripper opened to {percent}%", SkillResult.SUCCESS

    def cancel(self):
        return "Cannot cancel gripper open"
