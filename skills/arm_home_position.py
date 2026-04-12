#!/usr/bin/env python3
"""
Arm Home Position Skill - Move arm to its tucked home position.

This is the safe resting position where the arm is folded and out of the way.
Use this after any manipulation task (pick, place, handoff) to stow the arm
before driving.
"""

import time
from brain_client.skill_types import Skill, SkillResult, Interface, InterfaceType


# Measured home position: arm tucked with elbow folded in
HOME_JOINTS = [1.562, -1.887, 1.505, 1.201, -0.032, 0.041]


class ArmHomePosition(Skill):
    """Move the arm to its tucked home position (safe for driving)."""

    manipulation = Interface(InterfaceType.MANIPULATION)

    def __init__(self, logger):
        super().__init__(logger)
        self._cancelled = False

    @property
    def name(self):
        return "arm_home_position"

    def guidelines(self):
        return (
            "Move the robot arm to its tucked home position. Call this AFTER "
            "any pick or manipulation skill finishes, so the arm is safely "
            "stowed before the robot drives. Also use if the arm is in an "
            "awkward position and needs to be reset to a safe pose."
        )

    def execute(self, duration: int = 3):
        """Move arm to home position.

        Args:
            duration: Time in seconds for the motion (default 3).
        """
        self._cancelled = False

        if self.manipulation is None:
            return "Manipulation interface not available", SkillResult.FAILURE

        self.logger.info(
            f"Moving arm to home position {HOME_JOINTS} over {duration}s"
        )

        success = self.manipulation.move_to_joint_positions(
            joint_positions=HOME_JOINTS,
            duration=duration,
            blocking=False,
        )

        if not success:
            return "Failed to send arm home command", SkillResult.FAILURE

        start_time = time.time()
        while time.time() - start_time < duration:
            if self._cancelled:
                return "Arm motion cancelled", SkillResult.CANCELLED
            time.sleep(0.1)

        return "Arm is in home position, safe to drive", SkillResult.SUCCESS

    def cancel(self):
        self._cancelled = True
        return "Arm home motion cancelled"
