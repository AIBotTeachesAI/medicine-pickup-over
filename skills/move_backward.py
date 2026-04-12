# ~/skills/move_backward.py
import time
from brain_client.skill_types import Skill, SkillResult, Interface, InterfaceType

class MoveBackward(Skill):
    mobility = Interface(InterfaceType.MOBILITY)

    @property
    def name(self):
        return "move_backward"

    def guidelines(self):
        return "Use when you need to move the robot backward a little bit."

    def execute(self, duration: float = 1.5, speed: float = 0.1):
        """Move backward.

        Args:
            duration: How long to move in seconds
            speed: Backward speed magnitude in m/s (always positive; sign is applied internally)
        """
        self.mobility.send_cmd_vel(linear_x=-abs(speed), angular_z=0.0, duration=duration)
        time.sleep(duration + 0.2)
        # explicit stop
        self.mobility.send_cmd_vel(linear_x=0.0, angular_z=0.0, duration=0.1)
        return f"Moved backward for {duration}s", SkillResult.SUCCESS

    def cancel(self):
        self.mobility.send_cmd_vel(linear_x=0.0, angular_z=0.0, duration=0.1)
        return "Cancelled"
