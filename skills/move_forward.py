# ~/skills/move_forward.py
import time
from brain_client.skill_types import Skill, SkillResult, Interface, InterfaceType

class MoveForward(Skill):
    mobility = Interface(InterfaceType.MOBILITY)

    @property
    def name(self):
        return "move_forward"

    def guidelines(self):
        return "Use when you need to move the robot forward a little bit."

    def execute(self, duration: float = 1.5, speed: float = 0.1):
        """Move forward.

        Args:
            duration: How long to move in seconds
            speed: Forward speed in m/s
        """
        self.mobility.send_cmd_vel(linear_x=speed, angular_z=0.0, duration=duration)
        time.sleep(duration + 0.2)
        # explicit stop
        self.mobility.send_cmd_vel(linear_x=0.0, angular_z=0.0, duration=0.1)
        return f"Moved forward for {duration}s", SkillResult.SUCCESS

    def cancel(self):
        self.mobility.send_cmd_vel(linear_x=0.0, angular_z=0.0, duration=0.1)
        return "Cancelled"
