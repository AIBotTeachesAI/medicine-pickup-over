"""
FindObject skill for Innate Mars robot.

Queries Thor's perception server for a named object's 3D position,
then navigates there using Nav2.

Deploy: copy to ~/innate-os/skills/find_object_skill.py on Mars.
"""

import math
import os

import requests

from brain_client.skill_types import (
    Skill as Primitive,
    SkillResult as PrimitiveResult,
    Interface,
    InterfaceType,
)

THOR_URL = os.environ.get("THOR_URL", "http://172.17.30.90:8001")


class FindObjectSkill(Primitive):
    """Navigate to a remembered object by name."""

    mobility = Interface(InterfaceType.MOBILITY)

    @property
    def name(self):
        return "find_object"

    def guidelines(self):
        return (
            "Use this skill to navigate to a remembered object by name. "
            "The robot must have explored the room first (use explore_room). "
            "If the object is not found, try running explore_room with "
            "reset_scene=false to refresh spatial memory, then retry."
        )

    def execute(self, label: str = "medicine", standoff: float = 0.8):
        label = label.strip().lower()

        try:
            resp = requests.get(
                f"{THOR_URL}/mars/where_is",
                params={"label": label},
                timeout=10,
            )
            data = resp.json()
        except Exception as e:
            return f"Cannot reach Thor: {e}", PrimitiveResult.FAILURE

        if not data.get("found"):
            return (
                f"'{label}' not found in memory. Try mapping the room first.",
                PrimitiveResult.FAILURE,
            )

        x = data["x"]
        y = data["y"]
        conf = data.get("confidence", 0)
        n_obs = data.get("n_observations", 0)

        print(
            f"[find_object] '{label}' at ({x:.2f}, {y:.2f}) in map frame, "
            f"confidence={conf:.2f}, seen {n_obs} times"
        )

        # Offset target by standoff distance (stop 0.3m before the object)
        if standoff > 0:
            try:
                import requests as _req
                cache = _req.get("http://localhost:9999/latest", timeout=2).json()
                if cache.get("pose", {}).get("valid"):
                    rx = cache["pose"]["translation"][0]
                    ry = cache["pose"]["translation"][1]
                    dx, dy = x - rx, y - ry
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > standoff:
                        x = x - standoff * dx / dist
                        y = y - standoff * dy / dist
                        print(f"[find_object] standoff {standoff}m -> target ({x:.2f}, {y:.2f})")
            except Exception:
                pass  # fallback: navigate to exact position

        # Navigate to the object's position in map frame
        try:
            import sys as _sys
            _skills_dir = os.path.dirname(os.path.abspath(__file__))
            if _skills_dir not in _sys.path:
                _sys.path.insert(0, _skills_dir)
            from navigate_to_position import Nav2Controller
            from nav2_simple_commander.robot_navigator import TaskResult
            import logging as _logging; nav = Nav2Controller(_logging.getLogger("find_object"), self)
            result = nav.go_to_position(x, y, 0.0, local_frame=False)

            # Check if we're "close enough" even if Nav2 reports failure
            close_enough = False
            try:
                import requests as _req
                cache = _req.get("http://localhost:9999/latest", timeout=2).json()
                if cache.get("pose", {}).get("valid"):
                    rx = cache["pose"]["translation"][0]
                    ry = cache["pose"]["translation"][1]
                    dist_to_target = math.sqrt((x - rx)**2 + (y - ry)**2)
                    print(f"[find_object] distance to target: {dist_to_target:.2f}m")
                    close_enough = dist_to_target < 1.0  # within 1m = good enough
            except Exception:
                pass

            if result != TaskResult.SUCCEEDED and not close_enough:
                return f"Found '{label}' but navigation failed: {result}", PrimitiveResult.FAILURE
        except Exception as e:
            return f"Found '{label}' at ({x:.2f},{y:.2f}) but nav error: {e}", PrimitiveResult.FAILURE

        return (
            f"Arrived at '{label}' (confidence {conf:.0%}, seen {n_obs} times)",
            PrimitiveResult.SUCCESS,
        )

    def cancel(self):
        return "Find cancelled"
