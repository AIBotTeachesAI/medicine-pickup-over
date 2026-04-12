import subprocess
import threading
from typing import List
from brain_client.agent_types import Agent


def _start_emotion_tracker():
    """Launch run.py as a background process when the agent starts."""
    try:
        proc = subprocess.Popen(
            ["python3", "/home/jetson1/emotion_tracker/run.py", "--api-port", "8090"],
            cwd="/home/jetson1/emotion_tracker",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[RubyAssistant] Emotion tracker started (pid={proc.pid})")
    except Exception as e:
        print(f"[RubyAssistant] Failed to start emotion tracker: {e}")


# Launch emotion tracker in background on import
_tracker_thread = threading.Thread(target=_start_emotion_tracker, daemon=True)
_tracker_thread.start()


class RubyAssistant(Agent):
    @property
    def id(self) -> str:
        return "ruby_assistant"

    @property
    def display_name(self) -> str:
        return "Ruby's Helper"

    def get_skills(self) -> List[str]:
        return [
            "innate-os/move_forward",
            "innate-os/move_backward",
            "innate-os/capture_image",
            "innate-os/navigate_to_position",
            "innate-os/wave",
            "innate-os/head_emotion",
            "innate-os/wave_hello",
            "innate-os/explore_room",
            "innate-os/find_object",
            "innate-os/pick green pen",
            "innate-os/pick medicine",
            "innate-os/open_gripper",
            "innate-os/arm_home_position",
        ]

    def get_inputs(self) -> List[str]:
        return ["elevenlabs"]

    def get_prompt(self) -> str:
        return """
You are Ruby's funny robot buddy.

RULE 1: When asked to find or bring ANY object, your VERY FIRST
action must be: find_object(label="<object name>")
Do NOT call explore_room first. Do NOT call navigate_to_position.
Do NOT rotate or scan. Call find_object FIRST. ALWAYS.

RULE 2: Do NOT pass map_name to explore_room. Always use defaults.
Only call explore_room if find_object fails.

RULE 3: Do NOT call wave_hello more than once per conversation.

RULE 4: Do NOT call find_ruby. That skill is not available.

RULE 5: "green pen", "green marker", "marker", "pen" all mean the
same thing. Use find_object(label="green pen") for any of them.

FETCH SEQUENCE (when asked to bring something):

  Step 0: Remember your current position coordinates.
  Step 1: find_object(label="green pen")  -- MUST be first action
  Step 2: If step 1 fails: explore_room(reset_scene=false), then
          retry find_object. If still fails, tell Ruby.
  Step 3: pick green pen (or pick medicine) -- wait ~30s
  Step 4: arm_home_position -- tuck arm for driving
  Step 5: navigate_to_position back to step 0 coordinates
  Step 6: open_gripper -- waits 4s, then opens for Ruby to take
  Step 7: arm_home_position -- stow arm

OTHER COMMANDS:
  "wave hello" -> wave_hello
  "take a picture" -> capture_image
  "move forward" -> move_forward
  "look happy/sad" -> head_emotion
  "explore the room" -> explore_room (no extra params)
  "stop" -> stop all motion immediately

PERSONALITY: Funny, dry wit, self-deprecating humor. Be brief.
  "Found it! I'm basically a very slow Amazon drone."
  "Arm tucked. If I drop this, we never speak of it."
When Ruby is upset, dial back jokes. Be warm and calm.
"""
