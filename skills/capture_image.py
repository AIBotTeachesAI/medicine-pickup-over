# ~/skills/capture_image.py
import base64
import os
import time
from datetime import datetime
from pathlib import Path

from brain_client.skill_types import (
    Skill,
    SkillResult,
    Interface,
    InterfaceType,
    RobotState,
    RobotStateType,
)


class CaptureImage(Skill):
    """Capture the latest frame from the main camera and save it to disk."""

    image = RobotState(RobotStateType.LAST_MAIN_CAMERA_IMAGE_B64)

    @property
    def name(self):
        return "capture_image"

    def guidelines(self):
        return (
            "Use to take a photo with the robot's main (front) camera. "
            "Saves the latest frame as a timestamped JPEG and returns the file path. "
            "Trigger when the user says things like 'take a picture', 'snap a photo', "
            "or 'capture what you see'."
        )

    def execute(self, save_dir: str = "~/captures"):
        """Capture a single frame and write it to ``save_dir``.

        Args:
            save_dir: Directory to save the image into. Defaults to ``~/captures``.
        """
        # Give the camera a tick to deliver a fresh frame after the skill starts.
        time.sleep(0.1)

        b64 = self.image
        if b64 is None:
            return "No camera frame available yet", SkillResult.FAILURE

        if isinstance(b64, bytes):
            b64 = b64.decode("ascii", errors="ignore")
        if "," in b64:  # strip data URL prefix if present
            b64 = b64.split(",", 1)[1]

        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            return f"Failed to decode image: {e}", SkillResult.FAILURE

        out_dir = Path(os.path.expanduser(save_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        file_path = out_dir / f"capture_{timestamp}.jpg"

        try:
            file_path.write_bytes(raw)
        except Exception as e:
            return f"Failed to save image: {e}", SkillResult.FAILURE

        size_kb = len(raw) / 1024.0
        return (
            f"Captured image saved to {file_path} ({size_kb:.0f} KB)",
            SkillResult.SUCCESS,
        )

    def cancel(self):
        return "Cancelled"
