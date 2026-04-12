# ~/skills/find_green_marker_wave.py
import base64
import io
import time
import threading

import numpy as np
from PIL import Image

from brain_client.skill_types import (
    Skill,
    SkillResult,
    Interface,
    InterfaceType,
    RobotState,
    RobotStateType,
)

# Lazy globals so the model is loaded once and shared across skill invocations.
_MODEL = None
_PROCESSOR = None
_MODEL_LOCK = threading.Lock()
_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def _load_model():
    """Load Grounding DINO Tiny once, on demand. Pinned to CUDA in fp16."""
    global _MODEL, _PROCESSOR
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _PROCESSOR, _MODEL
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        _PROCESSOR = AutoProcessor.from_pretrained(_MODEL_ID)
        _MODEL = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                _MODEL_ID, torch_dtype=dtype
            )
            .to(device)
            .eval()
        )
        return _PROCESSOR, _MODEL


def _decode_b64_image(b64_str):
    """Convert a base64-encoded image string into a PIL RGB image."""
    if b64_str is None:
        return None
    if isinstance(b64_str, bytes):
        b64_str = b64_str.decode("ascii", errors="ignore")
    if "," in b64_str:  # strip data URL prefix if present
        b64_str = b64_str.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _is_green(pil_img, box_xyxy, min_ratio=0.25):
    """Confirm the bbox is actually green via HSV thresholding.

    Grounding DINO color grounding is unreliable; we trust it to find the
    marker, then verify color in pixel space.
    """
    import cv2  # opencv ships with the ROS Humble desktop install on mars

    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    w, h = pil_img.size
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return False

    crop = np.array(pil_img.crop((x1, y1, x2, y2)))  # RGB
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Green hue range in OpenCV HSV: ~35-85
    mask = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))
    ratio = float(mask.sum()) / 255.0 / max(mask.size, 1)
    return ratio >= min_ratio


def _detect_green_marker(processor, model, pil_img, box_threshold=0.30, text_threshold=0.25):
    """Run Grounding DINO Tiny and return any boxes that pass the green HSV check."""
    import torch

    text = "a green marker. a marker."  # Grounding DINO expects period-separated phrases
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    inputs = processor(images=pil_img, text=text, return_tensors="pt").to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[pil_img.size[::-1]],  # (h, w)
    )[0]

    confirmed = []
    for score, box, label in zip(results["scores"], results["boxes"], results["labels"]):
        box = box.detach().cpu().tolist()
        if _is_green(pil_img, box):
            confirmed.append((float(score), box, str(label)))
    return confirmed


class FindGreenMarkerWave(Skill):
    """Look around for a green marker. Wave when one is found."""

    mobility = Interface(InterfaceType.MOBILITY)
    manipulation = Interface(InterfaceType.MANIPULATION)
    head = Interface(InterfaceType.HEAD)
    image = RobotState(RobotStateType.LAST_MAIN_CAMERA_IMAGE_B64)

    @property
    def name(self):
        return "find_green_marker_wave"

    def guidelines(self):
        return (
            "Use when you want the robot to scan its surroundings for a green marker "
            "and wave hello when it finds one."
        )

    def execute(self, scan_steps: int = 8, dwell_seconds: float = 0.6):
        """Rotate in place, run Grounding DINO each step, wave on green marker.

        Args:
            scan_steps: How many discrete rotations to perform (full circle = 8).
            dwell_seconds: How long to pause after rotating before grabbing a frame.
        """
        import math

        self._cancelled = False

        # Warm the model up front so the first detection isn't slow.
        self._send_feedback("Loading Grounding DINO Tiny...")
        processor, model = _load_model()

        # Tilt head up slightly so the marker is in frame at human height.
        self.head.set_position(5)

        for i in range(scan_steps):
            if self._cancelled:
                return "Cancelled", SkillResult.CANCELLED

            self._send_feedback(f"Scanning {i + 1}/{scan_steps}")
            time.sleep(dwell_seconds)  # let the camera settle after rotating

            pil_img = _decode_b64_image(self.image)
            if pil_img is None:
                self._send_feedback("No camera frame yet")
            else:
                hits = _detect_green_marker(processor, model, pil_img)
                if hits:
                    best = max(hits, key=lambda h: h[0])
                    self._send_feedback(
                        f"Green marker detected (score={best[0]:.2f}) — waving"
                    )
                    self._wave()
                    return (
                        f"Found green marker (score={best[0]:.2f}) and waved",
                        SkillResult.SUCCESS,
                    )

            # Rotate 1/scan_steps of a full circle.
            self.mobility.rotate(2 * math.pi / scan_steps)

        return "No green marker found", SkillResult.SUCCESS

    def _wave(self):
        # Same wave pattern as the ScanAndWave example in the Innate docs.
        wave_left = [0.5, -0.3, 1.0, -0.5, 0.5, 0]
        wave_right = [0.5, -0.3, 1.0, -0.5, -0.5, 0]
        for _ in range(3):
            if self._cancelled:
                return
            self.manipulation.goto_joint_state(wave_left)
            self.manipulation.goto_joint_state(wave_right)

    def cancel(self):
        self._cancelled = True
        self.mobility.send_cmd_vel(linear_x=0.0, angular_z=0.0, duration=0.1)
        return "Cancelled"
