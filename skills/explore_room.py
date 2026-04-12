"""
ExploreRoom skill for Innate Mars robot.

Visits waypoints covering the current map. At each position and while driving,
captures RGB + depth + pose and POSTs to Thor's perception server for object
detection and 3D scene graph building.

Waypoints are generated dynamically from the occupancy grid map, ensuring
full coverage of free space regardless of which map is loaded.

Dependencies on Mars (already present):
    - brain_client (Innate SDK)
    - requests, numpy, cv2, yaml (pip / system)

The skill does NOT run any ML — all perception happens on Thor.
"""

import base64
import math
import os
import threading
import time
import traceback

import cv2
import numpy as np
import requests
import yaml

from brain_client.skill_types import (
    Skill as Primitive,
    SkillResult as PrimitiveResult,
    Interface,
    InterfaceType,
    RobotState,
    RobotStateType,
)

# Thor perception server
THOR_URL = os.environ.get("THOR_URL", "http://172.17.30.90:8001")

# Local depth cache (mars_depth_cache.py running on same machine)
DEPTH_CACHE_URL = os.environ.get("DEPTH_CACHE_URL", "http://localhost:9999")

# Maps directory
MAPS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps"
)

# Default map for waypoint generation
DEFAULT_MAP = "RubysHome"

CAPTURE_INTERVAL = 0.5  # seconds between captures while driving
SCAN_ANGLES = 4  # 360° scan stops (camera FOV ~104°, 4×90° gives full coverage)

# Minimum movement before sending a transit frame to Thor
# Reduces redundant captures on straight paths (~40% fewer HTTP requests)
MIN_CAPTURE_DISTANCE = 0.3  # meters
MIN_CAPTURE_ANGLE = 0.26    # radians (~15°)

# Fallback waypoints if map loading fails (Blue1 map, legacy)
_FALLBACK_WAYPOINTS = [
    (-5.00, +1.96),
    (-5.00, +0.96),
    (-6.00, -0.04),
    (-5.00, -0.04),
    (-7.00, -2.04),
]


# ─── Waypoint generation from occupancy grid ─────────────────


def _generate_waypoints(map_name=DEFAULT_MAP, grid_spacing=1.5, wall_margin=0.4):
    """
    Generate navigation waypoints from an occupancy grid map.

    1. Load the PGM occupancy grid + YAML metadata
    2. Identify explored free-space cells (excludes unknown/gray areas)
    3. Erode free space by wall_margin to keep robot away from obstacles
    4. Sample waypoints on a regular grid at grid_spacing intervals
    5. Order by nearest-neighbor from the robot's current position

    Args:
        map_name: Name of the map (without extension) in the maps/ directory.
        grid_spacing: Distance between waypoints in meters (default 1.5m).
        wall_margin: Minimum distance from walls in meters (default 0.4m).

    Returns:
        List of (x, y) tuples in map frame coordinates.
    """
    yaml_path = os.path.join(MAPS_DIR, f"{map_name}.yaml")
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    pgm_path = os.path.join(MAPS_DIR, config["image"])
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read map image: {pgm_path}")

    resolution = float(config["resolution"])
    origin = config["origin"]  # [x, y, theta]
    h, w = img.shape

    # In trinary mode, the map has three pixel values:
    #   ~254 (white) = explored free space
    #   ~205 (gray)  = unknown / unexplored
    #   ~0   (black) = occupied / walls
    # Only generate waypoints in explored free space (pixel > 220).
    free_mask = (img > 220).astype(np.uint8) * 255

    # Erode free space to keep waypoints away from walls and obstacles
    margin_pixels = int(wall_margin / resolution)
    if margin_pixels > 0:
        kernel_size = 2 * margin_pixels + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        free_mask = cv2.erode(free_mask, kernel)

    # Sample waypoints on a regular grid
    grid_pixels = max(1, int(grid_spacing / resolution))
    waypoints = []
    for row in range(grid_pixels // 2, h, grid_pixels):
        for col in range(grid_pixels // 2, w, grid_pixels):
            if free_mask[row, col] > 0:
                x = origin[0] + col * resolution
                y = origin[1] + (h - 1 - row) * resolution
                waypoints.append((round(x, 2), round(y, 2)))

    return waypoints


def _nearest_neighbor_order(waypoints, start_xy=None):
    """Order waypoints by nearest-neighbor heuristic starting from start_xy."""
    if len(waypoints) <= 1:
        return list(waypoints)

    remaining = list(waypoints)
    ordered = []

    # Use start_xy as the initial position (don't add it to the output)
    if start_xy is not None:
        current = start_xy
    else:
        current = remaining.pop(0)
        ordered.append(current)

    while remaining:
        best_idx = 0
        best_dist = float("inf")
        for i, wp in enumerate(remaining):
            d = (wp[0] - current[0]) ** 2 + (wp[1] - current[1]) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i
        current = remaining.pop(best_idx)
        ordered.append(current)

    return ordered


def _get_robot_position():
    """Get current robot position from depth cache, or None."""
    try:
        cache = requests.get(f"{DEPTH_CACHE_URL}/latest", timeout=2).json()
        if cache.get("pose", {}).get("valid"):
            return (
                cache["pose"]["translation"][0],
                cache["pose"]["translation"][1],
            )
    except Exception:
        pass
    return None


# ─── Skill class ─────────────────────────────────────────────


class ExploreRoomSkill(Primitive):
    """Roam the room visiting waypoints, capturing frames for Thor to build a 3D object map."""

    image = RobotState(RobotStateType.LAST_MAIN_CAMERA_IMAGE_B64)
    mobility = Interface(InterfaceType.MOBILITY)

    @property
    def name(self):
        return "explore_room"

    def guidelines(self):
        return (
            "Explore the room to build spatial memory of objects. The robot "
            "drives to waypoints across the map, capturing images and depth "
            "data for the perception server. After this finishes, use "
            "find_object to navigate to any detected object by name. "
            "Set reset_scene=false to keep existing detections and add new "
            "ones (incremental). Set reset_scene=true (default) to start "
            "fresh. This is the FIRST step before find_object can work."
        )

    def execute(
        self,
        map_name: str = DEFAULT_MAP,
        do_360_scan: bool = True,
        reset_scene: bool = True,
        waypoints_subset: str = "all",
        grid_spacing: float = 1.5,
    ):
        self._cancelled = False

        # Verify Thor is reachable
        try:
            resp = requests.get(f"{THOR_URL}/mars/health", timeout=5)
            if resp.status_code != 200:
                return "Thor perception server not reachable", PrimitiveResult.FAILURE
        except requests.ConnectionError:
            return f"Cannot connect to Thor at {THOR_URL}", PrimitiveResult.FAILURE

        # Optionally reset Thor's scene graph
        if reset_scene:
            requests.post(f"{THOR_URL}/mars/reset", timeout=5)
            print("[explore_room] Scene graph reset")
        else:
            print("[explore_room] Keeping existing scene graph (incremental mode)")

        # Generate waypoints from map
        try:
            wps = _generate_waypoints(
                map_name=map_name, grid_spacing=grid_spacing
            )
            print(
                f"[explore_room] Generated {len(wps)} waypoints from "
                f"'{map_name}' map (grid={grid_spacing}m)"
            )
        except Exception as e:
            print(f"[explore_room] Map waypoint generation failed: {e}")
            traceback.print_exc()
            print("[explore_room] Using fallback waypoints")
            wps = list(_FALLBACK_WAYPOINTS)

        # Filter waypoints subset if requested
        if waypoints_subset != "all":
            try:
                indices = [int(i) for i in waypoints_subset.split(",")]
                wps = [wps[i] for i in indices if i < len(wps)]
            except ValueError:
                pass

        if not wps:
            return "No valid waypoints generated", PrimitiveResult.FAILURE

        # Order waypoints by nearest-neighbor from robot's current position
        robot_pos = _get_robot_position()
        wps = _nearest_neighbor_order(wps, start_xy=robot_pos)
        print(
            f"[explore_room] Visiting {len(wps)} waypoints "
            f"(ordered from {'robot pos' if robot_pos else 'first waypoint'})"
        )

        total_captures = 0
        nav_errors = []
        nav_ok = 0

        # Initial 360° scan at current position (before any navigation)
        if do_360_scan:
            print("[explore_room] Initial 360° scan")
            total_captures += self._scan_360()

        # Visit each waypoint
        for i, (wx, wy) in enumerate(wps):
            if self._cancelled:
                return "Mapping cancelled", PrimitiveResult.CANCELLED

            print(f"[explore_room] Waypoint {i+1}/{len(wps)}: ({wx:.2f}, {wy:.2f})")

            # Start background capture thread during transit
            transit_stop = threading.Event()
            transit_captures = [0]

            def _transit_capture_loop():
                while not transit_stop.is_set():
                    transit_stop.wait(CAPTURE_INTERVAL)
                    if not transit_stop.is_set():
                        transit_captures[0] += self._capture_and_send()

            capture_thread = threading.Thread(
                target=_transit_capture_loop, daemon=True
            )
            capture_thread.start()

            # Navigate to waypoint (Nav2 handles path planning + obstacles)
            # On failure, try nearby offsets before giving up
            nav_succeeded = False
            try:
                self._navigate_to(wx, wy)
                nav_succeeded = True
            except Exception as e:
                print(f"[explore_room] Nav to ({wx:.1f},{wy:.1f}) failed: {e}, trying offsets")
                for dx, dy in [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]:
                    try:
                        self._navigate_to(wx + dx, wy + dy)
                        print(f"[explore_room] Reached offset ({wx+dx:.1f},{wy+dy:.1f})")
                        nav_succeeded = True
                        break
                    except Exception:
                        continue
                if not nav_succeeded:
                    err = f"wp{i}({wx:.1f},{wy:.1f}): unreachable"
                    print(f"[explore_room] {err}")
                    nav_errors.append(err)

            if nav_succeeded:
                nav_ok += 1
            else:
                transit_stop.set()
                capture_thread.join(timeout=2)
                total_captures += transit_captures[0]
                continue

            # Stop transit captures
            transit_stop.set()
            capture_thread.join(timeout=2)
            total_captures += transit_captures[0]
            print(f"[explore_room] Transit captured {transit_captures[0]} frames")

            # Capture at this position (stationary = better quality)
            total_captures += self._capture_and_send(force=True)

            # 360° scan every 3rd waypoint
            if do_360_scan and i % 3 == 0:
                total_captures += self._scan_360()

        # Final summary — report what objects were detected
        try:
            resp = requests.get(f"{THOR_URL}/mars/scene", timeout=10)
            scene = resp.json()
            objects = scene.get("objects", [])
            n_objects = len(objects)
            object_names = [o.get("label", "?") for o in objects]
        except Exception:
            n_objects = "unknown"
            object_names = []

        msg = (
            f"Mapping complete. {total_captures} frames captured, "
            f"{n_objects} objects detected. "
            f"Nav: {nav_ok}/{len(wps)} ok."
        )
        if object_names:
            msg += f" Objects found: {', '.join(object_names[:15])}"
        if nav_errors:
            msg += f" Errors: {'; '.join(nav_errors[:3])}"

        return msg, PrimitiveResult.SUCCESS

    def cancel(self):
        self._cancelled = True
        return "Mapping cancelled"

    # ─── Internal helpers ────────────────────────────────────────

    _cancelled = False
    _nav2 = None
    _last_capture_pose = None  # (x, y, yaw) of last successful capture

    def _navigate_to(self, x: float, y: float):
        """Use Nav2 to drive to a map-frame waypoint."""
        if self._nav2 is None:
            import sys

            skills_dir = os.path.dirname(os.path.abspath(__file__))
            if skills_dir not in sys.path:
                sys.path.insert(0, skills_dir)
            from navigate_to_position import Nav2Controller

            self._nav2 = Nav2Controller(self.logger, self)
        from nav2_simple_commander.robot_navigator import TaskResult

        result = self._nav2.go_to_position(x, y, 0.0, local_frame=False)
        if result != TaskResult.SUCCEEDED:
            raise RuntimeError(f"Navigation failed: {result}")

    def _scan_360(self) -> int:
        """Rotate in place, capturing at each angle."""
        captures = 0
        angle_step = 2 * math.pi / SCAN_ANGLES
        for _ in range(SCAN_ANGLES):
            self.mobility.rotate(angle_step)
            self.mobility.send_cmd_vel(
                linear_x=0.0, angular_z=0.0, duration=0.1
            )  # explicit stop
            time.sleep(1.0)  # wait for RGB + depth to settle
            self._capture_and_send(force=True)
            captures += 1
        return captures

    def _capture_and_send(self, force: bool = False) -> int:
        """Grab RGB from skill framework + depth/pose from cache, POST to Thor.

        Args:
            force: If True, skip the movement-delta check (used for stationary
                   captures and 360° scans where we always want a frame).
        """
        try:
            rgb_b64 = self.image
            if not rgb_b64:
                return 0

            # Get point cloud + pose from local cache
            cache_resp = requests.get(
                f"{DEPTH_CACHE_URL}/latest",
                timeout=2,
            )
            cache = cache_resp.json()
            if not cache.get("ready"):
                print(f"[explore_room] cache not ready: {cache.get('reason')}")
                return 0

            # Skip redundant transit frames if robot hasn't moved enough
            if not force and self._last_capture_pose is not None:
                pose = cache.get("pose", {})
                if pose.get("valid"):
                    cx = pose["translation"][0]
                    cy = pose["translation"][1]
                    cyaw = math.atan2(
                        2.0 * (pose["rotation"][3] * pose["rotation"][2]),
                        1.0 - 2.0 * (pose["rotation"][2] ** 2),
                    )
                    lx, ly, lyaw = self._last_capture_pose
                    dist = math.sqrt((cx - lx) ** 2 + (cy - ly) ** 2)
                    angle = abs(cyaw - lyaw)
                    if angle > math.pi:
                        angle = 2 * math.pi - angle
                    if dist < MIN_CAPTURE_DISTANCE and angle < MIN_CAPTURE_ANGLE:
                        return 0  # skip — haven't moved enough

            # POST bundle to Thor (RGB + organized point cloud + pose)
            bundle = {
                "rgb_b64": rgb_b64,
                "pointcloud_b64": cache["pointcloud_b64"],
                "pc_width": cache["pc_width"],
                "pc_height": cache["pc_height"],
                "pose": cache["pose"],
            }
            resp = requests.post(
                f"{THOR_URL}/mars/ingest_frame",
                json=bundle,
                timeout=10,
            )
            if resp.status_code == 200:
                # Update last capture pose
                pose = cache.get("pose", {})
                if pose.get("valid"):
                    self._last_capture_pose = (
                        pose["translation"][0],
                        pose["translation"][1],
                        math.atan2(
                            2.0 * (pose["rotation"][3] * pose["rotation"][2]),
                            1.0 - 2.0 * (pose["rotation"][2] ** 2),
                        ),
                    )
                result = resp.json()
                n = len(result.get("detections", []))
                if n > 0:
                    print(f"[explore_room] {n} objects detected")
                return 1
            else:
                print(f"[explore_room] Thor returned {resp.status_code}")
                return 0

        except Exception as e:
            print(f"[explore_room] capture error: {e}")
            return 0
