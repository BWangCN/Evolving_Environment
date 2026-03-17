"""Franka Panda robot configuration and action format specification."""

import numpy as np

# ============================================================
# Franka Panda Specifications
# ============================================================

# Home pose: end-effector position and orientation (quaternion xyzw)
# Corresponds to joint angles [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]
# EE approximately 40cm above table center, gripper pointing down
HOME_POSITION = np.array([0.3, 0.0, 0.45])
HOME_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz, gripper pointing -z

# Gripper
GRIPPER_MAX_OPENING = 0.08      # 8cm max opening width (meters)
GRIPPER_FINGER_LENGTH = 0.04    # Approximate finger length (meters)
GRIPPER_OPEN = 1.0              # Normalized open state
GRIPPER_CLOSE = 0.0             # Normalized close state

# Workspace limits (meters, in robot base frame)
WORKSPACE_X_RANGE = (-0.2, 0.6)   # forward/backward
WORKSPACE_Y_RANGE = (-0.4, 0.4)   # left/right
WORKSPACE_Z_RANGE = (0.01, 0.60)  # table surface to max reach

# ============================================================
# Wrist Camera Configuration
# ============================================================

# Camera-to-EE fixed transform
# Camera mounted on wrist, looking forward along gripper approach direction
CAMERA_TO_EE_POSITION = np.array([0.0, 0.0, -0.05])  # 5cm behind EE tip
CAMERA_TO_EE_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz, aligned with EE

# Camera intrinsics (640x480, approximate for typical wrist cam)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FX = 450.0
CAMERA_FY = 450.0
CAMERA_CX = 320.0
CAMERA_CY = 240.0

# ============================================================
# Trajectory Parameters
# ============================================================

# Phase heights and offsets (meters)
PRE_GRASP_HEIGHT = 0.10        # Height above grasp pose for pre-grasp
LIFT_HEIGHT = 0.12             # Height to lift after grasping
TRANSIT_HEIGHT = 0.30          # Height for transit between pick and place
PRE_PLACE_HEIGHT = 0.08        # Height above place target for pre-place
RETREAT_HEIGHT = 0.10          # Height to retreat after placing

# Interpolation density (waypoints per segment)
WAYPOINTS_PER_SEGMENT = 10

# ============================================================
# Collision Detection
# ============================================================

# Safety margins (meters) for inflated collision volumes
GRIPPER_SAFETY_MARGIN = 0.03           # Margin around gripper during approach/transit
GRASPED_OBJECT_SAFETY_MARGIN = 0.05    # Margin around grasped object during transport
PLACE_APPROACH_MARGIN = 0.01           # Reduced margin when approaching place target

# ============================================================
# π0.5 Action Format
# ============================================================

# π0.5 outputs 7-DoF EE delta actions:
# [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_state]
ACTION_DIM = 7

# Action normalization range for π0.5
# Position deltas: typically within ±2cm per step
ACTION_POS_SCALE = 0.02   # meters per unit
# Orientation deltas: typically within ±0.1 rad per step
ACTION_ROT_SCALE = 0.1    # radians per unit
