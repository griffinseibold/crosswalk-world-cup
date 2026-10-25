"""Constant-velocity Kalman filter for one bounding box.

State ``x = [cx, cy, w, h, vx, vy, vw, vh]``: box centre, size and their rates
of change in pixels per second. Measurements are ``[cx, cy, w, h]``. This is
the SORT/ByteTrack formulation with two deliberate choices:

- **Time steps come from timestamps**, not frame counts. The sampler keeps
  frames at a target rate, but drops (overflow policy) and variable-rate clips
  mean consecutive frames are not equally spaced, and velocities in pixels per
  second are what the conflict indicators need.
- **Noise scales with box height**, as in ByteTrack/DeepSORT: a pedestrian
  20 px tall far from the camera and one 200 px tall close to it should not
  share absolute pixel noise.

Plain NumPy, 8x8 matrices: a few microseconds per update, so no batching or
GPU is needed for tens of tracks per frame.
"""

from __future__ import annotations

import numpy as np

# Process noise per unit height: position (std of unmodelled acceleration
# integrated over 1 s) and velocity. From ByteTrack's defaults (1/20, 1/160
# per frame at 30 fps), converted to per-second units.
_STD_POS = 1.0 / 20.0
_STD_VEL = 1.0 / 160.0 * 30.0
_STD_MEAS = 1.0 / 20.0

_H = np.hstack([np.eye(4), np.zeros((4, 4))])


def box_to_z(box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float)


def z_to_box(z: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, w, h = (float(v) for v in z[:4])
    w, h = max(w, 0.0), max(h, 0.0)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class KalmanBox:
    def __init__(self, box: tuple[float, float, float, float]):
        z = box_to_z(box)
        self.x = np.concatenate([z, np.zeros(4)])
        h = max(z[3], 1.0)
        std = np.array(
            [2 * _STD_POS * h, 2 * _STD_POS * h, 2 * _STD_POS * h, 2 * _STD_POS * h,
             10 * _STD_VEL * h, 10 * _STD_VEL * h, 10 * _STD_VEL * h, 10 * _STD_VEL * h]
        )  # fmt: skip
        self.P = np.diag(std**2)

    def predict(self, dt: float) -> None:
        dt = max(float(dt), 0.0)
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        h = max(self.x[3], 1.0)
        # Noise grows with the time step, so a long gap widens the gate
        # instead of trusting a stale velocity.
        q_pos = (_STD_POS * h) ** 2 * dt
        q_vel = (_STD_VEL * h) ** 2 * dt
        Q = np.diag([q_pos] * 4 + [q_vel] * 4)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # A box cannot shrink below zero size.
        self.x[2] = max(self.x[2], 1e-3)
        self.x[3] = max(self.x[3], 1e-3)

    def update(self, box: tuple[float, float, float, float]) -> None:
        z = box_to_z(box)
        h = max(self.x[3], 1.0)
        R = np.diag([(_STD_MEAS * h) ** 2] * 4)
        y = z - _H @ self.x
        S = _H @ self.P @ _H.T + R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ _H) @ self.P

    @property
    def box(self) -> tuple[float, float, float, float]:
        return z_to_box(self.x)

    @property
    def velocity(self) -> tuple[float, float]:
        """Velocity of the box centre, pixels per second."""
        return float(self.x[4]), float(self.x[5])

    @property
    def foot_velocity(self) -> tuple[float, float]:
        """Velocity of the bottom centre (where an object meets the ground):
        centre velocity plus half the height's rate of change."""
        return float(self.x[4]), float(self.x[5] + self.x[7] / 2)
