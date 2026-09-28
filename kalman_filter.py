"""Small dependency-free 1D kinematic filter for live rail telemetry."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KinematicEstimate:
    position_km: float
    velocity_kmph: float
    acceleration_kmph_per_s: float
    gps_available: bool


class KinematicDeadReckoner:
    """Constant-acceleration Kalman-like smoother with explicit GPS-loss mode."""

    def __init__(self) -> None:
        self.position_km = 0.0
        self.velocity_kmph = 0.0
        self.acceleration_kmph_per_s = 0.0
        self.position_variance = 0.1
        self.velocity_variance = 4.0

    def update(self, measured_speed_kmph: float, dt_seconds: float, gps_available: bool = True) -> KinematicEstimate:
        dt = max(0.05, min(dt_seconds, 30.0))
        predicted_position = self.position_km + self.velocity_kmph * dt / 3600 + 0.5 * self.acceleration_kmph_per_s * dt * dt / 3600
        predicted_velocity = max(0.0, self.velocity_kmph + self.acceleration_kmph_per_s * dt)
        if gps_available:
            gain = self.velocity_variance / (self.velocity_variance + 2.5)
            correction = measured_speed_kmph - predicted_velocity
            self.velocity_kmph = predicted_velocity + gain * correction
            self.acceleration_kmph_per_s = max(-0.08, min(0.08, correction / dt))
            self.velocity_variance = max(0.25, (1 - gain) * self.velocity_variance)
            self.position_variance = min(4.0, self.position_variance + 0.01)
        else:
            self.velocity_kmph = predicted_velocity
            self.position_variance = min(25.0, self.position_variance + 0.25)
            self.velocity_variance = min(100.0, self.velocity_variance + 0.8)
        self.position_km = predicted_position
        return KinematicEstimate(self.position_km, self.velocity_kmph, self.acceleration_kmph_per_s, gps_available)
