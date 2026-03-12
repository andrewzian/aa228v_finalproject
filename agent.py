from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter


def _clamp(value: float, limit: float | None) -> float:
    if limit is None:
        return value
    return float(np.clip(value, -limit, limit))


@dataclass
class PIDAxis:
    kp: float
    ki: float
    kd: float
    integrator_limit: float | None = None
    output_limit: float | None = None

    def __post_init__(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be positive")

        if self.initialized:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
            self.initialized = True

        proportional_term = self.kp * error
        derivative_term = self.kd * derivative

        previous_integral = self.integral
        candidate_integral = previous_integral + error * dt
        candidate_integral = _clamp(candidate_integral, self.integrator_limit)
        integral_term = self.ki * candidate_integral

        control_unsaturated = proportional_term + integral_term + derivative_term
        control = _clamp(control_unsaturated, self.output_limit)

        if self.output_limit is not None and self.ki != 0.0 and control != control_unsaturated:
            saturating_high = control_unsaturated > control and error > 0.0
            saturating_low = control_unsaturated < control and error < 0.0

            if saturating_high or saturating_low:
                integral_term = self.ki * previous_integral
                control_unsaturated = proportional_term + integral_term + derivative_term
                control = _clamp(control_unsaturated, self.output_limit)
            else:
                self.integral = candidate_integral
        else:
            self.integral = candidate_integral

        self.prev_error = error
        return control


class SensorPIDController:
    """Simple PID controller using raw sensor readings (z_hat, theta_hat)."""

    def __init__(
        self,
        altitude_gains: Tuple[float, float, float] = (0.2, 0.01, 0.0),
        pitch_gains: Tuple[float, float, float] = (2.0, 0.1, 0.0),
        integral_limit: float = 0.5,
        pitch_command_limit: float = np.deg2rad(20.0),
        elevator_limit: float = np.deg2rad(25.0),
    ) -> None:
        self.altitude_pid = PIDAxis(
            *altitude_gains,
            integrator_limit=integral_limit,
            output_limit=pitch_command_limit,
        )
        self.pitch_pid = PIDAxis(
            *pitch_gains,
            integrator_limit=integral_limit,
            output_limit=elevator_limit,
        )
        self.elevator_limit = elevator_limit

    def reset(self) -> None:
        self.altitude_pid.reset()
        self.pitch_pid.reset()

    def update(
        self,
        z_hat: float,
        theta_hat: float,
        z_target: float,
        dt: float,
    ) -> float:
        altitude_error = z_target - z_hat
        theta_target = self.altitude_pid.update(altitude_error, dt)

        pitch_error = theta_target - theta_hat
        delta_e = self.pitch_pid.update(pitch_error, dt)

        return _clamp(delta_e, self.elevator_limit)


class AltitudePitchEKF:
    """2-state EKF for [z, theta] with nonlinear altitude update.

    Noise parameters are grounded in sensor.py:
        R[0,0]: z noise -- wave amplitude A~N(0.05,0.02) gives variance ~A²/2 ≈ 0.00125;
                vibration sigma_eps=0.01 adds 0.0001; penetration events (~5%, 0.1-0.5m)
                add ~0.005. Total ≈ 0.01 (std ~0.1 m).
        R[1,1]: pitch is a perfect measurement (theta_hat = theta in LiDARSensor),
                so set near zero.
    """

    def __init__(
        self,
        dt: float = 0.01,            # PyFly runs at 100 Hz
        forward_speed: float = 17.0, # matches AircraftEnvironment default Va0
        theta_damping: float = 1.5,
        elevator_effect: float = 3.0,
        q_diag: Tuple[float, float] = (0.1, 0.01),
        r_diag: Tuple[float, float] = (0.01, 1e-6), # z: wave+vib noise; theta: perfect
        p_diag: Tuple[float, float] = (1.0, 0.1),
        x0: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.dt = dt
        self.forward_speed = forward_speed
        self.theta_damping = theta_damping
        self.elevator_effect = elevator_effect

        self.ekf = ExtendedKalmanFilter(dim_x=2, dim_z=2)
        self.ekf.x = np.array(x0, dtype=float)
        # Initial state uncertainty
        self.ekf.P = np.diag(np.array(p_diag, dtype=float))
        # Process noise covariance (accounts for unmodeled dz, dx, dtheta dynamics)
        self.ekf.Q = np.diag(np.array(q_diag, dtype=float))
        # Measurement noise covariance (z: LiDAR wave+vib; theta: perfect)
        self.ekf.R = np.diag(np.array(r_diag, dtype=float))

    def set_dt(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = dt

    def _f(
        self,
        x: np.ndarray,
        delta_e: float,
        forward_speed: float | None = None,
    ) -> np.ndarray:
        z, theta = x
        speed = self.forward_speed if forward_speed is None else forward_speed
        z_next = z + self.dt * speed * np.sin(theta)
        theta_next = theta + self.dt * (
            -self.theta_damping * theta + self.elevator_effect * delta_e
        )
        return np.array([z_next, theta_next], dtype=float)

    def _F_jacobian(
        self,
        x: np.ndarray,
        _delta_e: float,
        forward_speed: float | None = None,
    ) -> np.ndarray:
        _, theta = x
        speed = self.forward_speed if forward_speed is None else forward_speed
        return np.array(
            [
                [1.0, self.dt * speed * np.cos(theta)],
                [0.0, 1.0 - self.dt * self.theta_damping],
            ],
            dtype=float,
        )

    @staticmethod
    def _h(x: np.ndarray) -> np.ndarray:
        return np.array([x[0], x[1]], dtype=float)

    @staticmethod
    def _H_jacobian(_x: np.ndarray) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    def predict(self, delta_e: float, forward_speed: float | None = None) -> None:
        x_prior = self.ekf.x.copy()
        F = self._F_jacobian(x_prior, delta_e, forward_speed=forward_speed)
        self.ekf.x = self._f(x_prior, delta_e, forward_speed=forward_speed)
        self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q

    def update(self, z_hat: float, theta_hat: float) -> None:
        measurement = np.array([z_hat, theta_hat], dtype=float)
        self.ekf.update(measurement, HJacobian=self._H_jacobian, Hx=self._h)

    @property
    def state(self) -> np.ndarray:
        return self.ekf.x.copy()


class EKFPIDController:
    """Estimate state with EKF first, then apply PID on estimated state."""

    def __init__(
        self,
        dt: float = 0.01,  # PyFly runs at 100 Hz
        altitude_gains: Tuple[float, float, float] = (0.2, 0.01, 0.0),
        pitch_gains: Tuple[float, float, float] = (2.0, 0.1, 0.0),
        lateral_damping_enabled: bool = False,
        lateral_gains: Tuple[float, float, float] = (0.5, 0.2, 0.0),
        aileron_limit: float = np.deg2rad(20.0),
        integral_limit: float = 0.5,
        pitch_command_limit: float = np.deg2rad(20.0),
        elevator_limit: float = np.deg2rad(25.0),
    ) -> None:
        self.dt = dt
        self.estimator = AltitudePitchEKF(dt=dt)
        self.pid = SensorPIDController(
            altitude_gains=altitude_gains,
            pitch_gains=pitch_gains,
            integral_limit=integral_limit,
            pitch_command_limit=pitch_command_limit,
            elevator_limit=elevator_limit,
        )
        self.lateral_damping_enabled = lateral_damping_enabled
        self.k_phi, self.k_p, self.k_r = lateral_gains
        self.aileron_limit = aileron_limit
        self.last_delta_e = 0.0
        self.aileron_command = 0.0

    def reset(self, x0: Tuple[float, float] = (0.5, 0.0)) -> None:
        self.estimator = AltitudePitchEKF(dt=self.dt, x0=x0)
        self.pid.reset()
        self.last_delta_e = 0.0
        self.aileron_command = 0.0

    def step(
        self,
        z_hat: float,
        theta_hat: float,
        z_target: float,
        forward_speed: float | None = None,
        roll_hat: float | None = None,
        omega_p: float | None = None,
        omega_r: float | None = None,
        dt: float | None = None,
    ) -> Tuple[float, np.ndarray]:
        if dt is not None:
            if dt <= 0:
                raise ValueError("dt must be positive")
            self.dt = dt
            self.estimator.set_dt(dt)

        self.estimator.predict(self.last_delta_e, forward_speed=forward_speed)
        self.estimator.update(z_hat, theta_hat)

        z_est, theta_est = self.estimator.state
        delta_e = self.pid.update(z_est, theta_est, z_target, self.dt)

        if (
            self.lateral_damping_enabled
            and roll_hat is not None
            and omega_p is not None
            and omega_r is not None
        ):
            delta_a = -(
                self.k_phi * roll_hat
                + self.k_p * omega_p
                + self.k_r * omega_r
            )
            self.aileron_command = _clamp(delta_a, self.aileron_limit)
        else:
            self.aileron_command = 0.0

        self.last_delta_e = delta_e

        return delta_e, self.estimator.state


if __name__ == "__main__":
    pid = SensorPIDController()
    ekf = EKFPIDController()
    ekf.reset(x0=(0.2122, 0.0))

    z_hat, theta_hat, z_target = 0.22, 0.0, 0.5
    de_pid = pid.update(z_hat, theta_hat, z_target, dt=0.01)
    de_ekf, x_est = ekf.step(z_hat, theta_hat, z_target)

    print(f"SensorPID  delta_e = {de_pid:.4f} rad")
    print(f"EKF+PID    delta_e = {de_ekf:.4f} rad,  z_est = {x_est[0]:.4f} m,  theta_est = {x_est[1]:.5f} rad")