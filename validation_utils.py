"""
Validation utilities: rollout primitive, robustness metric, and sensor density helpers.
"""

import numpy as np
from scipy.stats import norm, uniform

from environment import AircraftEnvironment
from sensor import LiDARSensor
from agent import EKFPIDController

DEFAULT_SPEC = {
    "z_min": 0.1,
    "z_max": 10.0,
    "pitch_min": -np.pi / 6,
    "pitch_max": np.pi / 6,
}


def compute_robustness(z_traj, theta_traj, specification):
    """
    Compute the robustness metric ρ for a trajectory.

    ρ = min(
        min(z_t - z_min),          # altitude floor margin
        min(z_max - z_t),          # altitude ceiling margin
        min_t( min(theta_t - pitch_min, pitch_max - theta_t) )  # pitch margin
    )

    Positive = safe; negative = failure.
    """
    z_arr = np.array(z_traj, dtype=float)
    theta_arr = np.array(theta_traj, dtype=float)

    z_min = specification["z_min"]
    z_max = specification["z_max"]
    pitch_min = specification["pitch_min"]
    pitch_max = specification["pitch_max"]

    floor_margins = z_arr - z_min
    ceiling_margins = z_max - z_arr
    pitch_margins = np.minimum(theta_arr - pitch_min, pitch_max - theta_arr)

    rho = min(
        float(np.min(floor_margins)),
        float(np.min(ceiling_margins)),
        float(np.min(pitch_margins)),
    )
    return rho


def run_rollout(
    sensor_params,
    z0,
    z_target,
    n_steps,
    seed,
    sensor_args,
    specification,
    ground_effect_enabled=True,
    enable_lateral_damper=True,
):
    """
    Run a single rollout and return trajectory + robustness.

    :param sensor_params: dict of optional overrides for sensor attrs: 'A', 'k', 'alpha'.
                          Pass {} to use the sensor's sampled values.
    :param z0: initial altitude (m)
    :param z_target: target altitude (m)
    :param n_steps: number of simulation steps
    :param seed: RNG seed (controls per-step ε_t and B_t via sensor.seed())
    :param sensor_args: base LiDARSensor __init__ kwargs
    :param specification: dict with z_min, z_max, pitch_min, pitch_max
    :param ground_effect_enabled: enable aerodynamic ground effect correction
    :param enable_lateral_damper: enable lateral stabilization damper
    :return: dict with rho, is_failure, z_traj, theta_traj,
             sensor_A, sensor_k, sensor_alpha, z0_actual
    """
    env = AircraftEnvironment(ground_effect_enabled=ground_effect_enabled)
    sensor = LiDARSensor(**sensor_args)
    env.seed(seed)
    sensor.seed(seed)

    state = env.reset(z0=z0)
    sensor.reset()

    # Override sensor per-rollout parameters after reset()
    if "A" in sensor_params:
        sensor.A = float(sensor_params["A"])
    if "k" in sensor_params:
        sensor.k = float(sensor_params["k"])
    if "alpha" in sensor_params:
        sensor.alpha = float(sensor_params["alpha"])

    ekf_pid = EKFPIDController(lateral_damping_enabled=enable_lateral_damper)
    ekf_pid.reset(x0=(z0, 0.0))

    z_traj = [float(state[0])]
    theta_traj = [float(state[4])]

    for _step in range(n_steps):
        z_true, _dz, x_true, dx_true, theta_true, _dth = state

        # Guard against NaN states
        if np.isnan(z_true) or np.isnan(theta_true):
            break

        roll_true, omega_p_true, omega_r_true = env.get_lateral_state()
        z_hat, theta_hat = sensor.measure(z_true, x_true, theta_true)

        delta_e_ekf, _x_est = ekf_pid.step(
            z_hat,
            theta_hat,
            z_target,
            forward_speed=dx_true,
            roll_hat=roll_true,
            omega_p=omega_p_true,
            omega_r=omega_r_true,
        )

        success, state = env.step(delta_e_ekf, delta_a=ekf_pid.aileron_command)

        z_next, _dz_next, _x_next, _dx_next, theta_next, _dth_next = state

        if np.isnan(z_next) or np.isnan(theta_next):
            break

        z_traj.append(float(z_next))
        theta_traj.append(float(theta_next))

        if not success:
            break

    rho = compute_robustness(z_traj, theta_traj, specification)

    return {
        "rho": rho,
        "is_failure": rho < 0.0,
        "z_traj": z_traj,
        "theta_traj": theta_traj,
        "sensor_A": sensor.A,
        "sensor_k": sensor.k,
        "sensor_alpha": sensor.alpha,
        "z0_actual": float(z0),
    }


class NominalSensorDistribution:
    """
    Computes log p_nominal(A, k, alpha) under the nominal sensor model.

    A     ~ N(mu_A, sigma_A)
    k     ~ N(mu_k, sigma_k)
    alpha ~ Uniform(alpha_min, alpha_max)
    """

    def __init__(self, sensor_args):
        self.mu_A = sensor_args["mu_A"]
        self.sigma_A = sensor_args["sigma_A"]
        self.mu_k = sensor_args["mu_k"]
        self.sigma_k = sensor_args["sigma_k"]
        self.alpha_min = sensor_args["alpha_min"]
        self.alpha_max = sensor_args["alpha_max"]

    def log_prob(self, A, k, alpha):
        """Return log p_nominal(A, k, alpha)."""
        log_p_A = norm.logpdf(A, self.mu_A, self.sigma_A)
        log_p_k = norm.logpdf(k, self.mu_k, self.sigma_k)
        log_p_alpha = uniform.logpdf(
            alpha, self.alpha_min, self.alpha_max - self.alpha_min
        )
        return float(log_p_A + log_p_k + log_p_alpha)
