"""
Full agent–environment–sensor loop.

Runs both SensorPIDController and EKFPIDController in parallel against the
same AircraftEnvironment + LiDARSensor rollout and prints a side-by-side
comparison of altitude tracking and elevator commands.
"""

import numpy as np

from environment import AircraftEnvironment
from sensor import LiDARSensor
from agent import SensorPIDController, EKFPIDController

# ── Parameters ────────────────────────────────────────────────────────────────
Z0       = 0.5   # initial altitude (m)
Z_TARGET = 0.5      # target altitude (m)
N_STEPS  = 2000      # timesteps
SEED     = 42
ENABLE_LATERAL_DAMPER = True
USE_EKF_PID_CONTROLLER = False
sensor_args = dict(
    mu_A=0.05,
    sigma_A=0.02,
    mu_k=2 * np.pi,       # ≈ 1 m wavelength
    sigma_k=0.5,
    sigma_eps=0.01,
    p_penetration=0.05,
    alpha_min=0.5,
    alpha_max=10,
    perfect_sensing=True,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
env    = AircraftEnvironment()
sensor = LiDARSensor(**sensor_args)
env.seed(SEED)
sensor.seed(SEED)

state = env.reset(z0=Z0)
sensor.reset()

raw_pid = SensorPIDController()

ekf_pid = EKFPIDController(
    lateral_damping_enabled=ENABLE_LATERAL_DAMPER,
)
ekf_pid.reset(x0=(Z0, 0.0))

# ── Run loop ──────────────────────────────────────────────────────────────────
print(
    f"{'step':>5}  {'z_true':>8}  {'z_hat':>8}  "
    f"{'de_pid':>8}  {'de_ekf':>8}  {'da_ekf':>8}  {'z_est':>8}  {'theta_est':>10}"
)
print("-" * 72)
print(f"Controller driving env.step: {'EKF+PID' if USE_EKF_PID_CONTROLLER else 'Pure PID'}")

for step in range(N_STEPS):
    z_true, _dz, x_true, dx_true, theta_true, _dth = state
    roll_true, omega_p_true, omega_r_true = env.get_lateral_state()
    z_hat, theta_hat = sensor.measure(z_true, x_true, theta_true)

    delta_e_raw = raw_pid.update(z_hat, theta_hat, Z_TARGET, dt=0.01)
    delta_e_ekf, x_est = ekf_pid.step(
        z_hat,
        theta_hat,
        Z_TARGET,
        forward_speed=dx_true,
        roll_hat=roll_true,
        omega_p=omega_p_true,
        omega_r=omega_r_true,
    )

    if USE_EKF_PID_CONTROLLER:
        delta_e_cmd = delta_e_ekf
        delta_a_cmd = ekf_pid.aileron_command
    else:
        delta_e_cmd = delta_e_raw
        delta_a_cmd = 0.0

    # Advance environment with selected controller command
    success, state = env.step(delta_e_cmd, delta_a=delta_a_cmd)

    if step % 20 == 0 or not success:
        print(
            f"{step:5d}  "
            f"{z_true:8.4f}  "
            f"{z_hat:8.4f}  "
            f"{delta_e_raw:8.4f}  "
            f"{delta_e_ekf:8.4f}  "
            f"{ekf_pid.aileron_command:8.4f}  "
            f"{x_est[0]:8.4f}  "
            f"{x_est[1]:10.5f}"
        )

    if not success:
        print("Simulation terminated early (constraint violation).")
        break
else:
    print(f"\nCompleted {N_STEPS} steps without constraint violation.")
