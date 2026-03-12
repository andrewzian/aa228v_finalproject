"""
Full agent–environment–sensor loop.

Runs both SensorPIDController and EKFPIDController in parallel against the
same AircraftEnvironment + LiDARSensor rollout and prints a side-by-side
comparison of altitude tracking and elevator commands.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import AircraftEnvironment
from sensor import LiDARSensor
from agent import SensorPIDController, EKFPIDController


def run_agent_env_sensor_loop(
    z0,
    z_target,
    n_steps,
    seed,
    enable_lateral_damper,
    use_ekf_pid_controller,
    sensor_args,
    specification,
    save_trajectory_plot=False,
    trajectory_plot_path="plots/plane_trajectory.png",
    ground_effect_enabled=False,
):
    """Run the full agent-environment-sensor rollout.

    This function is intended to be imported and called externally.
    
    :param z0: Initial altitude (m)
    :param z_target: Target altitude (m)
    :param n_steps: Number of simulation steps
    :param seed: Random seed
    :param enable_lateral_damper: Enable lateral stabilization damper
    :param use_ekf_pid_controller: Use EKF+PID vs pure PID
    :param sensor_args: Dict of sensor configuration parameters
    :param specification: Dict with z_min, z_max, pitch_min, pitch_max bounds
    :param save_trajectory_plot: Whether to save trajectory visualization
    :param trajectory_plot_path: Path to save trajectory plot
    :param ground_effect_enabled: Enable aerodynamic ground effect correction
    """
    if specification is not None:
        required_keys = ("z_min", "z_max", "pitch_min", "pitch_max")
        missing_keys = [key for key in required_keys if key not in specification]
        if missing_keys:
            raise ValueError(
                f"specification missing required keys: {missing_keys}. "
                "Expected z_min, z_max, pitch_min, pitch_max."
            )

    env = AircraftEnvironment(ground_effect_enabled=ground_effect_enabled)
    sensor = LiDARSensor(**sensor_args)
    env.seed(seed)
    sensor.seed(seed)

    state = env.reset(z0=z0)
    sensor.reset()
    x_traj = [float(state[2])]
    z_traj = [float(state[0])]
    delta_e_traj = [0.0]  # track elevator deflection commands

    raw_pid = SensorPIDController()
    ekf_pid = EKFPIDController(lateral_damping_enabled=enable_lateral_damper)
    ekf_pid.reset(x0=(z0, 0.0))

    print(
        f"{'step':>5}  {'z_true':>8}  {'z_hat':>8}  "
        f"{'de_pid':>8}  {'de_ekf':>8}  {'da_ekf':>8}  {'z_est':>8}  {'theta_est':>10}"
    )
    print("-" * 72)
    print(
        f"Controller driving env.step: {'EKF+PID' if use_ekf_pid_controller else 'Pure PID'}"
    )

    terminated_early = False
    termination_step = None
    termination_type = None
    termination_message = None
    violated_spec = None

    def check_spec_violations(z_value, pitch_value):
        if specification is None:
            return []

        violations = []
        if z_value < specification["z_min"]:
            violations.append(
                f"z={z_value:.4f} < z_min={specification['z_min']:.4f}"
            )
        if z_value > specification["z_max"]:
            violations.append(
                f"z={z_value:.4f} > z_max={specification['z_max']:.4f}"
            )
        if pitch_value < specification["pitch_min"]:
            violations.append(
                f"pitch={pitch_value:.5f} < pitch_min={specification['pitch_min']:.5f}"
            )
        if pitch_value > specification["pitch_max"]:
            violations.append(
                f"pitch={pitch_value:.5f} > pitch_max={specification['pitch_max']:.5f}"
            )

        return violations

    for step in range(n_steps):
        z_true, _dz, x_true, dx_true, theta_true, _dth = state
        pre_step_violations = check_spec_violations(z_true, theta_true)
        if pre_step_violations:
            terminated_early = True
            termination_step = step
            termination_type = "spec_violation"
            violated_spec = {
                "z": z_true,
                "pitch": theta_true,
                "violations": pre_step_violations,
            }
            termination_message = (
                f"Specification violated at step {step}: {', '.join(pre_step_violations)}"
            )
            print(termination_message)
            break

        roll_true, omega_p_true, omega_r_true = env.get_lateral_state()
        z_hat, theta_hat = sensor.measure(z_true, x_true, theta_true)

        delta_e_raw = raw_pid.update(z_hat, theta_hat, z_target, dt=0.01)
        delta_e_ekf, x_est = ekf_pid.step(
            z_hat,
            theta_hat,
            z_target,
            forward_speed=dx_true,
            roll_hat=roll_true,
            omega_p=omega_p_true,
            omega_r=omega_r_true,
        )

        if use_ekf_pid_controller:
            delta_e_cmd = delta_e_ekf
            delta_a_cmd = ekf_pid.aileron_command
        else:
            delta_e_cmd = delta_e_raw
            delta_a_cmd = 0.0

        success, state = env.step(delta_e_cmd, delta_a=delta_a_cmd)

        z_next, _dz_next, _x_next, _dx_next, theta_next, _dth_next = state
        x_traj.append(float(state[2]))
        z_traj.append(float(z_next))
        delta_e_traj.append(float(delta_e_cmd))
        post_step_violations = check_spec_violations(z_next, theta_next)
        if post_step_violations:
            terminated_early = True
            termination_step = step
            termination_type = "spec_violation"
            violated_spec = {
                "z": z_next,
                "pitch": theta_next,
                "violations": post_step_violations,
            }
            termination_message = (
                f"Specification violated at step {step}: {', '.join(post_step_violations)}"
            )
            print(termination_message)
            break

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
            terminated_early = True
            termination_step = step
            termination_type = "constraint_violation"
            termination_message = "Simulation terminated early (constraint violation)."
            print(termination_message)
            break
    else:
        print(f"\nCompleted {n_steps} steps without constraint violation.")

    trajectory_plot = None
    if save_trajectory_plot:
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Altitude trajectory
            ax1.plot(x_traj, z_traj, label="Plane trajectory", linewidth=2.5, color="steelblue")
            ax1.axhline(z_target, linestyle="--", linewidth=1.5, label="z_target", color="green")
            if specification is not None:
                ax1.axhline(
                    specification["z_min"],
                    linestyle=":",
                    linewidth=1.2,
                    label="z_min",
                    color="red",
                    alpha=0.7,
                )
                ax1.axhline(
                    specification["z_max"],
                    linestyle=":",
                    linewidth=1.2,
                    label="z_max",
                    color="red",
                    alpha=0.7,
                )
            ax1.set_ylabel("altitude z (m)", fontsize=11)
            ax1.set_title("Flight Trajectory and Control Authority", fontsize=12, fontweight="bold")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="best")
            
            # Elevator deflection
            delta_e_deg = [np.rad2deg(de) for de in delta_e_traj]
            ax2.plot(x_traj, delta_e_deg, label="Elevator deflection", linewidth=2.5, color="darkorange")
            ax2.axhline(0, linestyle="-", linewidth=1, alpha=0.5, color="black")
            ax2.axhline(25.0, linestyle="--", linewidth=1.2, label="+25° limit", color="red", alpha=0.7)
            ax2.axhline(-25.0, linestyle="--", linewidth=1.2, label="-25° limit", color="red", alpha=0.7)
            ax2.set_xlabel("x position (m)", fontsize=11)
            ax2.set_ylabel("elevator deflection (deg)", fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best")
            
            fig.tight_layout()
            fig.savefig(trajectory_plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved trajectory plot to {trajectory_plot_path}")
            trajectory_plot = {
                "saved": True,
                "path": trajectory_plot_path,
                "num_points": len(x_traj),
            }
        except Exception as exc:
            trajectory_plot = {
                "saved": False,
                "path": trajectory_plot_path,
                "error": str(exc),
            }

    return {
        "terminated_early": terminated_early,
        "termination_step": termination_step,
        "termination_type": termination_type,
        "termination_message": termination_message,
        "violated_spec": violated_spec,
        "final_state": state,
        "used_ekf_pid": use_ekf_pid_controller,
        "specification": specification,
        "trajectory_plot": trajectory_plot,
    }


# ── Parameters ────────────────────────────────────────────────────────────────
Z0 = 0.5  # initial altitude (m)
Z_TARGET = 0.5  # target altitude (m)
N_STEPS = 2000  # timesteps
SEED = 42
ENABLE_LATERAL_DAMPER = True
USE_EKF_PID_CONTROLLER = True
GROUND_EFFECT_ENABLED = True  # Enable ground effect aerodynamic correction
SAVE_TRAJECTORY_PLOT = True
TRAJECTORY_PLOT_PATH = "plots/plane_trajectory.png"
SENSOR_ARGS = dict(
    mu_A=0.1,
    sigma_A=0.05,
    mu_k=2 * np.pi,  # ≈ 1 m wavelength
    sigma_k=0.5,
    sigma_eps=0.01,
    p_penetration=0.05,
    alpha_min=0.10,
    alpha_max=10,
    perfect_sensing=False,
)
SPECIFICATION = dict(
    z_min=0.1,
    z_max=1.0,
    pitch_min=-np.deg2rad(15.0),
    pitch_max=np.deg2rad(15.0),
)


if __name__ == "__main__":
    run_agent_env_sensor_loop(
        z0=Z0,
        z_target=Z_TARGET,
        n_steps=N_STEPS,
        seed=SEED,
        enable_lateral_damper=ENABLE_LATERAL_DAMPER,
        use_ekf_pid_controller=USE_EKF_PID_CONTROLLER,
        sensor_args=SENSOR_ARGS,
        specification=SPECIFICATION,
        save_trajectory_plot=SAVE_TRAJECTORY_PLOT,
        trajectory_plot_path=TRAJECTORY_PLOT_PATH,
        ground_effect_enabled=GROUND_EFFECT_ENABLED,
    )
