"""
PyFly integration smoke test.

Validates AircraftEnvironment by running a fixed number of steps with zero
elevator input and printing the state vector at each step.

Expected behavior:
  - z  starts at ~5 m and stays roughly constant (slight sink/climb OK)
  - x  increases by ~Va0 * dt per step (~0.17 m at 17 m/s, dt=0.01)
  - theta and dtheta near zero for zero elevator
  - success=True for all steps (no constraint violations)
"""

from environment import AircraftEnvironment

# ── Test parameters ──────────────────────────────────────────────────────────
Z0    = 0.2122   # initial altitude (m)
X0    = 0.0   # initial longitudinal position (m)
VA0   = 17.0  # initial airspeed (m/s)
N_STEPS = 200  # number of timesteps to run (2 s at 100 Hz)

# ── Setup ─────────────────────────────────────────────────────────────────────
env = AircraftEnvironment()
env.seed(42)
state = env.reset(z0=Z0, x0=X0, Va0=VA0)

print(f"{'step':>5}  {'z':>7}  {'dz':>7}  {'x':>8}  {'dx':>7}  {'theta':>8}  {'dtheta':>8}  success")
print("-" * 72)
print(f"{'  0':>5}  {state[0]:7.3f}  {state[1]:7.3f}  {state[2]:8.3f}  {state[3]:7.3f}  {state[4]:8.4f}  {state[5]:8.4f}  -")

# ── Run loop ──────────────────────────────────────────────────────────────────
for step in range(1, N_STEPS + 1):
    success, state = env.step(delta_e=0.0)

    if step % 20 == 0 or not success:
        print(
            f"{step:5d}  "
            f"{state[0]:7.3f}  "
            f"{state[1]:7.3f}  "
            f"{state[2]:8.3f}  "
            f"{state[3]:7.3f}  "
            f"{state[4]:8.4f}  "
            f"{state[5]:8.4f}  "
            f"{'OK' if success else 'FAIL'}"
        )

    if not success:
        print("Simulation terminated early.")
        break
else:
    print(f"\nCompleted {N_STEPS} steps without constraint violation.")
