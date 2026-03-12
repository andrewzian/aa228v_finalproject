import numpy as np
from pyfly.pyfly import PyFly


class AircraftEnvironment:
    """
    Wraps PyFly to expose a 6-state vector s = [z, dz, x, dx, theta, dtheta]
    for a fixed-wing UAV flying in ground effect over water.

    State mapping (NED conventions -> inertial-frame approximations):
        z      = -position_d   (altitude above waterline, positive up, m)
        dz     = -velocity_w   (vertical velocity, body-frame approx, m/s)
        x      =  position_n   (longitudinal position, m)
        dx     =  velocity_u   (forward airspeed, m/s)
        theta  =  pitch        (pitch angle, rad)
        dtheta =  omega_q      (pitch rate, rad/s)

    Notes:
        - dz uses body-frame velocity_w as an approximation. For small pitch
          angles in IGE flight this is sufficient; a full inertial transform
          can be substituted later if needed.
        - The X8 airframe uses elevons (not a separate elevator). PyFly's
          actuation system internally mixes elevator/aileron commands into
          elevon_left/elevon_right deflections.
        - Commands to sim.step() are in radians for elevator/aileron and
          [0, 1] for throttle, matching the PIDController output convention.
        - Turbulence is disabled; atmospheric disturbances are handled by the
          stochastic sensor layer instead.
    """

    DEFAULT_THROTTLE = 0.5  # ~17 m/s cruise starting point; tune empirically

    def __init__(self, config_kw=None):
        """
        :param config_kw: (dict or None) optional overrides to PyFly config.
                          Merged on top of the low-altitude-safe defaults.
        """
        base_kw = {
            "turbulence": False,
            "wind_magnitude_min": 0,
            "wind_magnitude_max": 0,
        }
        if config_kw is not None:
            base_kw.update(config_kw)

        self.sim = PyFly(config_kw=base_kw)
        self.throttle = self.DEFAULT_THROTTLE

    def seed(self, s):
        """Seed the PyFly RNG."""
        self.sim.seed(s)

    def reset(self, z0=5.0, x0=0.0, Va0=17.0, theta0=0.0):
        """
        Reset the simulator to specified initial conditions.

        :param z0:     initial altitude above water surface (m, positive up)
        :param x0:     initial longitudinal position (m)
        :param Va0:    initial airspeed (m/s)
        :param theta0: initial pitch angle (rad)
        :return:       initial state vector np.ndarray [z, dz, x, dx, theta, dtheta]
        """
        init_state = {
            "position_n": x0,
            "position_e": 0.0,
            "position_d": -z0,   # NED: negative value = above ground/water
            "velocity_u": Va0,
            "velocity_v": 0.0,
            "velocity_w": 0.0,
            "roll":    0.0,
            "pitch":   theta0,
            "yaw":     0.0,
            "omega_p": 0.0,
            "omega_q": 0.0,
            "omega_r": 0.0,
        }
        self.sim.reset(state=init_state)
        return self.get_state()

    def step(self, delta_e, delta_a=0.0):
        """
        Advance the simulation by one timestep (dt = 0.01 s by default).

        :param delta_e: elevator deflection (rad); positive = nose-up
                        (controller convention: positive delta_e raises altitude).
                        Internally negated before passing to PyFly because PyFly's
                        X8 elevon model uses the opposite sign (positive PyFly
                        command → trailing-edge down on a flying-wing elevon →
                        nose-DOWN pitching moment).
        :param delta_a: aileron command (rad), positive = roll-right command.
        :return: (success, state)
                 success -- bool, False if a PyFly constraint was violated
                 state   -- np.ndarray [z, dz, x, dx, theta, dtheta]
        """
        commands = [-delta_e, delta_a, self.throttle]  # negate: our +δe = nose-up, PyFly +δe = nose-down
        success, info = self.sim.step(commands)
        return success, self.get_state()

    def get_lateral_state(self):
        """
        Extract lateral-directional states needed for stabilization.

        :return: np.ndarray [roll, omega_p, omega_r]
        """
        roll = self.sim.state["roll"].value
        omega_p = self.sim.state["omega_p"].value
        omega_r = self.sim.state["omega_r"].value
        return np.array([roll, omega_p, omega_r])

    def get_state(self):
        """
        Extract the 6-state vector from the current PyFly state.

        :return: np.ndarray [z, dz, x, dx, theta, dtheta]
        """
        z   = -self.sim.state["position_d"].value
        dz  = -self.sim.state["velocity_w"].value  # body-frame approximation
        x   =  self.sim.state["position_n"].value
        dx  =  self.sim.state["velocity_u"].value
        th  =  self.sim.state["pitch"].value
        dth =  self.sim.state["omega_q"].value
        return np.array([z, dz, x, dx, th, dth])
