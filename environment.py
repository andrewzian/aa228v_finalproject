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
    GROUND_EFFECT_REFERENCE_ALTITUDE = 3.0  # where ground effect becomes negligible (m)

    def __init__(self, config_kw=None, ground_effect_enabled=False):
        """
        :param config_kw: (dict or None) optional overrides to PyFly config.
                          Merged on top of the low-altitude-safe defaults.
        :param ground_effect_enabled: (bool) Enable ground effect aerodynamic correction.
                          When True, induced drag is reduced at low altitudes,
                          improving lift efficiency. Significant below ~3m altitude.
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
        self.ground_effect_enabled = ground_effect_enabled

    def seed(self, s):
        """Seed the PyFly RNG."""
        self.sim.seed(s)

    def get_ground_effect_factor(self, altitude):
        """
        Compute ground effect correction factor.
        
        Ground effect reduces induced drag as altitude decreases, improving
        aerodynamic efficiency. This is modeled as an altitude-dependent
        multiplicative efficiency gain:
        
            factor = 1.0 + (1.6 - 1.0) / (1 + altitude / h_ref)
        
        where h_ref = reference altitude at which effect is significant.
        
        Returns factor >= 1.0, where:
        - factor ≈ 1.6 at ground level (h=0, maximum ground effect)
        - factor ≈ 1.3 at h=h_ref (moderate effect)
        - factor → 1.0 as h → ∞ (negligible ground effect)
        
        :param altitude: current altitude above ground (m)
        :return: ground effect correction factor (float)
        """
        if altitude <= 0:
            altitude = 0.0001  # clamp to small positive value
        
        # Altitude-normalized ground effect: efficiency gain decays with altitude
        # Max gain is 0.6 (60% improvement from 1.0 baseline)
        # Decay around h_ref ≈ 0.8 m (where effect is most significant)
        h_ref = 0.5  # characteristic altitude for ground effect
        max_gain = 0.6
        
        factor = 1.0 + max_gain / (1.0 + altitude / h_ref)
        return factor

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
        
        Notes:
            When ground effect is enabled, an aerodynamic correction is applied
            to account for reduced induced drag at low altitudes. This improves
            altitude-holding efficiency and affects control responses.
        """
        commands = [-delta_e, delta_a, self.throttle]  # negate: our +δe = nose-up, PyFly +δe = nose-down
        success, info = self.sim.step(commands)
        
        # Apply ground effect correction if enabled
        if self.ground_effect_enabled:
            z = -self.sim.state["position_d"].value
            ge_factor = self.get_ground_effect_factor(z)
            
            # Ground effect increases lift efficiency, reducing sink rate.
            # Apply correction by scaling the vertical velocity (velocity_w).
            # Factor > 1.0 means better lift, so we reduce negative velocity_w.
            velocity_w_current = self.sim.state["velocity_w"].value
            if velocity_w_current > 0:  # sinking (positive w is down in body frame)
                # Reduce sink rate proportionally to ground effect factor
                # correction = (ge_factor - 1.0) represents the efficiency gain
                correction = velocity_w_current * (1.0 - 1.0 / ge_factor)
                velocity_w_corrected = velocity_w_current - correction
                # Clamp to reasonable bounds to avoid over-correction
                velocity_w_corrected = np.clip(velocity_w_corrected, -5.0, 0.5)
                self.sim.state["velocity_w"].value = velocity_w_corrected
        
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
