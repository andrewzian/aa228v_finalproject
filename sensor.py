"""
Stochastic LiDAR altimeter sensor model for IGE flight over water.

Sensor equation (per PDF):

    ẑ_t = { z_t + d(x; α)           with probability p_penetration
            z_t + η(x; A, k) + ε_t  with probability 1 - p_penetration

where:
    η(x; A, k) = A·sin(k·x)        choppy water surface displacement
    A  ~ N(μ_A, σ_A)                wave amplitude    (m),   sampled once per rollout
    k  ~ N(μ_k, σ_k)                wave number (rad/m),     sampled once per rollout
    ε_t ~ N(0, σ_ε²)                vehicle vibration noise, sampled each timestep
    d(x; α) = α                     LiDAR water-penetration depth (m, positive),
                                    modeled as constant for a given rollout
    α  ~ Uniform(α_min, α_max)      penetration depth,       sampled once per rollout

Pitch is read perfectly: θ̂ = θ.

Environmental parameters A, k, α are part of the "initial state" for a rollout
and are re-sampled on each call to reset().
"""

import numpy as np


class LiDARSensor:
    """
    Stochastic LiDAR altimeter with wave distortion and water-penetration effects.

    Parameters (all keyword args, with IGE-appropriate defaults):
        mu_A          mean wave amplitude (m)
        sigma_A       std of wave amplitude (m)
        mu_k          mean wave number (rad/m)  — default ≈ 1 m wavelength
        sigma_k       std of wave number (rad/m)
        sigma_eps     std of per-step vibration noise (m)
        p_penetration probability that LiDAR penetrates the water surface
        alpha_min     minimum water-penetration depth (m)
        alpha_max     maximum water-penetration depth (m)
        perfect_sensing if True, force ideal sensing (z_hat=z, theta_hat=theta)
    """

    def __init__(
        self,
        mu_A=0.05,
        sigma_A=0.02,
        mu_k=2 * np.pi,       # ≈ 1 m wavelength
        sigma_k=0.5,
        sigma_eps=0.01,
        p_penetration=0.05,
        alpha_min=0.10,
        alpha_max=10,
        perfect_sensing=False,
    ):
        self.perfect_sensing = perfect_sensing

        if self.perfect_sensing:
            mu_A = 0.0
            sigma_A = 0.0
            mu_k = 0.0
            sigma_k = 0.0
            sigma_eps = 0.0
            p_penetration = 0.0
            alpha_min = 0.0
            alpha_max = 0.0

        self.mu_A = mu_A
        self.sigma_A = sigma_A
        self.mu_k = mu_k
        self.sigma_k = sigma_k
        self.sigma_eps = sigma_eps
        self.p_penetration = p_penetration
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.rng = np.random.default_rng()

        # Per-rollout environmental parameters (set by reset)
        self.A = None
        self.k = None
        self.alpha = None

    def seed(self, s):
        """Seed the RNG."""
        self.rng = np.random.default_rng(s)

    def reset(self):
        """
        Sample per-rollout environmental parameters.
        Must be called before the first measure() of each episode.
        """
        if self.perfect_sensing:
            self.A = 0.0
            self.k = 0.0
            self.alpha = 0.0
        else:
            self.A     = self.rng.normal(self.mu_A, self.sigma_A)
            self.k     = self.rng.normal(self.mu_k, self.sigma_k)
            self.alpha = self.rng.uniform(self.alpha_min, self.alpha_max)

    def measure(self, z, x, theta):
        """
        Return a noisy altitude and (perfect) pitch measurement.

        :param z:     true altitude above waterline (m)
        :param x:     longitudinal position (m), drives wave phase
        :param theta: true pitch angle (rad)
        :return: (z_hat, theta_hat)
                 z_hat     -- noisy LiDAR altitude reading (m)
                 theta_hat -- pitch reading (= theta, perfect)
        """
        if self.perfect_sensing:
            return z, theta

        if self.rng.random() < self.p_penetration:
            # LiDAR beam penetrates the water surface; reads deeper than reality.
            # α is a positive offset (sensor over-reads altitude).
            z_hat = z + self.alpha
        else:
            eta   = self.A * np.sin(self.k * x)          # wave surface displacement
            eps   = self.rng.normal(0.0, self.sigma_eps)  # vibration noise
            z_hat = z + eta + eps

        return z_hat, theta  # pitch is perfect
