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
        nominal_params optional dict describing a second sensor distribution used
                       only for scoring the sampled trajectory likelihood
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
        nominal_params=None,
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

        self.proposal_params = {
            "mu_A": self.mu_A,
            "sigma_A": self.sigma_A,
            "mu_k": self.mu_k,
            "sigma_k": self.sigma_k,
            "sigma_eps": self.sigma_eps,
            "p_penetration": self.p_penetration,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
        }
        self.nominal_params = self._build_scoring_params(nominal_params)

        self.rng = np.random.default_rng()

        # Per-rollout environmental parameters (set by reset)
        self.A = None
        self.k = None
        self.alpha = None

        # Per-rollout log-likelihood bookkeeping.
        self.proposal_initial_log_likelihood = 0.0
        self.proposal_step_log_likelihood = 0.0
        self.proposal_total_log_likelihood = 0.0
        self.nominal_initial_log_likelihood = 0.0
        self.nominal_step_log_likelihood = 0.0
        self.nominal_total_log_likelihood = 0.0

        # Backward-compatible aliases for proposal likelihood.
        self.initial_log_likelihood = 0.0
        self.step_log_likelihood = 0.0
        self.total_log_likelihood = 0.0

    def _build_scoring_params(self, overrides):
        """Construct a scoring-parameter dict, defaulting to proposal params."""
        params = dict(self.proposal_params)
        if overrides is not None:
            params.update(overrides)

        if params.pop("perfect_sensing", False):
            params.update(
                {
                    "mu_A": 0.0,
                    "sigma_A": 0.0,
                    "mu_k": 0.0,
                    "sigma_k": 0.0,
                    "sigma_eps": 0.0,
                    "p_penetration": 0.0,
                    "alpha_min": 0.0,
                    "alpha_max": 0.0,
                }
            )

        return params

    @staticmethod
    def _normal_logpdf(x, mu, sigma):
        """Log-density of N(mu, sigma^2), robust to sigma=0 edge cases."""
        if sigma < 0:
            return -np.inf
        if sigma == 0:
            return 0.0 if np.isclose(x, mu) else -np.inf
        variance = sigma * sigma
        return -0.5 * np.log(2.0 * np.pi * variance) - ((x - mu) ** 2) / (2.0 * variance)

    @staticmethod
    def _uniform_logpdf(x, lower, upper):
        """Log-density of Uniform(lower, upper), robust to degenerate interval."""
        width = upper - lower
        if width < 0:
            return -np.inf
        if width == 0:
            return 0.0 if np.isclose(x, lower) else -np.inf
        if x < lower or x > upper:
            return -np.inf
        return -np.log(width)

    @staticmethod
    def _bernoulli_logpmf(is_success, p_success):
        """Log-probability of a Bernoulli outcome with edge-case handling."""
        if p_success < 0.0 or p_success > 1.0:
            return -np.inf
        if is_success:
            if p_success == 0.0:
                return -np.inf
            return np.log(p_success)
        if p_success == 1.0:
            return -np.inf
        return np.log(1.0 - p_success)

    def seed(self, s):
        """Seed the RNG."""
        self.rng = np.random.default_rng(s)

    def _initial_log_likelihood_under_params(self, params):
        """Score the rollout-level sampled parameters under a given distribution."""
        return (
            self._normal_logpdf(self.A, params["mu_A"], params["sigma_A"])
            + self._normal_logpdf(self.k, params["mu_k"], params["sigma_k"])
            + self._uniform_logpdf(self.alpha, params["alpha_min"], params["alpha_max"])
        )

    def _step_log_likelihood_under_params(self, penetrated, eps, params):
        """Score one sampled per-step disturbance under a given distribution."""
        step_log_likelihood = self._bernoulli_logpmf(
            penetrated,
            params["p_penetration"],
        )
        if not penetrated:
            step_log_likelihood += self._normal_logpdf(eps, 0.0, params["sigma_eps"])
        return step_log_likelihood

    def reset(self):
        """
        Sample per-rollout environmental parameters.
        Must be called before the first measure() of each episode.
        """
        self.A = self.rng.normal(self.mu_A, self.sigma_A)
        self.k = self.rng.normal(self.mu_k, self.sigma_k)
        self.alpha = self.rng.uniform(self.alpha_min, self.alpha_max)

        self.proposal_initial_log_likelihood = self._initial_log_likelihood_under_params(
            self.proposal_params
        )
        self.nominal_initial_log_likelihood = self._initial_log_likelihood_under_params(
            self.nominal_params
        )

        self.proposal_step_log_likelihood = 0.0
        self.proposal_total_log_likelihood = self.proposal_initial_log_likelihood
        self.nominal_step_log_likelihood = 0.0
        self.nominal_total_log_likelihood = self.nominal_initial_log_likelihood

        self.initial_log_likelihood = self.proposal_initial_log_likelihood
        self.step_log_likelihood = self.proposal_step_log_likelihood
        self.total_log_likelihood = self.proposal_total_log_likelihood

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
        penetrated = self.rng.random() < self.p_penetration

        if penetrated:
            # LiDAR beam penetrates the water surface; reads deeper than reality.
            # α is a positive offset (sensor over-reads altitude).
            z_hat = z + self.alpha
            eps = None
        else:
            eta   = self.A * np.sin(self.k * x)          # wave surface displacement
            eps   = self.rng.normal(0.0, self.sigma_eps)  # vibration noise
            z_hat = z + eta + eps

        proposal_step_log_likelihood = self._step_log_likelihood_under_params(
            penetrated,
            eps,
            self.proposal_params,
        )
        nominal_step_log_likelihood = self._step_log_likelihood_under_params(
            penetrated,
            eps,
            self.nominal_params,
        )

        self.proposal_step_log_likelihood += proposal_step_log_likelihood
        self.proposal_total_log_likelihood = (
            self.proposal_initial_log_likelihood + self.proposal_step_log_likelihood
        )
        self.nominal_step_log_likelihood += nominal_step_log_likelihood
        self.nominal_total_log_likelihood = (
            self.nominal_initial_log_likelihood + self.nominal_step_log_likelihood
        )

        self.step_log_likelihood = self.proposal_step_log_likelihood
        self.total_log_likelihood = self.proposal_total_log_likelihood

        return z_hat, theta  # pitch is perfect

    def get_log_likelihood_breakdown(self):
        """Return proposal-distribution trajectory log-likelihood components."""
        return {
            "initial": float(self.initial_log_likelihood),
            "stepwise": float(self.step_log_likelihood),
            "total": float(self.total_log_likelihood),
        }

    def get_log_likelihoods(self):
        """Return proposal and nominal trajectory likelihoods plus log-weight."""
        return {
            "proposal": {
                "initial": float(self.proposal_initial_log_likelihood),
                "stepwise": float(self.proposal_step_log_likelihood),
                "total": float(self.proposal_total_log_likelihood),
            },
            "nominal": {
                "initial": float(self.nominal_initial_log_likelihood),
                "stepwise": float(self.nominal_step_log_likelihood),
                "total": float(self.nominal_total_log_likelihood),
            },
            "log_weight": float(
                self.nominal_total_log_likelihood - self.proposal_total_log_likelihood
            ),
        }
