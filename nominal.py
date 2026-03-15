import numpy as np

ROLLOUT_ARGS = dict(
    mu_z0=0.5,
    sigma_z0=0.05,
    sensor_args=dict(
        mu_A=0.0,
        sigma_A=0.1,
        mu_k=2 * np.pi,  # wavelength = k / 2pi
        sigma_k=0.5,
        sigma_eps=0.01,
        p_penetration=0.05,
        alpha_min=0.10,
        alpha_max=10.0,
        perfect_sensing=False,
    ),
)

SPECIFICATION = dict(
    z_min=0.3,
    z_max=0.7,
    pitch_min=-np.deg2rad(15.0),
    pitch_max=np.deg2rad(15.0),
)