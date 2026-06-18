# src/forward_model.py

import numpy as np

from src.config_utils import load_yaml


def load_constants_from_yaml(path):
    """
    Load physical constants from a YAML file.
    Kept for compatibility with previous scripts.
    """
    return load_yaml(path)


def parse_model_parameters(x):
    """
    Convert model parameters from Giulio-style format to physical fractions.

    Input x:
        phi_percent : porosity in percent
        C_percent : clay content in percent
        S_b_percent : brine saturation in percent
        sigma_b_inv : inverse brine conductivity
        xi : empirical shear-modulus correction parameter
    """
    phi = x["phi_percent"] / 100.0
    C = x["C_percent"] / 100.0
    S_b = x["S_b_percent"] / 100.0

    sigma_b_inv = x["sigma_b_inv"]
    sigma_b = 1.0 / sigma_b_inv

    xi = x["xi"]

    return phi, C, S_b, sigma_b, xi


def wood_fluid_bulk_modulus(S_b, K_brine, K_air):
    """
    Effective fluid bulk modulus using Wood's model.
    """
    return 1.0 / ((1.0 - S_b) / K_air + S_b / K_brine)


def effective_density(phi, C, S_b, constants):
    """
    Compute bulk density of the saturated sediment.
    """
    rho_q = constants["solids"]["quartz"]["rho"]
    rho_c = constants["solids"]["clay"]["rho"]
    rho_b = constants["fluids"]["brine"]["rho"]
    rho_a = constants["fluids"]["air"]["rho"]

    rho_s = (1.0 - C) * rho_q + C * rho_c
    rho_f = (1.0 - S_b) * rho_a + S_b * rho_b

    rho = (1.0 - phi) * rho_s + phi * rho_f

    return rho


def dry_rock_moduli_simple(phi, C, xi, constants):
    """
    Simplified Krief dry-rock model.

    xi modifies the shear modulus exponent.
    """
    K_q = constants["solids"]["quartz"]["K"]
    K_c = constants["solids"]["clay"]["K"]
    mu_q = constants["solids"]["quartz"]["mu"]
    mu_c = constants["solids"]["clay"]["mu"]

    A_K = constants["krief"]["A_K"]
    A_mu = constants["krief"]["A_mu"]

    beta_q = 1.0 - C
    beta_c = C

    K_s = beta_q * K_q + beta_c * K_c
    mu_s = beta_q * mu_q + beta_c * mu_c

    exponent_K = A_K / (1.0 - phi)
    exponent_mu = xi * A_mu / (1.0 - phi)

    K_m = K_s * (1.0 - phi) ** exponent_K
    mu_m = mu_s * (1.0 - phi) ** exponent_mu

    return K_m, mu_m


def gassmann_bulk_modulus_simple(phi, C, S_b, K_m, constants):
    """
    Simplified Gassmann equation for the saturated bulk modulus.
    """
    K_q = constants["solids"]["quartz"]["K"]
    K_c = constants["solids"]["clay"]["K"]

    K_brine = constants["fluids"]["brine"]["K"]
    K_air = constants["fluids"]["air"]["K"]

    beta_q = 1.0 - C
    beta_c = C

    K_s = beta_q * K_q + beta_c * K_c
    K_f = wood_fluid_bulk_modulus(S_b, K_brine, K_air)

    numerator = (1.0 - K_m / K_s) ** 2
    denominator = (
        phi / K_f
        + (1.0 - phi) / K_s
        - K_m / (K_s ** 2)
    )

    K_G = K_m + numerator / denominator

    return K_G


def electrical_conductivity(phi, C, S_b, sigma_b, constants):
    """
    Bulk electrical conductivity using the CRIM model.
    """
    sigma_q = constants["solids"]["quartz"]["sigma"]
    sigma_c = constants["solids"]["clay"]["sigma"]
    sigma_a = constants["fluids"]["air"]["sigma"]

    gamma = constants["conductivity"]["gamma"]

    term = (
        (1.0 - phi) * (1.0 - C) * sigma_q ** gamma
        + (1.0 - phi) * C * sigma_c ** gamma
        + phi * S_b * sigma_b ** gamma
        + phi * (1.0 - S_b) * sigma_a ** gamma
    )

    sigma = term ** (1.0 / gamma)

    return sigma


def forward_model(x, constants):
    """
    Complete forward model.

    Input x:
        phi_percent
        C_percent
        S_b_percent
        sigma_b_inv
        xi

    Output:
        Vp
        Vs
        sigma
        rho
        K_G
        mu_G
    """
    phi, C, S_b, sigma_b, xi = parse_model_parameters(x)

    K_m, mu_m = dry_rock_moduli_simple(phi, C, xi, constants)

    K_G = gassmann_bulk_modulus_simple(phi, C, S_b, K_m, constants)
    mu_G = mu_m

    rho = effective_density(phi, C, S_b, constants)

    Vp = np.sqrt((K_G + 4.0 * mu_G / 3.0) / rho)
    Vs = np.sqrt(mu_G / rho)

    sigma = electrical_conductivity(phi, C, S_b, sigma_b, constants)

    return {
        "Vp": Vp,
        "Vs": Vs,
        "sigma": sigma,
        "rho": rho,
        "K_G": K_G,
        "mu_G": mu_G,
    }