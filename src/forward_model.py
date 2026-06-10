# src/forward_model.py

import numpy as np


def load_constants_from_yaml(path):
    """
    Load physical constants from a YAML file.
    """
    import yaml

    with open(path, "r") as f:
        constants = yaml.safe_load(f)

    return constants


def wood_fluid_bulk_modulus(S_b, K_brine, K_air):
    """
    Effective fluid bulk modulus using Wood's model.

    Parameters
    ----------
    S_b : float
        Brine saturation.
    K_brine : float
        Brine bulk modulus [Pa].
    K_air : float
        Air bulk modulus [Pa].

    Returns
    -------
    K_f : float
        Effective fluid bulk modulus [Pa].
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


def dry_rock_moduli_simple(phi, C, constants):
    """
    Simplified Krief dry-rock model.

    This is a first clean implementation.

    We approximate the solid bulk and shear moduli using Voigt averages,
    then apply the Krief porosity correction.

    Later, we can replace this with the full Hashin-Shtrikman version
    from equations A.6-A.9.
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
    exponent_mu = A_mu / (1.0 - phi)

    K_m = K_s * (1.0 - phi) ** exponent_K
    mu_m = mu_s * (1.0 - phi) ** exponent_mu

    return K_m, mu_m


def gassmann_bulk_modulus_simple(phi, C, S_b, K_m, constants):
    """
    Simplified Gassmann equation for the saturated bulk modulus.

    This uses an effective solid bulk modulus K_s obtained by Voigt average.
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

    Parameters
    ----------
    x : dict
        Dictionary with:
        - phi: porosity
        - C: clay content
        - S_b: brine saturation
        - sigma_b: brine conductivity [S/m]

    constants : dict
        Physical constants loaded from YAML.

    Returns
    -------
    output : dict
        Dictionary with:
        - Vp: P-wave velocity [m/s]
        - Vs: S-wave velocity [m/s]
        - sigma: bulk conductivity [S/m]
        - rho: bulk density [kg/m^3]
        - K_G: saturated bulk modulus [Pa]
        - mu_G: saturated shear modulus [Pa]
    """
    phi = x["phi"]
    C = x["C"]
    S_b = x["S_b"]
    sigma_b = x["sigma_b"]

    K_m, mu_m = dry_rock_moduli_simple(phi, C, constants)

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