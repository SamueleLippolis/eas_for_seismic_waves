# src/forward_model.py

import math
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


def compute_AK_Amu(C_percent):
    """
    Giulio's empirical dependence of Krief parameters on clay content.
    """
    A_K = 0.025 * C_percent + 2.37
    A_mu = -0.003 * C_percent + 3.14

    return A_K, A_mu


def invalid_output():
    """
    Standard invalid forward-model output.
    """
    return {
        "Vp": np.nan,
        "Vs": np.nan,
        "sigma": np.nan,
        "rho": np.nan,
        "K_G": np.nan,
        "mu_G": np.nan,
    }


def forward_model(x, constants):
    """
    Forward model aligned with Giulio's notebook.

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
    eps = 1e-18

    try:
        phi, C, S_b, sigma_b, xi = parse_model_parameters(x)
    except Exception:
        return invalid_output()

    C_percent = x["C_percent"]

    if not np.isfinite(sigma_b) or sigma_b <= 0:
        return invalid_output()

    if phi < 0 or phi >= 1:
        return invalid_output()

    # -------------------------
    # Constants
    # -------------------------
    K_q = constants["solids"]["quartz"]["K"]
    mu_q = constants["solids"]["quartz"]["mu"]
    rho_q = constants["solids"]["quartz"]["rho"]
    sigma_q = constants["solids"]["quartz"]["sigma"]

    K_c = constants["solids"]["clay"]["K"]
    mu_c = constants["solids"]["clay"]["mu"]
    rho_c = constants["solids"]["clay"]["rho"]
    sigma_c = constants["solids"]["clay"]["sigma"]

    K_b = constants["fluids"]["brine"]["K"]
    rho_b = constants["fluids"]["brine"]["rho"]

    K_a = constants["fluids"]["air"]["K"]
    rho_a = constants["fluids"]["air"]["rho"]
    sigma_a = constants["fluids"]["air"]["sigma"]

    gamma = constants["conductivity"]["gamma"]

    # -------------------------
    # Bulk density
    # -------------------------
    rho_s = (1.0 - C) * rho_q + C * rho_c
    rho_f = (1.0 - S_b) * rho_a + S_b * rho_b
    rho_bulk = (1.0 - phi) * rho_s + phi * rho_f

    if rho_bulk <= 0.0 or not np.isfinite(rho_bulk):
        return invalid_output()

    # -------------------------
    # Solid fractions
    # -------------------------
    beta_q = 1.0 - C
    beta_c = C

    K_V = beta_q * K_q + beta_c * K_c
    mu_V = beta_q * mu_q + beta_c * mu_c

    if abs(K_V) < eps or abs(mu_V) < eps:
        return invalid_output()

    # -------------------------
    # Hashin-Shtrikman-style averages
    # -------------------------
    K_max = max(K_q, K_c)
    K_min = min(K_q, K_c)

    mu_max = max(mu_q, mu_c)
    mu_min = min(mu_q, mu_c)

    dK = K_c - K_q
    dmu = mu_c - mu_q

    if abs(dK) < eps or abs(dmu) < eps:
        return invalid_output()

    denom_plus = (1.0 / dK) + beta_q / (K_q + (4.0 / 3.0) * mu_max)
    denom_minus = (1.0 / dK) + beta_q / (K_q + (4.0 / 3.0) * mu_min)

    if abs(denom_plus) < eps or abs(denom_minus) < eps:
        return invalid_output()

    K_HS_plus = K_q + (1.0 - beta_q) / denom_plus
    K_HS_minus = K_q + (1.0 - beta_q) / denom_minus
    K_HS = 0.5 * (K_HS_plus + K_HS_minus)

    denom_hq = K_max + 2.0 * mu_max
    denom_hc = K_min + 2.0 * mu_min

    if abs(denom_hq) < eps or abs(denom_hc) < eps:
        return invalid_output()

    hs_shear_q = mu_q + (mu_max / 6.0) * (
        (9.0 * K_max + 8.0 * mu_max) / denom_hq
    )

    hs_shear_c = mu_q + (mu_min / 6.0) * (
        (9.0 * K_min + 8.0 * mu_min) / denom_hc
    )

    denom_mu_plus = (1.0 / dmu) + beta_q / hs_shear_q
    denom_mu_minus = (1.0 / dmu) + beta_q / hs_shear_c

    if abs(denom_mu_plus) < eps or abs(denom_mu_minus) < eps:
        return invalid_output()

    mu_HS_plus = mu_q + (1.0 - beta_q) / denom_mu_plus
    mu_HS_minus = mu_q + (1.0 - beta_q) / denom_mu_minus
    mu_HS = 0.5 * (mu_HS_plus + mu_HS_minus)

    # -------------------------
    # Krief parameters depending on C
    # -------------------------
    A_K, A_mu = compute_AK_Amu(C_percent)

    denom = 1.0 - phi

    if denom <= 1e-15:
        return invalid_output()

    exponent_bulk = A_K / denom
    exponent_shear = xi * A_mu / denom

    one_minus_phi = 1.0 - phi
    bulk_factor = one_minus_phi ** exponent_bulk
    shear_factor = one_minus_phi ** exponent_shear

    K_m_q = (K_HS / K_V) * beta_q * K_q * bulk_factor
    K_m_c = (K_HS / K_V) * beta_c * K_c * bulk_factor
    K_m = K_m_q + K_m_c

    mu_m_q = (mu_HS / mu_V) * beta_q * mu_q * shear_factor
    mu_m_c = (mu_HS / mu_V) * beta_c * mu_c * shear_factor
    mu_m = mu_m_q + mu_m_c

    # -------------------------
    # Fluid bulk modulus
    # -------------------------
    denom_Kf = (1.0 - S_b) / K_a + S_b / K_b

    if abs(denom_Kf) < eps:
        return invalid_output()

    K_f = 1.0 / denom_Kf

    # -------------------------
    # Generalized Gassmann
    # -------------------------
    alpha_q = beta_q - K_m_q / K_q
    alpha_c = beta_c - K_m_c / K_c

    phi_q_prime = alpha_q - beta_q * phi
    phi_c_prime = alpha_c - beta_c * phi

    M_inv = (phi_q_prime / K_q) + (phi_c_prime / K_c) + (phi / K_f)

    if not np.isfinite(M_inv) or abs(M_inv) < eps:
        return invalid_output()

    M = 1.0 / M_inv

    K_G = K_m + (alpha_q + alpha_c) ** 2 * M
    mu_G = mu_m

    if K_G <= 0 or mu_G <= 0:
        return invalid_output()

    # -------------------------
    # Seismic velocities
    # -------------------------
    Vp = math.sqrt((K_G + (4.0 / 3.0) * mu_G) / rho_bulk)
    Vs = math.sqrt(mu_G / rho_bulk)

    # -------------------------
    # Electrical conductivity
    # -------------------------
    term_sigma = (
        (1.0 - phi) * (1.0 - C) * (sigma_q ** gamma)
        + (1.0 - phi) * C * (sigma_c ** gamma)
        + phi * S_b * (sigma_b ** gamma)
        + phi * (1.0 - S_b) * (sigma_a ** gamma)
    )

    if not np.isfinite(term_sigma) or term_sigma <= 0.0:
        return invalid_output()

    sigma_bulk = term_sigma ** (1.0 / gamma)

    return {
        "Vp": Vp,
        "Vs": Vs,
        "sigma": sigma_bulk,
        "rho": rho_bulk,
        "K_G": K_G,
        "mu_G": mu_G,
    }