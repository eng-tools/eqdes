"""
References:
Sullivan, Salawdeh S., Pecker A. (2010). Soil-foundation-structure interaction considerations
 for performance-based design of RC wall structures on shallow foundations.
"""

import numpy as np

from eqdes.extensions.exceptions import DesignError


def displacement_profile_frame(theta_c, heights, hm_factor, foundation=False,
                         fd_height=0.0, theta_f=0.0, verbose=0):
    heights = np.array(heights)
    if foundation:
        displaced_shape = np.zeros(len(heights))
        heights_ss = heights[1:] - fd_height
        displaced_shape[1:] += cal_displaced_shape(theta_c, heights_ss, btype="frame")
        displaced_shape += theta_f * heights
    else:
        displaced_shape = cal_displaced_shape(theta_c, heights, btype="frame")
    displacements = displaced_shape * hm_factor
    if verbose:
        print('theta_c: ', theta_c)
        print('hm_factor: ', hm_factor)
        print('displacements: ', displacements)
    return displacements


def cal_higher_mode_factor(n_storeys, btype="frame"):
    """
    Calculates the higher mode factor according to DDBD12 CL.
    :param n_storeys:
    :param btype:
    :return:
    """
    # Higher mode factor for frame structures
    if btype == "frame":
        if n_storeys < 6:
            factor = 1.0
        elif n_storeys <= 16:
            factor = 1.0 - 0.15 * float((n_storeys - 6.0)) / (16.0 - 6.0)
        else:
            factor = 0.85
    else:
        if n_storeys < 10:
            factor = 1.0
        elif n_storeys <= 16:
            factor = (1.0 - 0.06 * float((n_storeys - 10.0)) / (16.0 - 10.0))
        else:
            factor = 0.85
    return factor


def cal_displaced_shape(theta_c, heights, btype="frame"):
    heights = np.array(heights)
    max_height = max(heights)
    if btype == "frame":
        return theta_c * heights * (4 * max_height - heights) / (4 * max_height - heights[0])


def equivalent_sdof(masses, displacements, heights):
    mass_x_disp = masses * displacements
    mass_x_disp2 = masses * displacements ** 2
    mass_x_disp_x_height = masses * displacements * heights

    delta_d = sum(mass_x_disp2) / sum(mass_x_disp)
    mass_eff = sum(mass_x_disp) / delta_d
    height_eff = sum(mass_x_disp_x_height) / sum(mass_x_disp)

    return delta_d, mass_eff, height_eff

    
def yield_displacement(theta_y, height_eff):  # TODO: check where this comes from?
    return theta_y * height_eff


def yield_displacement_wall(phi_y, heights, max_height):
    """
    The yield displacement of a concrete wall. Eq. 6 (Sullivan et al. 2010).
    :param phi_y: yield curvature
    :param heights: height
    :return:
    """

    return phi_y * heights ** 2 / 2 + phi_y * heights ** 3 / max_height


def conc_frame_yield_drift(fye, youngs_steel, av_bay_length, av_beam_depth):
    """
    Yield drift of a concrete frame from DDBD (Priestley et al. (2007)
    :param fye: Effective yeild strength of reinforcing steel
    :param youngs_steel: Young's modulus of reinforcing steel
    :param av_bay_length: Average bay length
    :param av_beam_depth: average beam depth
    :return:
    """
    return 0.5 * fye / youngs_steel * (av_bay_length / av_beam_depth)


def ductility(delta_current, delta_y):
    """
    Computes the ductility for a given displacement and yield.
    :param delta_current: current displacement
    :param delta_y: yield displacement
    :return:
    """
    return delta_current / delta_y


def reduction_factor(xi, near_field=False):
    if near_field > 1.0:
        # Damping reduction factor
        return (0.07 / (0.02 + xi)) ** 0.25
    else:
        # Damping reduction factor
        return (0.07 / (0.02 + xi)) ** 0.5


def damping_from_reduction_factor(eta):
    """
    Inversion of reduction factor equation
    :param eta:
    :return:
    """
    return (0.07 / eta ** 2) - 0.02


def equivalent_viscous_damping(mu, mtype="concrete", btype="frame"):
    """
    Calculate the equivalent viscous damping based on the ductility and structural type.
    :param mu: Displacement ductility
    :param mtype: material type
    :param btype: building type (e.g. frame or wall)
    :return:
    """
    pie = 3.141
    if mu < 1:
        return 0.05
    if mtype == "concrete":
        if btype == "frame":
            # Equivalent viscous damping for concrete frame
            return 0.05 + 0.565 * (mu - 1) / (mu * pie)
        if btype == "wall":
            # Equivalent viscous damping for concrete wall (Sullivan et al., 2010)
            return 0.05 + 0.444 * (mu - 1) / (mu * pie)


def effective_period(delta_d, eta, corner_disp, corner_period):
    corner_disp_eff = corner_disp * eta
    if delta_d > corner_disp_eff:
        return 0.0
    else:
        return corner_period * delta_d / corner_disp_eff


def effective_period_from_stiffness(mass_eff, k_eff):
    """
    Calculates the effective period based on the mass and stiffness
    :param mass_eff: effective mass
    :param k_eff: effective stiffness
    :return:
    """
    return 2 * 3.141 * np.sqrt(mass_eff / k_eff)


def effective_stiffness_from_base_shear(v_base, disp):
    """
    Calculates the effective stiffness based on the base shear and displacement.
    Typically used in displacement based assessment
    :return:
    """
    return v_base / disp


def displacement_from_effective_period(eta, corner_disp, t_eff, corner_period):
    """
    Displacement of SDOF using displacement-based assessment. # Eq. 11 Millen et al. (2016)
    :param eta: Displacement reduction factor
    :param corner_disp: Corner spectral displacement
    :param t_eff: Effective displacement
    :param corner_period:
    :return:
    """
    if t_eff > corner_period:
        return eta * corner_disp
    return eta * corner_disp * t_eff / corner_period


def effective_stiffness(mass_eff, t_eff):
    """
    Calculates the effective period based on the mass and period
    :param mass_eff: effective mass
    :param t_eff: effective period
    :return:
    """
    return (4 * 3.141 ** 2 * mass_eff) / t_eff ** 2


def design_base_shear(k_eff, delta_d):

    return k_eff * delta_d


def bilinear_load_factor(ductility_current, ductility_max, r):
    """
    Computes the load
    :param ductility_current: Current ductility
    :param ductility_max: Maximum ductility
    :param r: post-yield bi-linear
    :return: factor to reduce maximum load
    """
    hardening_load = r * (ductility_max - 1)
    if ductility_current > ductility_max:
        raise DesignError("Current ductility: {0}, exceeds maximum ductility {1}".format(ductility_current,
                                                                                         ductility_max))
    elif ductility_current > 1.0:
        return 1.0 - hardening_load + r * (ductility_current - 1)
    else:  # Elastic behaviour
        return (1.0 - hardening_load) * ductility_current


def calculate_storey_forces(masses, displacements, v_base):
    mass_x_disp = np.array(masses) * np.array(displacements)
    storey_forces = 0.9 * v_base * mass_x_disp / sum(mass_x_disp)  # Newtons per storey
    storey_forces[-1] = storey_forces[-1] + 0.1 * v_base
    return storey_forces


def p_delta_moment(mass_eff, delta_d):
    return mass_eff * 9.8 * delta_d


def p_delta_base_shear(mass_eff, delta_d, h_eff, v_base):

    moment_pd = p_delta_moment(mass_eff, delta_d)
    pd_factor = moment_pd / v_base * h_eff
    if pd_factor > 0.1:
        c_p_delta = 0.5
        return c_p_delta * moment_pd / h_eff
    else:
        return 0


def yield_curvature(epsilon_y, length, btype="wall"):
    if btype == "wall":
        return 2.0 * epsilon_y / length


def add_foundation(ss_heights, ss_masses, fd_height, fd_mass):
    # add foundation to heights
    heights = list(ss_heights)
    heights.insert(0, 0)
    heights = np.array(heights) + fd_height
    # add foundation to masses
    storey_masses = list(ss_masses)
    storey_masses.insert(0, fd_mass)
    storey_masses = np.array(storey_masses)
    return heights, storey_masses

