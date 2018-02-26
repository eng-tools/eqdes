import numpy as np
from eqdes.extensions.exceptions import DesignError


def foundation_rotation_reduction_factor_millen(cor_norm_rot):
    """
    Returns the displacement reduction factor of the foundation for a given corrected normalised rotation.

    :param cor_norm_rot:
    """

    dmf = np.sqrt(1.0 / (1.0 + 5.0 * (1 - np.exp(-0.15 * cor_norm_rot))))
    return dmf


def foundation_rotation_reduction_factor(cor_norm_rot, method="Millen"):
    """
    Returns the displacement reduction factor of the foundation for a given rotation
    :param cor_norm_rot:
    """

    func_map = {
        "Millen": foundation_rotation_reduction_factor_millen,

    }
    return func_map[method](cor_norm_rot)


def foundation_shear_reduction_factor():
    """
    Returns the displacement reduction factor for soil-foundation shear deformation.
    :param cor_norm_rot:
    """
    # According to Millen (2016)
    return 0.76


def foundation_stiffness_ratio_paolucci(ALR, FR, DR):
    """
    From Paolucci et al. (2013)
    Returns the degradation of foundation stiffness for a given foundation rotation
    :param ALR: axial load ratio
    :param FR: foundation rotation
    :param DR: soil relative density
    """
    # TODO: refactor into a dictionary
    if DR == 90:
        Nratio = [2, 3, 4.5, 6, 7.5, 9, 10, 15, 20, 25, 30]
        a = [
            458.36,
            281.95,
            262.81,
            292.81,
            324.76,
            378.05,
            415.5,
            575.36,
            1010.99,
            2461.06,
            5192.13]
        m = [1.30, 1.11, 1.00, 0.94, 0.91, 0.89, 0.88, 0.83, 0.86, 0.95, 1.02]
    elif DR == 60:
        Nratio = [2, 3, 4.5, 6, 7.5, 9, 10, 15, 20, 25, 30]
        a = [
            686.26,
            386.24,
            339.87,
            352.13,
            398.44,
            433.12,
            452.44,
            653.02,
            1219.47,
            2461.06,
            5192.13]
        m = [1.30, 1.11, 0.98, 0.92, 0.89, 0.86, 0.84, 0.79, 0.83, 0.89, 0.96]

    for A in range(len(Nratio) - 1):
        if ALR <= Nratio[A + 1] and ALR >= Nratio[A]:
            stiffvals = np.zeros((2))
            stiffvals[0] = 1.0 / (1.0 + a[A] * FR ** m[A])
            stiffvals[1] = 1.0 / (1.0 + a[A + 1] * FR ** m[A + 1])
            stiffRatio = np.interp(ALR, [Nratio[A], Nratio[A + 1]], stiffvals)

    return stiffRatio


def foundation_damping_paolucci(ALR, foundation_rotation, DR):
    """
    Returns the damping of the foundation for a given rotation
    :param ALR: Axial load ratio
    :param foundation_rotation:
    :param DR:
    """
    # TODO: refactor into a dictionary
    zeta_min = 0.036
    if DR == 90:
        Nratio = [2, 3, 4.5, 6, 7.5, 9, 10, 15, 20, 25, 30]
        alpha = [
            27.73,
            32.76,
            43.93,
            62.25,
            66.96,
            85.08,
            95.60,
            164.42,
            233.70,
            305.97,
            382.51]  # 4th number a bit high
        zeta_max = 0.25
    elif DR == 60:
        Nratio = [2, 3, 4.5, 6, 7.5, 9, 10, 15, 20, 25, 30]
        alpha = [
            39.39,
            47.61,
            67.79,
            90.64,
            104.49,
            119.20,
            130.85,
            210.42,
            285.15,
            367.70,
            442.47]
        zeta_max = 0.37

    for A in range(len(Nratio) - 1):
        if ALR <= Nratio[A + 1] and ALR >= Nratio[A]:
            dmpingvalues = np.zeros((2))
            dmpingvalues[0] = (zeta_min + (zeta_max - zeta_min) *
                (1 - np.exp(-alpha[A] * foundation_rotation)))
            dmpingvalues[1] = (zeta_min + (zeta_max - zeta_min) *
                (1 - np.exp(-alpha[A + 1] * foundation_rotation)))
            damping = np.interp(ALR, [Nratio[A], Nratio[A + 1]], dmpingvalues)

    return damping


def foundation_rotation_stiffness_ratio_millen(cor_norm_rot):
    """
    From Millen Thesis (2016)
    Returns the degradation of foundation stiffness for a given normalised foundation rotation.
    """
    if hasattr(cor_norm_rot, "__len__"):
        stiff_values = []
        for cnr in cor_norm_rot:
            stiff_values.append(foundation_rotation_stiffness_ratio_millen(cnr))
        return np.array(stiff_values)

    inter = 0.8
    slope = -0.04
    if cor_norm_rot > 100:
        raise DesignError('cor_norm_rot exceeds equation limits. %.3f <= 100' % cor_norm_rot)
    stiff_ratio = min(1.0, -0.7 * (1 - np.exp(-0.18 * cor_norm_rot)) + inter + slope * np.log10(cor_norm_rot))

    return stiff_ratio


def foundation_rotation_stiffness_ratio(cor_norm_rot, method="Millen"):
    """
    Returns the degradation of foundation stiffness for a given normalised foundation rotation
    """
    func_map = {
        "Millen": foundation_rotation_stiffness_ratio_millen,

    }
    return func_map[method](cor_norm_rot)


def calculate_pseudo_uplift_angle(weight, width, k_f_0, axial_load_ratio, alpha, zeta=1.5):
    """
    Calculates the pseudo uplift angle according to Chatzigogos et al. (2011)
    :param weight:
    :param width:
    :param k_f_0:
    :param axial_load_ratio:
    :param alpha:
    :param zeta:
    :return:
    """
    return (weight * width / alpha / k_f_0) * np.exp(-zeta * 1.0 / axial_load_ratio)


def calculate_corrected_normalised_rotation(norm_rot, bhr):
    """
    Corrected normalised rotation according to Millen (2015)
    Correction is for shear force
    Normalisation is against pseudo uplift angle
    :param norm_rot: Normalised rotation angle
    :param bhr: Ratio of Foundation width to Effective height of superstructure mass
    :return:
    """
    return norm_rot ** (1 - 0.2 * bhr) * 10 ** (.25 * bhr)


def system_reduction_factor(delta_ss, delta_frot, delta_fshear, eta_ss, eta_frot, eta_fshear):
    """
    Calculates the system displacement reduction factor based on the foundation and superstrucutre
    displacement reduction factors.
    :param delta_ss: superstructure displacement
    :param delta_frot: displacement due to foundation rotation
    :param delta_fshear: displacement due to soil-foundation shear deformation
    :param eta_ss: superstructure displacement reduction factor
    :param eta_frot: foundation rotation displacement reduction factor
    :param eta_fshear: soil foundation shear deformation displacement reduction factor
    :return:
    """
    delta_total = delta_ss + delta_frot + delta_fshear
    return (delta_ss * eta_ss + delta_frot * eta_frot + delta_fshear * eta_fshear) / delta_total


def bearing_capacity(f_area, soil_q):
    return f_area * soil_q




