import numpy as np
import geofound as gf
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


def calc_foundation_rotational_stiffness_ratio_millen_et_al_2018(cor_norm_rot):
    """
    From Millen Thesis (2016)
    Returns the degradation of foundation stiffness for a given normalised foundation rotation.
    """
    if hasattr(cor_norm_rot, "__len__"):
        stiff_values = []
        for cnr in cor_norm_rot:
            stiff_values.append(calc_foundation_rotational_stiffness_ratio_millen_et_al_2018(cnr))
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


def calc_fd_rot_via_millen_et_al_2020_alt_form(k_rot_el, l_in, n_load, n_cap, psi, ms, h_eff):
    if n_load >= n_cap:
        return None
    h_0 = 2.0 * k_rot_el
    inv_lamb = np.sqrt((2 * ms / (l_in * n_load)) ** 2 * ((l_in / (2 * h_eff * psi)) ** 2 + 1 / (1 - n_load / n_cap) ** 2))
    if inv_lamb >= 1:
        return None
    inv_k_plastic = 1. / (h_0 * np.log(1. / inv_lamb))
    theta_f = ms * (1. / k_rot_el + inv_k_plastic)
    if theta_f < 0:
        return None
    # theta_f = ms * (1. / k_rot_el)
    return theta_f


def calc_moment_capacity_via_millen_et_al_2020(l_in, n_load, n_cap, psi, h_eff):
    f_a = 1 / np.sqrt(1. / (1 - n_load / n_cap) ** 2 + (l_in / (2 * psi * h_eff)) ** 2)
    return n_load * l_in / 2 * f_a


def calc_fd_rot_via_millen_et_al_2020(k_rot_el, l_in, n_load, n_cap, psi, m_f, h_eff, f_p=0.5, mval=None):
    m_cap = calc_moment_capacity_via_millen_et_al_2020(l_in, n_load, n_cap, psi, h_eff)
    rot = np.where(m_f > m_cap, mval, m_f * (np.log(m_cap / m_f) + f_p) / (k_rot_el * np.log(m_cap / m_f)))
    rot = np.where(m_f == 0, 0, rot)
    if not hasattr(m_f, '__len__'):
        return rot.item()
    return rot


def calc_fd_rot_via_millen_et_al_2020_w_tie_beams(k_rot_el, l_in, n_load, n_cap, psi, ms, h_eff, k_tbs=0.0):
    m_cap = calc_moment_capacity_via_millen_et_al_2020(l_in, n_load, n_cap, psi, h_eff)
    m_tb_extreme = k_tbs * 0.03  # 3% rotation
    if ms > m_cap + m_tb_extreme:
        return None
    m_f = max([ms - m_tb_extreme, ms * 0.5])
    theta = 1000
    if k_tbs == 0:
        return calc_fd_rot_via_millen_et_al_2020(k_rot_el, l_in, n_load, n_cap, psi, m_f, h_eff)
    m_f_max = m_cap
    m_f_min = 0
    prev_m_f = m_f

    for i in range(100):
        prev_theta = theta
        theta = calc_fd_rot_via_millen_et_al_2020(k_rot_el, l_in, n_load, n_cap, psi, m_f, h_eff)
        if theta is None:
            theta = 0.031
        m_tb = k_tbs * theta
        m_r = m_f + m_tb
        if m_r < ms:
            m_f_min = m_f
            m_f = (m_f + m_f_max) / 2
        else:
            m_f_max = m_f
            m_f = (m_f + m_f_min) / 2
        if abs(theta - prev_theta) / theta < 0.01:
            return theta

        if i == 99:
            return None


def calc_recentring_ratio_via_deng_et_al_2014(a_ratio):
    """
    cite: Deng:2014bg

    Parameters
    ----------
    a_ratio

    Returns
    -------

    """
    return 1.0 / (2.6 * a_ratio + 1)


def calc_res_rot_via_deng_et_al_2014(a_ratio, peak_rot):
    rd = calc_recentring_ratio_via_deng_et_al_2014(a_ratio)
    z = (1 - rd) * peak_rot
    return z


def calc_evd_ratio_via_deng_et_al_2014(a_ratio, peak_rot, mok_ratio, hbrat=0.29):
    """mok_ratio: Moment over initial stiffness ratio (same as h in Deng paper)"""
    rd = calc_recentring_ratio_via_deng_et_al_2014(a_ratio)
    evd_n1 = 1. / (2 * np.pi) * (3 - 3 * rd)
    yield2_rot = mok_ratio / hbrat
    evd = np.where(peak_rot >= yield2_rot, 1. / (2 * np.pi) * (4 - 3 * rd - yield2_rot / peak_rot), evd_n1 * (peak_rot - mok_ratio / 2) / (yield2_rot - mok_ratio / 2))
    evd = np.clip(evd, 0, None)
    if hasattr(a_ratio, '__len__') or hasattr(peak_rot, '__len__') or hasattr(mok_ratio, '__len__'):
        return evd
    return np.asscalar(evd)


def view_calc_evd_ratio_via_deng_et_al_2014():
    import matplotlib.pyplot as plt
    mok_ratio = 0.01
    one_o_ars = [4, 8, 15, 20, 30]
    for i in range(len(one_o_ars)):
        a_ratio = 1. / one_o_ars[i]
        hbrat = 0.29
        yield2_rot = mok_ratio / hbrat
        prots = np.linspace(0, 8 * yield2_rot, 100)
        evds = calc_evd_ratio_via_deng_et_al_2014(a_ratio, prots, mok_ratio, hbrat)
        plt.plot(prots / yield2_rot, evds)
    plt.show()




def calc_mom_from_rot_salimath_2018(theta_f, fd, k_rot_el, m_cap):
    l_ip = getattr(fd, fd.ip_axis)
    l_oop = getattr(fd, fd.oop_axis)

    if l_ip < l_oop:
        chi = l_oop / l_ip - 0.02  # expression for moment about long axis
    else:
        chi = 2 * l_ip / l_oop  # expression for moment about short axis
    beta = 1 / chi * (2 * fd.n_load / fd.n_cap + 0.6)
    mom = theta_f / (beta / k_rot_el + theta_f / m_cap)
    return mom

def calc_rot_stiffness_ratio_gazetas_et_al_2013(theta_f, fd, k_rot_el):
    b = getattr(fd, fd.ip_axis)
    l = getattr(fd, fd.oop_axis)
    n = fd.n_load
    fos = fd.n_cap / fd.n_load

    chi = 1 - 0.8 / fos
    norm_angle = n * b / (4 * k_rot_el * chi) * (1 - 0.22 * (1 - 1. / fos) ** 2 * (b / l) ** 0.2)

def calc_mom_from_rot_adamidis_et_al_2013(theta_f, fd, k_rot_el, fos):
    l_ip = getattr(fd, fd.ip_axis)
    l_oop = getattr(fd, fd.oop_axis)
    chi = 1 - 0.8 / fos
    theta_s = fd.n_load * l_ip / (k_rot_el * chi) * (1 - 0.22 * (1 - 1 / fos) ** 2 * (lip / l_oop) ** 0.2)

if __name__ == '__main__':
    view_calc_evd_ratio_via_deng_et_al_2014()