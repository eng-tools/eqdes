import numpy as np

import eqdes.models.wall_building
from eqdes import models as em, dbd_tools as dt
from eqdes.extensions.exceptions import DesignError
from eqdes.nonlinear_foundation import calc_fd_rot_via_millen_et_al_2020, calc_moment_capacity_via_millen_et_al_2020


def design_rc_wall(sw, hz, design_drift=0.025, **kwargs):
    """
    Displacement-based design of a reinforced concrete wall.

    :param sw: SingleWall object
    :param hz: Hazard Object
    :param design_drift: Design drift
    :param kwargs:
    :return: DesignedWall object
    """

    dw = eqdes.models.wall_building.DispBasedRCWall(sw, hz)  # TODO: move outside
    dw.design_drift = design_drift
    verbose = kwargs.get('verbose', dw.verbose)
    dw.gen_static_values()
    # k = min(0.2 * (fu / fye - 1), 0.08)  # Eq 4.31b
    k = min(0.15 * (dw.fu / dw.fye - 1), 0.06)  # Eq 6.5a from DDBD code
    l_c = dw.max_height
    long_db = dw.preferred_bar_diameter
    l_sp = 0.022 * dw.fye * long_db / 1.0e6  # Eq 4.30
    l_p = max(k * l_c + l_sp + 0.1 * dw.wall_depth, 2 * l_sp)
    phi_y = dt.yield_curvature(dw.epsilon_y, dw.wall_depth, btype="wall")
    dw.phi_y = phi_y
    delta_y = dt.yield_displacement_wall(phi_y, dw.heights, dw.max_height)
    phi_p = dw.phi_material - phi_y
    # determine whether code limit or material strain governs
    theta_ss_code = design_drift
    dw.theta_y = dw.epsilon_y * dw.max_height / dw.wall_depth
    theta_p_code = (theta_ss_code - dw.theta_y)
    theta_p = min(theta_p_code, phi_p * l_p)

    # Assume no torsional effects
    increments = theta_p + dw.theta_y
    for i in range(20):
        reduced_theta_p = theta_p - increments * float(i) / 20
        if reduced_theta_p > 0.0:
            non_linear = 1

            delta_p = reduced_theta_p * (dw.heights - (0.5 * l_p - l_sp))
            if verbose > 2:
                print('reduced_theta_p: ', reduced_theta_p)
                print('delta_p: ', delta_p)

            delta_st = delta_y + delta_p
        else:
            raise DesignError('can not handle linear design, resize footing')

        dw.design_drift = reduced_theta_p + dw.theta_y

        delta_ls = delta_st

        displacements = delta_ls * dw.hm_factor

        dw.delta_d, dw.mass_eff, dw.height_eff = dt.equivalent_sdof(dw.storey_mass, displacements, dw.heights)
        delta_y = dt.yield_displacement_wall(phi_y, dw.height_eff, dw.max_height)
        dw.mu = dt.ductility(dw.delta_d, delta_y)
        dw.xi = dt.equivalent_viscous_damping(dw.mu, mtype="concrete", btype="wall")
        dw.eta = dt.reduction_factor(dw.xi)
        dw.t_eff = dt.effective_period(dw.delta_d, dw.eta, hz.corner_disp, hz.corner_period)

        if verbose > 1:
            print('Delta_D: ', dw.delta_d)
            print('Effective mass: ', dw.mass_eff)
            print('Effective height: ', dw.height_eff)
            print('Mu: ', dw.mu)
            print('theta yield', dw.theta_y)
            print('xi: ', dw.xi)
            print('Reduction Factor: ', dw.eta)
            print('t_eff', dw.t_eff)

        if dw.t_eff > 0:
            break
        else:
            if verbose > 1:
                print("drift %.2f is not compatible" % reduced_theta_p)
    k_eff = dt.effective_stiffness(dw.mass_eff, dw.t_eff)
    dw.v_base = dt.design_base_shear(k_eff, dw.delta_d)
    dw.storey_forces = dt.calculate_storey_forces(dw.storey_mass, displacements, dw.v_base, btype='wall')
    return dw


# def org_design_rc_wall_w_sfsi_via_millen_et_al_2020(wb, hz, sl, fd, design_drift=0.025, mval=None, **kwargs):
#     """
#     Displacement-based design of a concrete wall.
#
#     :param wb: WallBuilding object
#     :param hz: Hazard Object
#     :param design_drift: Design drift
#     :param kwargs:
#     :return: DesignedWall object
#     """
#     dw = eqdes.models.wall_building.DesignedSFSIRCWall(wb, hz, sl, fd)
#     dw.design_drift = design_drift
#     dw.static_dbd_values()
#     dw.static_values()
#     design_rc_wall_w_sfsi_via_millen_et_al_2020(dw, design_drift=design_drift, **kwargs)

def design_rc_wall_w_sfsi_via_millen_et_al_2020(dw, design_drift=0.025, mval=None, **kwargs):
    """
    Displacement-based design of a concrete wall.

    :param wb: DBDWallBuilding object
    :param hz: Hazard Object
    :param design_drift: Design drift
    :param kwargs:
    :return: DesignedWall object
    """
    attrs = ['hz', 'fd', 'sl']
    for attr in attrs:
        assert hasattr(dw, attr)
    dw.design_drift = design_drift
    verbose = kwargs.get('verbose', dw.verbose)

    # add foundation to heights and masses
    heights, storey_masses = dt.add_foundation(dw.heights, dw.storey_masses, dw.fd.height, dw.fd.mass)
    storey_mass_p_wall = storey_masses  # single wall model used

    # k = min(0.2 * (fu / fye - 1), 0.08)  # Eq 4.31b
    k = min(0.15 * (dw.fu / dw.fye - 1), 0.06)  # Eq 6.5a from DDBD code
    l_c = dw.max_height
    long_db = dw.preferred_bar_diameter
    l_sp = 0.022 * dw.fye * long_db / 1.0e6  # Eq 4.30
    l_p = max(k * l_c + l_sp + 0.1 * dw.wall_depth, 2 * l_sp)
    phi_y = dt.yield_curvature(dw.epsilon_y, dw.wall_depth, btype="wall")
    dw.phi_y = phi_y
    delta_y = dt.yield_displacement_wall(phi_y, heights, dw.max_height)
    phi_p = dw.phi_material - phi_y
    # determine whether code limit or material strain governs
    theta_ss_code = design_drift
    dw.theta_y = dw.epsilon_y * dw.max_height / dw.wall_depth
    theta_p_code = (theta_ss_code - dw.theta_y)
    theta_p = min(theta_p_code, phi_p * l_p)

    # Assume no torsional effects
    increments = theta_p + dw.theta_y
    for i in range(20):
        reduced_theta_p = theta_p - increments * float(i) / 20
        if reduced_theta_p > 0.0:
            non_linear = 1

            delta_p = reduced_theta_p * (dw.heights - (0.5 * l_p - l_sp))
            delta_p = np.insert(delta_p, 0, 0)  # no plastic deformation at foundation
            if verbose > 2:
                print('reduced_theta_p: ', reduced_theta_p)
                print('delta_p: ', delta_p)

            delta_st = delta_y + delta_p
        else:
            raise DesignError('can not handle linear design, resize footing')

        dw.design_drift = reduced_theta_p + dw.theta_y

        delta_ls = delta_st

        displacements = delta_ls * dw.hm_factor

        dw.delta_d, dw.mass_eff, dw.height_eff = dt.equivalent_sdof(storey_mass_p_wall, displacements, heights)
        delta_y = dt.yield_displacement_wall(phi_y, dw.height_eff, dw.max_height)
        dw.mu = dt.ductility(dw.delta_d, delta_y)
        dw.xi = dt.equivalent_viscous_damping(dw.mu, mtype="concrete", btype="wall")
        dw.eta = dt.reduction_factor(dw.xi)
        dw.t_eff = dt.effective_period(dw.delta_d, dw.eta, dw.hz.corner_disp, dw.hz.corner_period)

        if verbose > 1:
            print('Delta_D: ', dw.delta_d)
            print('Effective mass: ', dw.mass_eff)
            print('Effective height: ', dw.height_eff)
            print('Mu: ', dw.mu)
            print('theta yield', dw.theta_y)
            print('xi: ', dw.xi)
            print('Reduction Factor: ', dw.eta)
            print('t_eff', dw.t_eff)

        if dw.t_eff > 0:
            break
        else:
            if verbose > 1:
                print("drift %.2f is not compatible" % reduced_theta_p)
    k_eff = dt.effective_stiffness(dw.mass_eff, dw.t_eff)
    dw.v_base = dt.design_base_shear(k_eff, dw.delta_d)
    moment_f = dw.v_base * dw.height_eff
    dw.m_base = moment_f - dw.v_base * dw.fd.height
    psi = 0.75 * np.tan(dw.sl.phi_r)
    dw.m_f_cap = calc_moment_capacity_via_millen_et_al_2020(dw.fd.length, dw.total_weight, dw.fd.n_ult, psi,
                                                       dw.height_eff)
    theta_f = calc_fd_rot_via_millen_et_al_2020(dw.fd.k_m_0, dw.fd.length, dw.total_weight, dw.fd.n_ult, psi, moment_f,
                                                dw.height_eff, mval=mval)
    if theta_f is None:

        raise DesignError(f'Foundation moment ({moment_f}) exceeds foundation moment capacity ({dw.m_f_cap})')
    print("theta_f: ", theta_f)
    dw.theta_f = theta_f

    print("moment_f: ", moment_f, 'm_cap,f: ', dw.m_f_cap)

    # TODO: ADD calculation of rotation

    dw.storey_forces = dt.calculate_storey_forces(storey_mass_p_wall, displacements, dw.v_base, btype='wall')
    return dw