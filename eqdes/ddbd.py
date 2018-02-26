
import numpy as np

from sfsimodels import loader as ml
from sfsimodels import output as mo

from eqdes import models as sm
from eqdes import ddbd_tools as dt
from eqdes import nonlinear_foundation as nf
from eqdes.extensions.exceptions import DesignError


def dbd_frame(fb, hz, design_drift=0.02, **kwargs):

    df = sm.DesignedFrame(fb, hz)
    df.design_drift = design_drift
    verbose = kwargs.get('verbose', df.verbose)

    for i in range(100):
        mu_reduction_factor = 1.0 - float(i) / 100
        theta_c = df.design_drift * mu_reduction_factor
        displacements = dt.displacement_profile_frame(theta_c, df.heights, df.hm_factor)
        df.delta_d, df.mass_eff, df.height_eff = dt.equivalent_sdof(df.storey_mass_p_frame, displacements, df.heights)
        df.theta_y = dt.conc_frame_yield_drift(df.fye, df.concrete.youngs_steel, df.av_bay, df.av_beam)
        delta_y = dt.yield_displacement(df.theta_y, df.height_eff)
        df.mu = dt.ductility(df.delta_d, delta_y)
        df.xi = dt.equivalent_viscous_damping(df.mu, mtype="concrete", btype="frame")
        df.eta = dt.reduction_factor(df.xi)
        df.t_eff = dt.effective_period(df.delta_d, df.eta, df.hz.corner_disp, df.hz.corner_period)

        if verbose > 1:
            print('Delta_D: ', df.delta_d)
            print('Effective mass: ', df.mass_eff)
            print('Effective height: ', df.height_eff)
            print('Mu: ', df.mu)
            print('theta yield', df.theta_y)
            print('xi: ', df.xi)
            print('Reduction Factor: ', df.eta)
            print('t_eff', df.t_eff)

        if df.t_eff > 0:
            break
        else:
            if verbose > 1:
                print("drift %.2f is not compatible" % theta_c)
    k_eff = dt.effective_stiffness(df.mass_eff, df.t_eff)
    df.v_base = dt.design_base_shear(k_eff, df.delta_d)
    df.storey_forces = dt.calculate_storey_forces(df.storey_mass_p_frame, displacements, df.v_base)
    return df


def wall(wb, hz, design_drift=0.025, **kwargs):
    """
    Displacement-based design of a concrete wall.
    :param wb: WallBuilding object
    :param hz: Hazard Object
    :param design_drift: Design drift
    :param kwargs:
    :return: DesignedWall object
    """

    dw = sm.DesignedWall(wb, hz)
    dw.design_drift = design_drift
    verbose = kwargs.get('verbose', dw.verbose)
    dw.static_dbd_values()
    # k = min(0.2 * (fu / fye - 1), 0.08)  # Eq 4.31b
    k = min(0.15 * (dw.fu / dw.fye - 1), 0.06)  # Eq 6.5a from DDBD code
    l_c = dw.max_height
    long_db = dw.preferred_bar_diameter
    l_sp = 0.022 * dw.fye * long_db / 1.0e6  # Eq 4.30
    l_p = max(k * l_c + l_sp + 0.1 * dw.wall_depth, 2 * l_sp)
    phi_y = dt.yield_curvature(dw.epsilon_y, dw.wall_depth, btype="wall")
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

        dw.delta_d, dw.mass_eff, dw.height_eff = dt.equivalent_sdof(dw.storey_mass_p_wall, displacements, dw.heights)
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
    dw.storey_forces = dt.calculate_storey_forces(dw.storey_mass_p_wall, displacements, dw.v_base)
    return dw


def dbd_sfsi_frame(fb, hz, sl, fd, deisgn_drift=0.02, found_rot=0.00001, found_rot_tol=0.02, found_rot_iterations=20, **kwargs):

    df = sm.DesignedSFSIFrame(fb, hz, sl, fd)  # TODO: change sp to sl
    df.design_drift = deisgn_drift
    df.theta_f = found_rot
    verbose = kwargs.get('verbose', df.verbose)
    df.static_values()

    # add foundation to heights and masses
    heights, storey_masses = dt.add_foundation(df.heights, df.storey_masses, df.fd.height, df.fd.mass)
    df.storey_mass_p_frame = storey_masses / df.n_seismic_frames

    # initial estimate of foundation shear
    for iteration in range(found_rot_iterations):  # iterate the foundation rotation
        df.delta_fshear = 0

        for i in range(100):
            mu_reduction_factor = 1.0 - float(i) / 100
            theta_c = df.design_drift * mu_reduction_factor
            displacements = dt.displacement_profile_frame(theta_c, heights, df.hm_factor, foundation=True,
                                                    fd_height=df.fd.height, theta_f=df.theta_f)
            df.delta_d, df.mass_eff, df.height_eff = dt.equivalent_sdof(df.storey_mass_p_frame, displacements, heights)
            df.theta_y = dt.conc_frame_yield_drift(df.fye, df.concrete.youngs_steel, df.av_bay, df.av_beam)
            delta_y = dt.yield_displacement(df.theta_y, df.height_eff - df.fd.height)

            df.delta_frot = df.theta_f * df.height_eff
            df.delta_f = df.delta_frot + df.delta_fshear
            df.delta_ss = df.delta_d - df.delta_f

            df.mu = dt.ductility(df.delta_ss, delta_y)
            if df.mu < 0:
                    raise DesignError('foundation rotation to large, Mu < 0.0')
            eta_fshear = nf.foundation_shear_reduction_factor()
            xi_ss = dt.equivalent_viscous_damping(df.mu)
            eta_ss = dt.reduction_factor(xi_ss)

            norm_rot = found_rot / df.theta_pseudo_up
            bhr = (df.fd.width / df.height_eff)
            cor_norm_rot = nf.calculate_corrected_normalised_rotation(norm_rot, bhr)
            eta_frot = nf.foundation_rotation_reduction_factor(cor_norm_rot)
            if verbose >= 1:
                print("mu: ", df.mu)
                print('DRF_ss: ', eta_ss)
                print('DRF_frot: ', eta_frot)

            eta_sys = nf.system_reduction_factor(df.delta_ss, df.delta_frot, df.delta_fshear, eta_ss, eta_frot, eta_fshear)

            df.eta = eta_sys
            df.xi = dt.damping_from_reduction_factor(eta_sys)
            df.t_eff = dt.effective_period(df.delta_d, df.eta, df.hz.corner_disp, df.hz.corner_period)

            if verbose > 1:
                print('Delta_D: ', df.delta_d)
                print('Effective mass: ', df.mass_eff)
                print('Effective height: ', df.height_eff)
                print('Mu: ', df.mu)
                print('theta yield', df.theta_y)
                print('xi: ', df.xi)
                print('Reduction Factor: ', df.eta)
                print('t_eff', df.t_eff)

            if df.t_eff > 0:
                break
            else:
                if verbose > 1:
                    print("drift %.2f is not compatible" % theta_c)
        k_eff = dt.effective_stiffness(df.mass_eff, df.t_eff)
        v_base_dynamic = dt.design_base_shear(k_eff, df.delta_d)
        v_base_p_delta = dt.p_delta_base_shear(df.mass_eff, df.delta_d, df.height_eff, v_base_dynamic)
        df.v_base = v_base_dynamic + v_base_p_delta
        df.storey_forces = dt.calculate_storey_forces(df.storey_mass_p_frame, displacements, df.v_base)

        stiffness_ratio = nf.foundation_rotation_stiffness_ratio(cor_norm_rot)
        k_f_eff = df.k_f_0 * stiffness_ratio
        temp_found_rot = found_rot
        moment_f = df.v_base * df.height_eff
        found_rot = moment_f / k_f_eff
        df.delta_fshear = df.v_base / (0.5 * df.k_f0_shear)

        iteration_diff = (abs(temp_found_rot - found_rot) / found_rot)
        if iteration_diff < found_rot_tol:
            if verbose:
                print('found_rot_i-1: ', temp_found_rot)
                print('found_rot_i: ', found_rot)
                print('n_iterations: ', iteration)
            break
        elif iteration == found_rot_iterations - 1:
            print('found_rot_i-1: ', temp_found_rot)
            print('found_rot_i: ', found_rot)
            print('iteration difference: ', iteration_diff)
            df.mu = -1
            raise DesignError('Could not find convergence')


        df.theta_f = found_rot
    return df


def run_frame_fixed():
    fb = sm.FrameBuilding()
    hz = sm.Hazard()
    hz = ml.load_hazard_sample_data(hz)
    ml.load_frame_building_sample_data(fb)
    designed_frame = dbd_frame(fb, hz)
    print(designed_frame.n_seismic_frames)
    para = [mo.output_to_table(hz)]
    para.append(mo.output_to_table(fb))
    para.append(mo.add_table_ends(para))
    print("".join(para))

    for item in hz.__dict__:
        print(item, hz.__dict__[item])





def run_frame_sfsi():
    fb = sm.FrameBuilding()
    hz = sm.Hazard()
    sp = sm.Soil()
    fd = sm.RaftFoundation()
    ml.load_hazard_sample_data(hz)
    ml.load_frame_building_sample_data(fb)
    ml.load_soil_sample_data(sp)
    ml.load_foundation_sample_data(fd)
    designed_frame = dbd_sfsi_frame(fb, hz, sp, fd, verbose=0)
    para = mo.output_to_table(designed_frame, olist="all")
    para += mo.output_to_table(fd)
    para += mo.output_to_table(sp)
    para += mo.output_to_table(hz)

    para = mo.add_table_ends(para)
    print(para)


if __name__ == '__main__':
    # run_frame_sfsi()
    # run_frame_dba_fixed()
    run_frame_fixed()
