
import numpy as np
import sfsimodels as sm
from sfsimodels import output as mo

import eqdes.models.frame_building
import eqdes.models.hazard
from eqdes import models as em
from eqdes import dbd_tools as dt
from eqdes import nonlinear_foundation as nf
from eqdes.extensions.exceptions import DesignError
from eqdes import moment_equilibrium

from eqdes.nonlinear_foundation import calc_moment_capacity_via_millen_et_al_2020, calc_fd_rot_via_millen_et_al_2020, \
    calc_fd_rot_via_millen_et_al_2020_w_tie_beams


def design_rc_frame(fb, hz, design_drift=0.02, **kwargs):
    """
    Displacement-based design of a reinforced concrete frame building.

    :param fb: sfsimodels.FrameBuilding
    :param hz:
    :param design_drift:
    :param kwargs:
    :return:
    """

    df = eqdes.models.frame_building.DesignedRCFrame(fb, hz)
    df.design_drift = design_drift
    verbose = kwargs.get('verbose', df.verbose)

    for i in range(100):
        mu_reduction_factor = 1.0 - float(i) / 100
        theta_c = df.design_drift * mu_reduction_factor
        displacements = dt.displacement_profile_frame(theta_c, df.heights, df.hm_factor)
        df.delta_d, df.mass_eff, df.height_eff = dt.equivalent_sdof(df.storey_mass_p_frame, displacements, df.heights)
        df.theta_y = dt.conc_frame_yield_drift(df.fye, df.concrete.e_mod_steel, df.av_bay, df.av_beam)
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
    df.storey_forces = dt.calculate_storey_forces(df.storey_mass_p_frame, displacements, df.v_base, btype='frame')
    return df


def design_rc_frame_w_sfsi_via_millen_et_al_2020(fb, hz, sl, fd, design_drift=0.02, found_rot=0.00001,
                                         found_rot_tol=0.02, found_rot_iterations=20, **kwargs):
    import geofound as gf
    df = eqdes.models.frame_building.DesignedSFSIRCFrame(fb, hz, sl, fd)
    df.design_drift = design_drift
    verbose = kwargs.get('verbose', df.verbose)
    df.static_values()
    psi = 0.75 * np.tan(df.sl.phi_r)

    # add foundation to heights and masses
    heights, storey_masses = dt.add_foundation(df.heights, df.storey_masses, df.fd.height, df.fd.mass)
    df.storey_mass_p_frame = storey_masses / df.n_seismic_frames

    # initial estimate of foundation shear
    # for iteration in range(found_rot_iterations):  # iterate the foundation rotation
    #     df.delta_fshear = 0
    disp_compatible = False
    fd_compatible = False
    for i in range(100):
        fd_compatible = False
        mu_reduction_factor = 1.0 - float(i) / 100
        theta_c = df.design_drift * mu_reduction_factor
        temp_found_rot = found_rot
        temp_delta_fshear = 0
        for iteration in range(found_rot_iterations):  # iterate the foundation rotation

            displacements = dt.displacement_profile_frame(theta_c, heights, df.hm_factor, foundation=True,
                                                    fd_height=df.fd.height, theta_f=temp_found_rot)
            df.delta_d, df.mass_eff, df.height_eff = dt.equivalent_sdof(df.storey_mass_p_frame, displacements, heights)
            df.theta_y = dt.conc_frame_yield_drift(df.fye, df.concrete.e_mod_steel, df.av_bay, df.av_beam)
            delta_y = dt.yield_displacement(df.theta_y, df.height_eff - df.fd.height)

            df.delta_frot = df.theta_f * df.height_eff
            df.delta_f = df.delta_frot + temp_delta_fshear
            df.delta_ss = df.delta_d - df.delta_f

            df.mu = dt.ductility(df.delta_ss, delta_y)
            if df.mu < 0:
                raise DesignError('foundation rotation to large, Mu < 0.0')
            eta_fshear = nf.foundation_shear_reduction_factor()
            xi_ss = dt.equivalent_viscous_damping(df.mu)
            eta_ss = dt.reduction_factor(xi_ss)

            norm_rot = temp_found_rot / df.theta_pseudo_up
            bhr = (df.fd.width / df.height_eff)
            cor_norm_rot = nf.calculate_corrected_normalised_rotation(norm_rot, bhr)
            eta_frot = nf.foundation_rotation_reduction_factor(cor_norm_rot)
            if verbose >= 1:
                print("mu: ", df.mu)
                print('DRF_ss: ', eta_ss)
                print('DRF_frot: ', eta_frot)

            eta_sys = nf.system_reduction_factor(df.delta_ss, df.delta_frot, temp_delta_fshear, eta_ss, eta_frot, eta_fshear)

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
                disp_compatible = True
                k_eff = dt.effective_stiffness(df.mass_eff, df.t_eff)
                v_base_dynamic = dt.design_base_shear(k_eff, df.delta_d)
                v_base_p_delta = dt.p_delta_base_shear(df.mass_eff, df.delta_d, df.height_eff, v_base_dynamic)
                df.v_base = v_base_dynamic + v_base_p_delta
                prev_delta_fshear = temp_delta_fshear
                temp_delta_fshear = df.v_base / (0.5 * df.fd.k_h_0)
                df.storey_forces = dt.calculate_storey_forces(df.storey_mass_p_frame, displacements, df.v_base, btype='frame')
                n_ult = df.fd.n_ult
                moment_f = df.v_base * df.height_eff
                hf = df.height_eff
                prev_found_rot = temp_found_rot

                temp_found_rot = calc_fd_rot_via_millen_et_al_2020(df.fd.k_m_0, df.fd.length, df.total_weight, n_ult,
                                                                   psi, moment_f, df.height_eff)
                if abs(prev_delta_fshear - temp_delta_fshear) / df.delta_d < 0.01 and abs(prev_found_rot - temp_found_rot) * hf / df.delta_d < 0.01:
                    fd_compatible = True
                    break
                else:
                    fd_compatible = False

            else:
                break

        if disp_compatible:
            break
    if not fd_compatible:
        print(i, iteration)
        raise DesignError(f'Foundation displacements not compatible in design (prev: {prev_found_rot}, last: {found_rot})')
    if not disp_compatible:
        raise DesignError('System displacements not compatible in design')

    found_rot = temp_found_rot
    df.theta_f = temp_found_rot
    df.delta_fshear = temp_delta_fshear

    df.theta_f = found_rot

    if fd.type == 'pad_foundation':
        assert isinstance(fd, sm.PadFoundation)
        ip_axis = 'length'
        mom_ratio = 0.6
        moment_beams_cl, moment_column_bases, axial_seismic = moment_equilibrium.assess(df, df.storey_forces, mom_ratio)
        # TODO: need to account for minimum column base moment which shifts mom_ratio
        h_eff = df.interstorey_heights[0] * mom_ratio + fd.height
        pad = df.fd.pad
        pad.n_ult = df.soil_q * pad.area
        col_loads = df.get_column_vert_loads()
        ext_nloads = max(col_loads[0])
        int_nloads = np.max(col_loads[1:-1])

        m_foot_int = np.max(moment_column_bases[1:-1]) * h_eff / df.interstorey_heights[0]
        pad.n_load = int_nloads
        tie_beams = getattr(fd, f'tie_beam_in_{ip_axis}_dir')
        tb_length = (fd.length - (fd.pad_length * fd.n_pads_l)) / (fd.n_pads_l - 1)
        tie_beams = None
        if hasattr(fd, f'tie_beam_in_{ip_axis}_dir'):
            tie_beams = getattr(fd, f'tie_beam_in_{ip_axis}_dir')
        if tie_beams is not None:
            tb_sect = getattr(fd, f'tie_beam_in_{ip_axis}_dir').s[0]
            e_mod_conc = sm.materials.calc_e_mod_conc_via_mander_1988(tb_sect.mat.fc)
            assert isinstance(tb_sect, sm.sections.RCBeamSection)
            # See supporting_docs/tie-beam-stiffness-calcs.pdf
            k_ties = (6 * tb_sect.i_rot_ww_cracked * e_mod_conc) / tb_length
        else:
            k_ties = 0
        l_in = getattr(pad, ip_axis)
        k_f_0_pad = gf.stiffness.calc_rotational_via_gazetas_1991(sl, pad, ip_axis=ip_axis)
        rot_ipad = calc_fd_rot_via_millen_et_al_2020_w_tie_beams(k_f_0_pad, l_in, int_nloads, pad.n_ult, psi,
                                                                 m_foot_int, h_eff, 2 * k_ties)
        # Exterior footings
        if rot_ipad is None:  # First try moment ratio of 0.5
            # m_cap = pad.n_load * getattr(pad, ip_axis) / 2 * (1 - pad.n_load / pad.n_ult)
            m_cap = calc_moment_capacity_via_millen_et_al_2020(l_in, pad.n_load, pad.n_ult, psi, h_eff)
            raise DesignError(f"Design failed - interior footing moment demand ({m_foot_int/1e3:.3g})"
                              f" kNm exceeds capacity (~{m_cap/1e3:.3g} kNm)")
        m_foot_ext = np.max(moment_column_bases[np.array([0, -1])]) * h_eff / df.interstorey_heights[0]
        pad.n_load = ext_nloads
        # rot_epad = check_local_footing_rotations(sl, pad, m_foot_ext, h_eff, ip_axis=ip_axis, k_ties=k_ties)
        rot_epad = calc_fd_rot_via_millen_et_al_2020_w_tie_beams(k_f_0_pad, l_in, ext_nloads, pad.n_ult, psi,
                                                                 m_foot_ext, h_eff, k_ties)
        if rot_epad is None:
            m_cap = pad.n_load * getattr(pad, ip_axis) / 2 * (1 - pad.n_load / pad.n_ult)
            raise DesignError(f"Design failed - interior footing moment demand ({m_foot_ext/1e3:.3g})"
                              f" kNm exceeds capacity (~{m_cap/1e3:.3g} kNm)")
        if max([rot_ipad, rot_epad]) - found_rot > theta_c - df.theta_y:
            # footing should be increased or design drift increased
            pad_rot = max([rot_ipad, rot_epad])
            plastic_rot = theta_c - df.theta_y
            raise DesignError(f"Design failed - footing rotation ({pad_rot:.3g}) "
                              f"exceeds plastic rotation (~{plastic_rot:.3g})")

    return df


def design_rc_frame_w_sfsi_via_millen_et_al_2018(fb, hz, sl, fd, design_drift=0.02, found_rot=0.00001, found_rot_tol=0.02, found_rot_iterations=20, **kwargs):

    df = eqdes.models.frame_building.DesignedSFSIRCFrame(fb, hz, sl, fd)
    df.design_drift = design_drift
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
            df.theta_y = dt.conc_frame_yield_drift(df.fye, df.concrete.e_mod_steel, df.av_bay, df.av_beam)
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
        df.storey_forces = dt.calculate_storey_forces(df.storey_mass_p_frame, displacements, df.v_base, btype='frame')

        stiffness_ratio = nf.calc_foundation_rotational_stiffness_ratio_millen_et_al_2018(cor_norm_rot)
        k_f_eff = df.fd.k_m_0 * stiffness_ratio
        temp_found_rot = found_rot
        moment_f = df.v_base * df.height_eff
        found_rot = moment_f / k_f_eff
        df.delta_fshear = df.v_base / (0.5 * df.fd.k_h_0)

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
    from tests import models_for_testing as ml
    hz = eqdes.models.hazard.Hazard()
    ml.load_hazard_test_data(hz)
    fb = ml.initialise_frame_building_test_data()
    designed_frame = design_rc_frame(fb, hz)
    # print(designed_frame.n_seismic_frames)
    para = [mo.output_to_table(hz)]
    para.append(mo.output_to_table(fb))
    para = mo.add_table_ends("".join(para))
    print("".join(para))

    for item in hz.__dict__:
        print(item, hz.__dict__[item])


def run_design_rc_frame_w_sfsi_via_millen_et_al_2018():
    from tests import models_for_testing as ml
    fb = ml.initialise_frame_building_test_data()
    hz = eqdes.models.hazard.Hazard()
    sp = sm.Soil()
    fd = sm.RaftFoundation()
    ml.load_hazard_test_data(hz)
    ml.load_soil_test_data(sp)
    ml.load_raft_foundation_test_data(fd)
    designed_frame = design_rc_frame_w_sfsi_via_millen_et_al_2018(fb, hz, sp, fd, verbose=0)
    para = mo.output_to_table(designed_frame, olist="all")
    para += mo.output_to_table(fd)
    para += mo.output_to_table(sp)
    para += mo.output_to_table(hz)

    para = mo.add_table_ends(para)
    print(para)


def run_design_rc_frame_w_sfsi_via_millen_et_al_2020():
    from tests import test_dbd
    fb, fd, sp, hz = test_dbd.load_system(n_storeys=3, n_bays=2)
    designed_frame = design_rc_frame_w_sfsi_via_millen_et_al_2020(fb, hz, sp, fd, verbose=0)
    print('delta_ss: ', designed_frame.delta_ss)
    print('delta_f: ', designed_frame.delta_f)
    print(designed_frame.axial_load_ratio)
    assert np.isclose(designed_frame.axial_load_ratio, 4.7847976462)
    assert np.isclose(designed_frame.delta_ss, 0.14170439)
    assert np.isclose(designed_frame.delta_f, 0.001570054), designed_frame.delta_f


if __name__ == '__main__':
    run_design_rc_frame_w_sfsi_via_millen_et_al_2020()
    # run_frame_dba_fixed()
    # run_frame_fixed()
