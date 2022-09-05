import numpy as np

from sfsimodels import loader as ml
from sfsimodels import output as mo

from eqdes import models as em
import sfsimodels as sm
from eqdes import dbd_tools as dt
from eqdes import nonlinear_foundation as nf
from eqdes import moment_equilibrium
import geofound as gf
from eqdes import fns

from eqdes.extensions.exceptions import DesignError

from eqdes.nonlinear_foundation import calc_moment_capacity_via_millen_et_al_2020, calc_fd_rot_via_millen_et_al_2020, \
    calc_fd_rot_via_millen_et_al_2020_w_tie_beams


def assess_rc_frame(fb, hz, theta_max, otm_max, **kwargs):
    """
    Displacement-based assessment of a frame building

    :param fb: FrameBuilding Object
    :param hz: Hazard Object
    :param theta_max: [degrees], maximum structural interstorey drift
    :param otm_max: [N], maximum overturning moment
    :param kwargs:
    :return:
    """

    af = em.AssessedRCFrame(fb, hz)
    af.otm_max = otm_max
    af.theta_max = theta_max
    verbose = kwargs.get('verbose', af.verbose)

    ductility_reduction_factors = 100
    theta_c = theta_max
    for i in range(ductility_reduction_factors):
        mu_reduction_factor = 1.0 - float(i) / ductility_reduction_factors
        theta_c = theta_max * mu_reduction_factor
        displacements = dt.displacement_profile_frame(theta_c, af.heights, af.hm_factor)
        af.delta_max, af.mass_eff, af.height_eff = dt.equivalent_sdof(af.storey_mass_p_frame, displacements, af.heights)
        af.theta_y = dt.conc_frame_yield_drift(af.fye, af.concrete.e_mod_steel, af.av_bay, af.av_beam)
        af.delta_y = dt.yield_displacement(af.theta_y, af.height_eff)
        af.mu = dt.ductility(af.delta_max, af.delta_y)
        if i == 0:
            af.max_mu = af.mu
        af.xi = dt.equivalent_viscous_damping(af.mu)
        af.eta = dt.reduction_factor(af.xi)
        otm = otm_max * dt.bilinear_load_factor(af.mu, af.max_mu, af.post_yield_stiffness_ratio)
        af.v_base = otm / af.height_eff
        af.k_eff = af.v_base / af.delta_max
        af.t_eff = dt.effective_period_from_stiffness(af.mass_eff, af.k_eff)

        af.delta_demand = dt.displacement_from_effective_period(af.eta, af.hz.corner_disp,
                                                                af.t_eff, af.hz.corner_period)

        if verbose > 1:
            print('Delta_D: ', af.delta_max)
            print('Effective mass: ', af.mass_eff)
            print('Effective height: ', af.height_eff)
            print('Mu: ', af.mu)
            print('theta yield', af.theta_y)
            print('xi: ', af.xi)
            print('Reduction Factor: ', af.eta)
            print('t_eff', af.t_eff)
        if af.delta_demand > af.delta_max:  # failure occurs
            af.mu = af.delta_demand / af.delta_y
            # af.delta_demand
            break
        else:
            if verbose > 1:
                print("drift %.2f is not compatible" % theta_c)
    af.assessed_drift = theta_c
    af.storey_forces = dt.calculate_storey_forces(af.storey_mass_p_frame, displacements, af.v_base, btype='frame')
    return af


def assess_rc_frame_w_sfsi_via_millen_et_al_2020(dfb, hz, sl, fd, theta_max, otm_max=None, mcbs=None, **kwargs):
    """
    Displacement-based assessment of a frame building considering SFSI


    :param dfb: DesignedRCFrameBuilding Object
    :param hz: Hazard Object
    :param theta_max: [degrees], maximum structural interstorey drift
    :param otm_max: [N],Maximum overturning moment
    :param mcbs: [Nm], Column base moments (required if foundation is PadFoundation)
    :param found_rot: [rad], initial guess of foundation rotation
    :param kwargs:
    :return:
    """
    horz2vert_mass = kwargs.get('horz2vert_mass', 1.0)
    af = em.AssessedSFSIRCFrame(dfb, hz, sl, fd)

    af.theta_max = theta_max

    verbose = kwargs.get('verbose', af.verbose)

    af.static_values()

    # add foundation to heights
    heights = list(af.heights)
    heights.insert(0, 0)
    heights = np.array(heights) + af.fd.height
    # add foundation to masses
    storey_masses = list(af.storey_masses)
    storey_masses.insert(0, af.fd.mass)
    storey_masses = np.array(storey_masses)
    af.storey_mass_p_frame = storey_masses / af.n_seismic_frames

    ductility_reduction_factors = 100
    iterations_ductility = kwargs.get('iterations_ductility', ductility_reduction_factors)
    iterations_rotation = kwargs.get('iterations_rotation', 20)
    theta_c = theta_max
    # if m_col_base is greater than m_foot then
    if otm_max is None:
        otm_max = moment_equilibrium.calc_otm_capacity(af)
    af.theta_y = dt.conc_frame_yield_drift(af.fye, af.concrete.e_mod_steel, af.av_bay, af.av_beam)

    for i in range(iterations_ductility):
        mu_reduction_factor = 1.0 - float(i) / ductility_reduction_factors
        theta_c = theta_max * mu_reduction_factor
        displacements = dt.displacement_profile_frame(theta_c, heights, af.hm_factor, foundation=True,
                                                    fd_height=af.fd.height, theta_f=af.theta_f)
        af.delta_max, af.mass_eff, af.height_eff = dt.equivalent_sdof(af.storey_mass_p_frame, displacements, heights)

        af.delta_y = dt.yield_displacement(af.theta_y, af.height_eff - af.fd.height)
        approx_delta_f = af.theta_f * af.height_eff
        af.delta_ss = af.delta_max - approx_delta_f
        af.mu = dt.ductility(af.delta_ss, af.delta_y)
        if i == 0:
            af.max_mu = af.mu
        af.xi = dt.equivalent_viscous_damping(af.mu)
        eta_ss = dt.reduction_factor(af.xi)

        otm = otm_max * dt.bilinear_load_factor(af.mu, af.max_mu, af.post_yield_stiffness_ratio)

        # Foundation behaviour
        eta_fshear = nf.foundation_shear_reduction_factor()
        af.delta_fshear = af.v_base / (0.5 * af.k_f0_shear)
        moment_f = otm
        found_rot_tol = 0.00001
        bhr = (af.fd.width / af.height_eff)
        n_ult = af.fd_bearing_capacity
        psi = 0.75 * np.tan(sl.phi_r)
        found_rot = nf.calc_fd_rot_via_millen_et_al_2020(af.k_f_0, af.fd.length, af.total_weight, n_ult,
                                                      psi, moment_f, af.height_eff)
        norm_rot = found_rot / af.theta_pseudo_up
        cor_norm_rot = nf.calculate_corrected_normalised_rotation(norm_rot, bhr)

        eta_frot = nf.foundation_rotation_reduction_factor(cor_norm_rot)
        af.delta_frot = af.theta_f * af.height_eff
        af.delta_f = af.delta_frot + af.delta_fshear
        af.delta_max = af.delta_ss + af.delta_f
        eta_sys = nf.system_reduction_factor(af.delta_ss, af.delta_frot, af.delta_fshear, eta_ss, eta_frot, eta_fshear)

        af.eta = eta_sys
        af.theta_f = found_rot
        af.v_base = otm / (af.height_eff - af.fd.height)  # Assume hinge at top of foundation.
        af.k_eff = af.v_base / af.delta_max
        af.t_eff = dt.effective_period_from_stiffness(af.mass_eff, af.k_eff)
        if verbose > 1:
            print('Delta_max: ', af.delta_max)
            print('Effective mass: ', af.mass_eff)
            print('Effective height: ', af.height_eff)
            print('Mu: ', af.mu)
            print('theta yield', af.theta_y)
            print('xi: ', af.xi)
            print('Reduction Factor: ', af.eta)
            print('t_eff', af.t_eff)

        af.delta_demand = dt.displacement_from_effective_period(af.eta, af.hz.corner_disp,
                                                                af.t_eff, af.hz.corner_period)

        if af.delta_demand > af.delta_max:  # failure occurs
            af.mu = (af.delta_demand - af.delta_f) / af.delta_y
            # af.delta_demand
            break
        else:
            if verbose > 1:
                print("drift %.2f is not compatible" % theta_c)
    if fd.type == 'pad_foundation':
        # assert isinstance(fd, em.PadFoundation)
        ip_axis = 'length'

        af.storey_forces = dt.calculate_storey_forces(af.storey_mass_p_frame, displacements, af.v_base, btype='frame')
        # moment_beams_cl, moment_column_bases, axial_seismic = moment_equilibrium.assess(af, af.storey_forces, mom_ratio)
        mom_ratio = 0.6  # TODO: need to validate !
        if mcbs is None:
            mcbs = af.get_column_base_moments()
        # TODO: need to account for minimum column base moment which shifts mom_ratio
        h1 = af.interstorey_heights[0]
        h_eff = h1 * mom_ratio + fd.height
        pad = af.fd.pad
        pad.n_ult = af.soil_q * pad.area
        col_loads = af.get_column_vert_loads()
        col_loads = np.mean(col_loads, axis=1)  # average since multple frames
        ext_nloads = col_loads[0]
        int_nloads = np.max(col_loads[1:-1])

        m_foot_int = np.max(mcbs[1:-1]) * h_eff / (h1 * mom_ratio)
        pad.n_load = int_nloads
        tie_beams = None
        if hasattr(fd, f'tie_beam_in_{ip_axis}_dir'):
            tie_beams = getattr(fd, f'tie_beam_in_{ip_axis}_dir')
        if tie_beams is not None:
            tb_sect = getattr(fd, f'tie_beam_in_{ip_axis}_dir').s[0]
            tb_length = (fd.length - (fd.pad_length * fd.n_pads_l)) / (fd.n_pads_l - 1)
            assert isinstance(tb_sect, sm.sections.RCBeamSection)
            # See supporting_docs/tie-beam-stiffness-calcs.pdf
            e_mod_conc = sm.materials.calc_e_mod_conc_via_mander_1988(tb_sect.mat.fc)
            k_ties = (6 * tb_sect.i_rot_ww_cracked * e_mod_conc) / tb_length
        else:
            k_ties = 0
        l_in = getattr(pad, ip_axis)
        k_f_0_pad = gf.stiffness.calc_rotational_via_gazetas_1991(sl, pad, ip_axis=ip_axis)
        rot_ipad = calc_fd_rot_via_millen_et_al_2020_w_tie_beams(k_f_0_pad, l_in, int_nloads, pad.n_ult, psi,
                                                                 m_foot_int, h_eff, 2 * k_ties)
        # TODO: change to cycle through all
        # Exterior footings
        if rot_ipad is None:  # First try moment ratio of 0.5
            # m_cap = pad.n_load * getattr(pad, ip_axis) / 2 * (1 - pad.n_load / pad.n_ult)
            m_cap = calc_moment_capacity_via_millen_et_al_2020(l_in, pad.n_load, pad.n_ult, psi, h_eff)
            raise DesignError(f"Assessment failed - interior footing moment demand ({m_foot_int/1e3:.3g})"
                              f" kNm exceeds capacity (~{m_cap/1e3:.3g} kNm)")
        m_foot_ext = np.max(mcbs[np.array([0, -1])]) * h_eff / (h1 * mom_ratio)
        pad.n_load = ext_nloads
        # rot_epad = check_local_footing_rotations(sl, pad, m_foot_ext, h_eff, ip_axis=ip_axis, k_ties=k_ties)
        rot_epad = calc_fd_rot_via_millen_et_al_2020_w_tie_beams(k_f_0_pad, l_in, ext_nloads, pad.n_ult, psi,
                                                                 m_foot_ext, h_eff, k_ties)
        if rot_epad is None:
            m_cap = pad.n_load * getattr(pad, ip_axis) / 2 * (1 - pad.n_load / pad.n_ult)
            raise DesignError(f"Assessment failed - interior footing moment demand ({m_foot_ext/1e3:.3g})"
                              f" kNm exceeds capacity (~{m_cap/1e3:.3g} kNm)")
        if max([rot_ipad, rot_epad]) - found_rot > theta_c - af.theta_y:
            # footing should be increased or design drift increased
            pad_rot = max([rot_ipad, rot_epad])
            plastic_rot = theta_c - af.theta_y
            raise DesignError(f"Assessment failed - footing rotation ({pad_rot:.3g}) "
                              f"exceeds plastic rotation (~{plastic_rot:.3g})")
        af.m_foot = np.zeros(af.n_bays + 1)
        af.m_foot[0] = m_foot_ext
        af.m_foot[-1] = m_foot_ext
        af.m_foot[1:-1] = m_foot_int

    af.theta_f = found_rot
    af.assessed_drift = theta_c
    af.storey_forces = dt.calculate_storey_forces(af.storey_mass_p_frame, displacements, af.v_base, btype='frame')
    return af


def calc_base_moment_rotation(af, fd, sl, theta_col_y, mom_ratio=0.6, peak_rot=0.1, mcbs=None):
    if mcbs is None:  # provide mcbs if mu<1
        mcbs = fns.get_column_base_moments(af)
    # Assume column rotation at moment capacity is at 5x column yield rotation and has post-yield stiffness ratio of 0.01
    b = 0.01
    mu_peak = 5
    mybs = mcbs / (1 + (mu_peak - 1) * b)
    k_cols = mybs / theta_col_y
    if fd.type == 'pad_foundation':
        ip_axis = fd.ip_axis
        tie_beams = getattr(fd, f'tie_beam_in_{ip_axis}_dir')
        pad = af.fd.pad
        pad.n_ult = af.soil_q * pad.area
        psi = 0.75 * np.tan(np.radians(sl.phi))
        l_in = getattr(pad, ip_axis)
        h1 = af.interstorey_heights[0]
        # NOTE: could validate at the end that the mom_ratio is correct
        h_eff = h1 * mom_ratio + fd.height
        # k_f_0_pad = gf.stiffness.calc_rotational_via_gazetas_1991(sl, pad, ip_axis=ip_axis)
        col_loads = af.get_column_vert_loads()
        col_loads = np.mean(col_loads, axis=1)  # use average since it is for multiple frames - but want average
        if tie_beams is not None:
            tb_sect = getattr(fd, f'tie_beam_in_{ip_axis}_dir').s[0]
            tb_length = (fd.length - (fd.pad_length * fd.n_pads_l)) / (fd.n_pads_l - 1)
            assert isinstance(tb_sect, sm.sections.RCBeamSection)
            # See supporting_docs/tie-beam-stiffness-calcs.pdf
            if not hasattr(tb_sect.mat, 'e_mod_conc'):
                tb_sect.mat.e_mod_conc = sm.materials.calc_e_mod_conc_via_mander_1988(tb_sect.mat.fc)
            k_ties = (6 * tb_sect.i_rot_ww_cracked * tb_sect.mat.e_mod_conc) / tb_length
        else:
            k_ties = 0
        rot_f_vals = []
        rot_col_vals = []
        mom_combo_vals = []
        mom_f_vals = []
        for i in range(len(col_loads)):
            nload = col_loads[i] + fd.pad.mass * 9.8

            m_cap = calc_moment_capacity_via_millen_et_al_2020(l_in, nload, pad.n_ult, psi, h_eff)
            mom_f = np.linspace(0, 0.99 * m_cap, 100)
            l_in = getattr(pad, ip_axis)
            k_f_0_pad = gf.stiffness.calc_rotational_via_gazetas_1991(sl, pad, ip_axis=ip_axis)
            rot_fs = calc_fd_rot_via_millen_et_al_2020(k_f_0_pad, l_in, nload, pad.n_ult, psi, mom_f, h_eff, mval=-1)
            # TODO: build extra rot here

            if max(rot_fs) < peak_rot:
                extra_rot_points = int((peak_rot - max(rot_fs)) / (peak_rot / 20))
                extra_rots = np.linspace(max(rot_fs), peak_rot, extra_rot_points + 1)[1:]
                extra_moms = m_cap * np.ones_like(extra_rots)
                rot_fs = np.concatenate([rot_fs, extra_rots])
                mom_f = np.concatenate([mom_f, extra_moms])
            # else:
            #     inds = np.where(rot_fs < peak_rot)
            #     rot_fs = rot_fs[inds]
            #     mom_f = mom_f[inds]
            if i == 0 or i == len(col_loads) - 1:
                k_tbs = k_ties
            else:
                k_tbs = 2 * k_ties

            mom_f_plus_tbs = k_tbs * rot_fs + mom_f
            mom_ft_at_cb = mom_f_plus_tbs * (h1 * mom_ratio) / h_eff
            if mybs[i] > mom_ft_at_cb[-1]:
                rot_f_at_col_yield = np.interp(0.99 * mom_ft_at_cb[-1], mom_ft_at_cb, rot_fs)
            else:
                rot_f_at_col_yield = np.interp(mybs[i], mom_ft_at_cb, rot_fs)
            mom_f_at_col_yield = np.interp(rot_f_at_col_yield, rot_fs, mom_f)
            ind = sm.interp_left(rot_f_at_col_yield, rot_fs) + 1
            rot_fs = np.insert(rot_fs, ind, rot_f_at_col_yield)
            mom_ft_at_cb = np.insert(mom_ft_at_cb, ind, mybs[i])
            mom_f = np.insert(mom_f, ind, mom_f_at_col_yield)


            # TODO: insert mybs here
            rot_col = np.where(mom_ft_at_cb < mybs[i], mom_ft_at_cb / k_cols[i], theta_col_y + (mom_ft_at_cb - mybs[i]) / (k_cols[i] * b))
            rot_combo = rot_col + rot_fs
            # mom_c = np.interp(theta_col_p, rot_combo, mom_f_plus_tbs)
            # mcbs[i] = mom_c
            mom_f_vals.append(mom_f)
            mom_combo_vals.append(mom_ft_at_cb)
            rot_f_vals.append(rot_fs)
            rot_col_vals.append(rot_col)
        np.array(rot_f_vals, dtype=float)
        np.array(rot_col_vals, dtype=float)
        np.array(mom_f_vals, dtype=float)
        np.array(mom_combo_vals, dtype=float)
        return np.array(rot_f_vals, dtype=float), np.array(rot_col_vals, dtype=float), \
               np.array(mom_f_vals, dtype=float), np.array(mom_combo_vals, dtype=float)


def push_over_rc_frame_w_sfsi_via_millen_et_al_2021(dfb, sl, fd, theta_max, mcbs=None, **kwargs):
    """
    Displacement-based assessment of a frame building considering SFSI


    :param dfb: DesignedRCFrameBuilding Object
    :param theta_max: [degrees], maximum structural interstorey drift
    :param otm_max: [N],Maximum overturning moment
    :param mcbs: [Nm], Column base moments (required if foundation is PadFoundation)
    :param found_rot: [rad], initial guess of foundation rotation
    :param kwargs:
    :return:
    """

    # NOTE: The global rotation is only considered based on A*d^2, and the local rotation is dealt with separately.

    horz2vert_mass = kwargs.get('horz2vert_mass', 1.0)
    af = em.AssessedSFSIRCFrame(dfb, sm.SeismicHazard(), sl, fd)

    af.theta_max = theta_max

    verbose = kwargs.get('verbose', af.verbose)

    af.static_values()

    # add foundation to heights
    heights = list(af.heights)
    heights.insert(0, 0)
    heights = np.array(heights) + af.fd.height
    # add foundation to masses
    storey_masses = list(af.storey_masses)
    storey_masses.insert(0, af.fd.mass)
    storey_masses = np.array(storey_masses)
    af.storey_mass_p_frame = storey_masses / af.n_seismic_frames

    ductility_reduction_factors = 100
    iterations_ductility = kwargs.get('iterations_ductility', ductility_reduction_factors)
    iterations_rotation = kwargs.get('iterations_rotation', 20)

    af.theta_y = dt.conc_frame_yield_drift(af.fye, af.concrete.e_mod_steel, af.av_bay, af.av_beam)
    theta_col_y = af.theta_y * 0.8
    m_col_bases = fns.get_column_base_moments(af)
    mom_ratio = 0.6

    displacements = dt.displacement_profile_frame(theta_max, af.heights, af.hm_factor)
    delta_max_fb, mass_eff_fb, height_eff_fb = dt.equivalent_sdof(af.storey_mass_p_frame[1:], displacements, af.heights)
    delta_y_fb = dt.yield_displacement(af.theta_y, height_eff_fb)
    af.max_mu = delta_max_fb / delta_y_fb
    theta_p = 0.8 * af.theta_y
    max_drift_duct = theta_max / af.theta_y
    ducts = np.linspace(0.1, max_drift_duct, 10)
    peak_rot = af.theta_y * ducts[-1]
    # TODO: if m_col_base is greater than m_foot then

    # Make sure that the peak rotation and the col yield rotation are included in the rot_col_vals output else inpter fails later
    rot_f_vals, rot_col_vals, mom_f_vals, mom_combo_vals = calc_base_moment_rotation(af, fd, sl, theta_col_y,
                                                                                     mom_ratio=mom_ratio, peak_rot=0.1)
    otm_max = moment_equilibrium.calc_otm_capacity(af, mcbs=m_col_bases)
    otm_max_from_beams = otm_max - np.sum(m_col_bases)

    disps = []
    vbases = []
    mfs = []
    nfs = []
    for i in range(len(ducts)):
        theta_c = af.theta_y * ducts[i]
        if ducts[i] < 0.8:
            pass
            # Assume that column yield occurs at 80% of yield drift
            # theta_c = af.theta_y * ducts[i] / 0.8
            # Reduce column base moments
        displacements = dt.displacement_profile_frame(theta_c, heights, af.hm_factor, foundation=True,
                                                    fd_height=af.fd.height, theta_f=af.theta_f)
        af.delta_max, af.mass_eff, af.height_eff = dt.equivalent_sdof(af.storey_mass_p_frame, displacements, heights)

        af.delta_y = dt.yield_displacement(af.theta_y, af.height_eff - af.fd.height)
        approx_delta_f = af.theta_f * af.height_eff
        af.delta_ss = af.delta_max - approx_delta_f
        af.mu = dt.ductility(af.delta_ss, af.delta_y)
        af.xi = dt.equivalent_viscous_damping(af.mu)
        eta_ss = dt.reduction_factor(af.xi)
        mcbs_w_sfsi = np.zeros_like(m_col_bases, dtype=float)
        for i in range(len(m_col_bases)):
            mcbs_w_sfsi[i] = np.interp(theta_c, rot_f_vals[i] + rot_col_vals[i], mom_combo_vals[i])
        mu_factor = dt.bilinear_load_factor(af.mu, af.max_mu, af.post_yield_stiffness_ratio)
        otm_from_beams = otm_max_from_beams * mu_factor
        otm_at_col_base = otm_from_beams + np.sum(mcbs_w_sfsi)  # TODO: could split 1st order and 2nd up
        print('otm_at_col_base: ', otm_at_col_base, otm_from_beams, np.sum(mcbs_w_sfsi), theta_c,mom_combo_vals[2][-1])
        # Foundation behaviour
        eta_fshear = nf.foundation_shear_reduction_factor()
        af.delta_fshear = af.v_base / (0.5 * af.k_f0_shear)
        moment_f = otm_at_col_base * af.height_eff / (af.height_eff - af.fd.height)
        found_rot_tol = 0.00001
        bhr = (af.fd.width / af.height_eff)
        n_ult = af.fd_bearing_capacity
        psi = 0.75 * np.tan(sl.phi_r)
        found_rot = nf.calc_fd_rot_via_millen_et_al_2020(af.k_f_0, af.fd.length, af.total_weight, n_ult,
                                                      psi, moment_f, af.height_eff)
        norm_rot = found_rot / af.theta_pseudo_up
        cor_norm_rot = nf.calculate_corrected_normalised_rotation(norm_rot, bhr)

        eta_frot = nf.foundation_rotation_reduction_factor(cor_norm_rot)
        af.delta_frot = af.theta_f * af.height_eff
        af.delta_f = af.delta_frot + af.delta_fshear
        af.delta_max = af.delta_ss + af.delta_f
        eta_sys = nf.system_reduction_factor(af.delta_ss, af.delta_frot, af.delta_fshear, eta_ss, eta_frot, eta_fshear)

        af.eta = eta_sys
        af.theta_f = found_rot
        af.v_base = otm_at_col_base / (af.height_eff - af.fd.height)  # Assume hinge at top of foundation.
        af.k_eff = af.v_base / af.delta_max
        af.t_eff = dt.effective_period_from_stiffness(af.mass_eff, af.k_eff)
        if verbose > 1:
            print('Delta_max: ', af.delta_max)
            print('Effective mass: ', af.mass_eff)
            print('Effective height: ', af.height_eff)
            print('Mu: ', af.mu)
            print('theta yield', af.theta_y)
            print('xi: ', af.xi)
            print('Reduction Factor: ', af.eta)
            print('t_eff', af.t_eff)
        disps.append(af.delta_max)
        vbases.append(af.v_base)
        h1 = af.interstorey_heights[0]
        h_eff = h1 * mom_ratio + fd.height
        mom_f = np.zeros(af.n_cols)
        for cc in range(af.n_cols):
            mom_f[cc] = np.interp(mcbs_w_sfsi[cc], mom_combo_vals[cc], mom_f_vals[cc])
        mf = mom_f #* h_eff / h1 * mom_ratio
        mfs.append(mf)
        axial_seismic = moment_equilibrium.calc_seismic_axial_load_limit(af)
        col_loads = af.get_column_vert_loads()
        col_loads = np.mean(col_loads, axis=1)
        nfloads = col_loads + fd.pad.mass * 9.8
        nfloads += axial_seismic * mu_factor
        # nfloads[-1] += -axial_seismic[-1] * mu_factor
        nfs.append(nfloads)
    return np.array(vbases), np.array(disps), np.array(mfs), np.array(nfs)