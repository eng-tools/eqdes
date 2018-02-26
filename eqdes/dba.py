import numpy as np

from sfsimodels import loader as ml
from sfsimodels import output as mo

from eqdes import models as sm
from eqdes import ddbd_tools as dt
from eqdes import nonlinear_foundation as nf


def dba_frame(fb, hz, theta_max, otm_max, **kwargs):
    """
    Displacement-based assessment of a frame building
    :param fb: FrameBuilding Object
    :param hz: Hazard Object
    :param theta_max: [degrees], maximum structural interstorey drift
    :param otm_max: [N], maximum overturning moment
    :param kwargs:
    :return:
    """

    af = sm.AssessedFrame(fb, hz)
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
        af.theta_y = dt.conc_frame_yield_drift(af.fye, af.concrete.youngs_steel, af.av_bay, af.av_beam)
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
    return af


def frame_sfsi(fb, hz, sl, fd, theta_max, otm_max, found_rot=0.00001, **kwargs):
    """
    Displacement-based assessment of a frame building considering SFSI
    :param fb: FrameBuilding Object
    :param hz: Hazard Object
    :param theta_max: [degrees], maximum structural interstorey drift
    :param otm_max: [N], maximum overturning moment
    :param found_rot: [rad], initial guess of foundation rotation
    :param kwargs:
    :return:
    """

    af = sm.AssessedSFSIFrame(fb, hz, sl, fd)
    af.otm_max = otm_max
    af.theta_max = theta_max
    af.theta_f = found_rot
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

    for i in range(iterations_ductility):
        mu_reduction_factor = 1.0 - float(i) / ductility_reduction_factors
        theta_c = theta_max * mu_reduction_factor
        displacements = dt.displacement_profile_frame(theta_c, heights, af.hm_factor, foundation=True,
                                                    fd_height=af.fd.height, theta_f=af.theta_f)
        af.delta_max, af.mass_eff, af.height_eff = dt.equivalent_sdof(af.storey_mass_p_frame, displacements, heights)
        af.theta_y = dt.conc_frame_yield_drift(af.fye, af.concrete.youngs_steel, af.av_bay, af.av_beam)
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
        for j in range(iterations_rotation):
            norm_rot = found_rot / af.theta_pseudo_up
            cor_norm_rot = nf.calculate_corrected_normalised_rotation(norm_rot, bhr)
            if verbose > 1:
                print("soil_q: ", af.soil_q)
                print("axial load ratio: ", af.axial_load_ratio)
                print("theta_f: ", af.theta_f)
                print('cor_norm_rot: ', cor_norm_rot)
            stiffness_ratio = nf.foundation_rotation_stiffness_ratio(cor_norm_rot)
            k_f_eff = af.k_f_0 * stiffness_ratio
            temp_found_rot = found_rot

            found_rot = moment_f / k_f_eff

            iteration_diff = (abs(temp_found_rot - found_rot) / found_rot)
            if iteration_diff < found_rot_tol:
                break
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
    af.assessed_drift = theta_c
    return af


def run_frame_dba_fixed():
    fb = sm.FrameBuilding()
    hz = sm.Hazard()
    hz = ml.load_hazard_sample_data(hz)
    ml.load_frame_building_sample_data(fb)

    designed_frame = dba_frame(fb, hz, theta_max=0.035, otm_max=3e5, verbose=0)
