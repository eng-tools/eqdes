from eqdes import models as em
from eqdes import dbd_tools as dt


def assess_rc_wall(wb, hz, theta_max, otm_max, **kwargs):
    """
    Displacement-based assessment of a frame building

    :param wb: FrameBuilding Object
    :param hz: Hazard Object
    :param theta_max: [degrees], maximum structural interstorey drift
    :param otm_max: [N], maximum overturning moment
    :param kwargs:
    :return:
    """

    af = em.AssessedRCWall(wb, hz)
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


