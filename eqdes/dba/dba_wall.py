import numpy as np
from eqdes import models as em
from eqdes import dbd_tools as dt


def assess_rc_wall(aw, hz, drift_lim, otm_lim, vstorey_lim, **kwargs):
    """
    Displacement-based assessment of a frame building

    :param wb: FrameBuilding Object
    :param hz: Hazard Object
    :param theta_max: [degrees], maximum structural interstorey drift
    :param otm_max: [N], maximum overturning moment
    :param kwargs:
    :return:
    """

    verbose = kwargs.get('verbose', aw.verbose)
    
    heights = aw.heights
    # k = min(0.15 * (aw.fu / aw.fye - 1), 0.06)  # Eq 6.5a from DDBD code
    k = min(0.2 * (aw.fu / aw.fye - 1), 0.08)  # NZ Seismic assess guide
    l_c = aw.max_height
    long_db = aw.preferred_bar_diameter
    l_sp = 0.022 * aw.fye * long_db / 1.0e6  # Eq 4.30
    l_p = max(k * l_c + l_sp + 0.1 * aw.wall_depth, 2 * l_sp)
    phi_y = dt.yield_curvature(aw.epsilon_y, aw.wall_depth, btype="wall")
    aw.phi_y = phi_y
    delta_y = aw.phi_y * heights ** 2 / 2 * (1 - heights / (3 * aw.max_height))
    # delta_y = dt.yield_displacement_wall(phi_y, heights, aw.max_height)
    phi_plas = aw.phi_wall_lim - phi_y
    # determine whether external limits (e.g. frame / code) or material strain governs
    aw.drift_y = aw.epsilon_y * aw.max_height / aw.wall_depth
    drift_plas_lim = (drift_lim - aw.drift_y)
    drift_plas = min(drift_plas_lim, phi_plas * l_p)

    # Assume no torsional effects
    delta_p = drift_plas * (aw.heights - (0.5 * l_p - l_sp))
    if verbose > 2:
        print('drift_p: ', drift_plas)
        print('delta_p: ', delta_p)

    delta_st = delta_y + delta_p

    aw.design_drift = drift_plas + aw.drift_y

    delta_ls = delta_st

    displacements = delta_ls * aw.hm_factor

    aw.delta_cap, aw.mass_eff, aw.height_eff = dt.calc_equivalent_sdof(aw.storey_mass, displacements, aw.heights)
    aw.v_base = otm_lim / aw.height_eff
    aw.storey_forces = dt.calculate_storey_forces(aw.storey_mass, displacements, aw.v_base, btype='wall')
    v_storeys_nom = np.cumsum(aw.storey_forces[::-1])[::-1]
    hm_fac = dt.calc_higher_mode_factor(aw.n_storeys, btype='wall')
    v_storeys = hm_fac * aw.os_fac * v_storeys_nom
    v_storeys_max = max(v_storeys)
    if v_storeys_max > vstorey_lim:
        aw.v_base = vstorey_lim
        raise ValueError('shear governs')

    # delta_y = dt.yield_displacement_wall(phi_y, aw.height_eff, aw.max_height)
    delta_y = phi_y * aw.height_eff ** 2 / 2 * (1 - aw.height_eff / 3 / aw.max_height)
    aw.mu = dt.ductility(aw.delta_cap, delta_y)
    aw.xi = dt.equivalent_viscous_damping(aw.mu, mtype="concrete", btype="wall")
    aw.eta = dt.reduction_factor(aw.xi)

    aw.k_eff = aw.v_base / aw.delta_cap
    aw.t_eff = 2 * np.pi * (aw.mass_eff / aw.k_eff) ** 0.5
    aw.delta_demand = dt.displacement_from_effective_period(aw.eta, hz.corner_disp,
                                                            aw.t_eff, hz.corner_period)
    aw.acc_demand = aw.delta_demand * (2 *np.pi / aw.t_eff)
    aw.acc_cap = aw.v_base / aw.mass_eff
    aw.nbs = aw.delta_cap / aw.delta_demand
    if verbose > 1:
        print('Delta_cap: ', aw.delta_cap)
        print('Effective mass: ', aw.mass_eff)
        print('Effective height: ', aw.height_eff)
        print('Mu: ', aw.mu)
        print('drift yield', aw.drift_y)
        print('xi: ', aw.xi)
        print('Reduction Factor: ', aw.eta)
        print('t_eff', aw.t_eff)

    return aw


