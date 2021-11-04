import eqdes
import numpy as np


def test_calculate_rotation_via_millen_et_al_2020():
    mom = 200.
    k_rot = 1000.0e2
    psi = 0.4
    h_eff = 3.0
    l_in = 3.0
    n_load = 300.
    n_cap = 3000.
    theta = eqdes.nonlinear_foundation.calc_fd_rot_via_millen_et_al_2020_alt_form(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
    assert np.isclose(theta, 0.0053710398), theta
    theta = eqdes.nonlinear_foundation.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
    assert np.isclose(theta, 0.0053710398), theta

    n_load = 2000.
    n_cap = 3000.
    l_in = 5.0
    mom = 100.
    theta = eqdes.nonlinear_foundation.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
    assert np.isclose(theta, 0.0011910855), theta

    # very large moment
    mom = 3000.
    theta = eqdes.nonlinear_foundation.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
    assert theta is None

    # n_load equal to n_cap
    mom = 10
    n_load = 300.
    n_cap = 300.
    theta = eqdes.nonlinear_foundation.calc_fd_rot_via_millen_et_al_2020_alt_form(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
    assert theta is None


def test_calc_fd_rot_via_millen_et_al_2020_w_tie_beams():
    k_rot = 1000.0e2
    k_tbs = 100.0e2
    psi = 0.4
    h_eff = 3.0
    l_in = 3.0
    n_load = 300.
    n_cap = 3000.

    mom = 308.5

    theta_w_tbs = eqdes.nonlinear_foundation.calc_fd_rot_via_millen_et_al_2020_w_tie_beams(k_rot, l_in, n_load, n_cap, psi, mom,
                                                                                           h_eff,
                                                                                           k_tbs=k_tbs)

    mom_tbs = theta_w_tbs * k_tbs
    rots_adj = eqdes.nonlinear_foundation.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom - mom_tbs, h_eff)
    assert np.isclose(theta_w_tbs, rots_adj, rtol=0.01)


