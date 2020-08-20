import matplotlib.pyplot as plt
import numpy as np
import eqdes
from bwplot import cbox


def create():
    k_rot = 1000.0e2
    psi = 0.4
    h_eff = 3.0
    l_in = 3.0
    n_load = 300.
    n_cap = 3000.
    m_cap = n_load * l_in / 2 * (1 - n_load / n_cap)
    moms = np.linspace(0.001, 0.95 * m_cap, 40)
    # moms = [0.001]
    rots = []
    for mom in moms:

        theta = eqdes.dbd.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
        rots.append(theta)

    plt.plot(rots, moms)
    plt.show()


def create_w_k_ext():
    k_rot = 1000.0e2
    k_ext = 100.0e2
    psi = 0.4
    h_eff = 3.0
    l_in = 3.0
    n_load = 300.
    n_cap = 3000.
    m_cap = n_load * l_in / 2 * (1 - n_load / n_cap)
    moms = np.linspace(0.001, 1.05 * m_cap, 40)
    # moms = [308.5]
    rots = []
    rots_w_ext = []
    mom_ext = []
    rots_adj = []
    for i, mom in enumerate(moms):
        print(i, mom)
        theta = eqdes.dbd.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
        rots.append(theta)
        theta_w_ext = eqdes.dbd.calc_fd_rot_via_millen_et_al_2020_w_tie_beams(k_rot, l_in, n_load, n_cap, psi, mom, h_eff,
                                                            k_tbs=k_ext)
        rots_w_ext.append(theta_w_ext)
        if theta_w_ext is None:
            mom_ext.append(None)
            rots_adj.append(None)
        else:
            mom_ext.append(theta_w_ext * k_ext)
            rots_adj.append(eqdes.dbd.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom - mom_ext[-1], h_eff))

    plt.plot(rots, moms)
    plt.plot(rots_w_ext, moms, ls='--')
    plt.plot(rots_adj, moms)
    plt.plot(rots_w_ext, mom_ext)
    plt.show()


def create_mom_rot_vs_alt_form():
    k_rot = 1000.0e2
    psi = 0.4
    h_eff = 3.0
    l_in = 3.0

    n_cap = 3000.
    nrs = [1.5, 2, 3, 5., 10]
    for nr in nrs:
        n_load = n_cap / nr
        m_cap = n_load * l_in / 2 * (1 - n_load / n_cap)
        moms = np.linspace(0.001, 0.99 * m_cap, 40)
        rots = []
        rots_alt = []

        for mom in moms:
            rots.append(eqdes.dbd.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom, h_eff))
            rots_alt.append(eqdes.dbd.calc_fd_rot_via_millen_et_al_2020_alt_form(k_rot, l_in, n_load, n_cap, psi, mom, h_eff))

        plt.plot(rots, moms, c=cbox(0))
        plt.plot(rots_alt, moms, ls='--', c=cbox(1))
    plt.show()


if __name__ == '__main__':
    create_w_k_ext()
    # create_mom_rot_vs_alt_form()
    # create_mom_rot_vs_alt_form()
