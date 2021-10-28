import numpy as np
import sfsimodels as sm


def calc_confined_via_mander_1988(col_sect, fc_unconf):
    assert isinstance(col_sect, sm.sections.RCDetailedSection)

    eps_unconf = 0.002
    dbs = np.concatenate(col_sect.bar_diams)
    area_cc = col_sect.area_c * (1 - col_sect.area_steel / col_sect.area_c)
    # Compute clear spacing
    xc_top = np.diff(col_sect.bar_centres[0])  # distance between bar centres
    bar_widths = (np.array(col_sect.bar_diams[0][1:]) + np.array(col_sect.bar_diams[0][:-1])) / 2
    wis_top = xc_top - bar_widths  # clear distance between bars
    xc_bot = np.diff(col_sect.bar_centres[-1])  # distance between bar centres
    bar_widths = (np.array(col_sect.bar_diams[-1][1:]) + np.array(col_sect.bar_diams[-1][:-1])) / 2
    wis_bot = xc_bot - bar_widths
    xc_side = np.diff(col_sect.layer_depths)
    dbs_side = np.array([bar_ds[0] for bar_ds in col_sect.bar_diams])
    bar_widths = (dbs_side[1:] + dbs_side[:-1]) / 2
    wis_side = xc_side - bar_widths
    # Eq 20
    a_para = np.sum(wis_top ** 2 / 6) + np.sum(wis_bot ** 2 / 6) + 2 * np.sum(wis_side ** 2 / 6)
    b_c = col_sect.width_c
    d_c = col_sect.depth_c
    s_dash = col_sect.spacing_trans - col_sect.db_trans  # clearing spacing
    area_e = (b_c * d_c - a_para) * (1 - s_dash / (2 * b_c)) * (1 - s_dash / (2 * d_c))  # Eq 21
    k_e = area_e / area_cc  # Eq 10
    area_steel_trans_x = col_sect.db_trans ** 2 / 4 * np.pi * col_sect.nb_trans_x
    area_steel_trans_y = col_sect.db_trans ** 2 / 4 * np.pi * col_sect.nb_trans_y
    rho_tran_x = area_steel_trans_x / (col_sect.spacing_trans * d_c)
    rho_tran_y = area_steel_trans_y / (col_sect.spacing_trans * d_c)
    rho_tran = (rho_tran_x + rho_tran_y) / 2
    f_lat = rho_tran * col_sect.fy_trans
    f_lat_dash = f_lat * k_e  # Effective uniform lateral confining stress from rebar
    fc_conf = fc_unconf * (-1.254 + 2.254 * np.sqrt(1 + 7.94 * f_lat_dash / fc_unconf) - 2 * f_lat_dash / fc_unconf)
    eps_conf = eps_unconf * (1 + 5 * (fc_conf / fc_unconf - 1))
    # x = eps_unconf / eps_conf
    # e_sec_conf = fc_conf / eps_conf
    # e_mod = 5.0e3 * np.sqrt(fc_unconf)
    # r = e_mod / (e_mod - e_sec_conf)
    return fc_conf, eps_conf
