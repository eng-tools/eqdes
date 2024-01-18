import numpy as np


from eqdes import models
import eqdes.nonlinear_foundation
from tests import models_for_testing as ml
from eqdes import dbd
from eqdes import models as dm
from eqdes import design_spectra
import sfsimodels as sm
from eqdes import dbd_tools as dt
import geofound as gf

from tests.checking_tools import isclose


def test_sesoc_wall():
    n_storeys = 3
    wb = sm.SingleWall(n_storeys)
    wb.wall_width = 0.2  # m
    wb.wall_depth = 4.3  # m
    wb.interstorey_heights = 3.6 * np.ones(n_storeys)  # m
    wb.n_walls = 1

    wb.storey_masses = 300.0e3 * np.ones(n_storeys)
    wb.storey_n_loads = 9.8 * wb.storey_masses

    # hazard
    hz = models.Hazard()
    hz.z_factor = 0.3  # Hazard factor
    hz.r_factor = 1.0  # Return period factor
    hz.n_factor = 1.0  # Near-fault factor
    hz.magnitude = 7.5  # Magnitude of earthquake
    hz.corner_period = 3.0  # s
    hz.site_class = 'D'
    hz.corner_acc_factor = 6.42 / 9.8  # site=D

    wb.material = sm.materials.ReinforcedConcreteMaterial()
    # dw = dbd.wall(wb, hz, design_drift=0.025)
    dw = eqdes.models.wall_building.DispBasedRCWall(wb, hz)
    dw.preferred_bar_diameter = 0.02
    dw.fye = 440.0e6
    dw.fu = 510.0e6
    dw.epsilon_y = min(dw.fye / dw.concrete.e_mod_steel, 0.002)
    dw.os_fac = 1.25
    c_wall = 0.735  # m # from slide 52
    k_d = 15 - 20 * c_wall / wb.wall_depth
    phi_wall_y = dt.yield_curvature(dw.epsilon_y, dw.wall_depth, btype="wall")
    dw.phi_wall_lim = k_d * phi_wall_y
    dw.gen_static_values()
    vbase_lim = 1600e3  # todo convert to coordinates
    dw = eqdes.dba.assess_rc_wall(dw, hz, otm_lim=9350e3, drift_lim=0.015, vstorey_lim=vbase_lim)
    assert np.isclose(dw.delta_cap, 0.107, rtol=0.107), dw.delta_cap
    assert np.isclose(dw.v_base, 1090.0e3, rtol=0.01), dw.v_base
    assert np.isclose(dw.nbs, 0.70, rtol=0.03), dw.nbs


if __name__ == '__main__':
    test_sesoc_wall()