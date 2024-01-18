
import numpy as np

import eqdes.dbd.dbd_wall
import eqdes.models.foundation
import eqdes.models.hazard
import eqdes.models.soil
import eqdes.models.wall_building
import eqdes.nonlinear_foundation
from tests import models_for_testing as ml
from eqdes import dbd
from eqdes import models as dm
from eqdes import design_spectra
import sfsimodels as sm
import geofound as gf

from tests.checking_tools import isclose


def to_be_test_ddbd_sfsi_wall_from_millen_pdf_paper_2018():
    fb = eqdes.models.wall_building.WallBuilding(n_storeys=6)
    sl = eqdes.models.soil.Soil()
    fd = eqdes.models.foundation.PadFoundation()
    hz = eqdes.models.hazard.Hazard()
    fb.id = 1
    fb.wall_depth = 3.4
    fb.wall_width = 0.3
    fb.number_walls = 4
    fb.interstorey_heights = 3.4 * np.ones(fb.n_storeys)
    fb.name = 'My' + str(fb.n_storeys) + 'storeyRBwall'  # character String
    fb.building_length = 20.0
    fb.building_width = 12.0  # m
    fb.raft_foundation = 0

    fd.height = 1.0
    fd.width = 5.5

    fb.DBaspect = 0.2
    fb.LBaspect = 2.5
    # loads:
    fb.gravity = 9.8
    fb.soil_type = 'C'
    hz.z_factor = design_spectra.calculate_z(0.4, fb.soil_type)
    hz.r_factor = 1.0
    hz.n_factor = 1.0

    fb.Live_load = 3.0e3  # Pa
    fb.floor_weight = 0.0e3  # Pa
    fb.additional_weight = 6.0e3  # Pa    #partitions, ceilings, serivces
    fb.wall_axial_contribution = 1.0
    fb.conc_weight = 23.5e3
    # Design options:

    drift_limit = 0.012
    foundation_rotation = 0.0009

    # Soil properties
    sl.g_mod = 40e6  # Pa
    sl.poissons_ratio = 0.3  # Poisson's ratio of the soil
    sl.relative_density = 0.60  # %
    sl.phi = 36
    sl.unit_moist_weight = 18e3  # N/m3
    sl.cohesion = 0

    #############################
    # Material info
    fb.fy = 300e6
    sl.E_s = 200e9
    sl.fc = 30e6  # Pa
    sl.E_conc = (3320 * np.sqrt(40.0) + 6900.0) * 1e6  # 37000000000;   #Pa    (3320*sqrt(40.0)+6900.0)*1e6
    sl.Conc_Poissons_ratio = 0.18


def test_case_study_wall_pbd_wall_fixed_base():
    n_storeys = 6
    sw = sm.SingleWall(n_storeys)
    sw.wall_width = 0.3  # m
    sw.wall_depth = 3.4  # m
    sw.interstorey_heights = 3.4 * np.ones(n_storeys)  # m
    sw.n_walls = 1
    floor_length = 20 / 2  # m
    floor_width = 12 / 2  # m
    g_load = 6000.  # Pa
    q_load = 3000.  # Pa
    eq_load_factor = 0.4
    floor_pressure = g_load + eq_load_factor * q_load
    sw.storey_masses = floor_pressure * floor_length * floor_width * np.ones(n_storeys) / 9.8

    # hazard
    hz = eqdes.models.hazard.Hazard()
    hz.z_factor = 0.4  # Hazard factor
    hz.r_factor = 1.0  # Return period factor
    hz.n_factor = 1.0  # Near-fault factor
    hz.magnitude = 7.5  # Magnitude of earthquake
    hz.corner_period = 3.0  # s
    hz.corner_acc_factor = 0.4

    sw.material = sm.materials.ReinforcedConcreteMaterial()
    dw = eqdes.models.wall_building.DispBasedRCWall(sw)
    dw.preferred_bar_diameter = 0.032
    dw = eqdes.dbd.dbd_wall.design_rc_wall(dw, hz, design_drift=0.025)


def test_case_study_wall_pbd_wall_w_sfsi():
    n_storeys = 6
    wb = sm.SingleWall(n_storeys)
    wb.wall_width = 0.3  # m
    wb.wall_depth = 3.4  # m
    wb.interstorey_heights = 3.4 * np.ones(n_storeys)  # m
    wb.n_walls = 1
    floor_length = 20 / 2  # m
    floor_width = 12 / 2  # m
    g_load = 6000.  # Pa
    q_load = 3000.  # Pa
    eq_load_factor = 0.4
    floor_pressure = g_load + eq_load_factor * q_load
    wb.storey_masses = floor_pressure * floor_length * floor_width * np.ones(n_storeys) / 9.8
    wb.storey_n_loads = 9.8 * wb.storey_masses

    fd = eqdes.models.foundation.RaftFoundation()
    fd.height = 1.3
    fd.length = 5.6  # m # from HDF
    fd.width = 2.25  # m # from HDF
    fd.depth = 0.0  # TODO: check this
    fd.mass = 0.0

    # soil properties from HDF
    sl = eqdes.models.soil.Soil()
    sl.g_mod = 40e6  # Pa
    sl.poissons_ratio = 0.3
    sl.phi = 36.0  # degrees
    # sl.phi_r = np.radians(sl.phi)
    sl.cohesion = 0.0
    sl.unit_dry_weight = 18000.  # TODO: check this

    # hazard
    hz = eqdes.models.hazard.Hazard()
    hz.z_factor = 0.4  # Hazard factor
    hz.r_factor = 1.0  # Return period factor
    hz.n_factor = 1.0  # Near-fault factor
    hz.magnitude = 7.5  # Magnitude of earthquake
    hz.corner_period = 3.0  # s
    hz.corner_acc_factor = 0.4

    n_wall_eq = np.sum(wb.storey_masses) / wb.n_walls * 9.8
    n_cap_from_hdf = 12.1e6  # N
    n_wall_eq_from_hdf = 2.31e6  # N

    # n_from_input_file = (4.0e2 + 1.905e3) * 1e3
    alpha = 4.

    wb.material = sm.materials.ReinforcedConcreteMaterial()
    # dw = dbd.wall(wb, hz, design_drift=0.025)
    dw = eqdes.models.wall_building.DispBasedRCWall(wb, sl=sl, fd=fd)
    dw.preferred_bar_diameter = 0.032
    dw.gen_static_values()
    dw = eqdes.dbd.design_rc_wall_w_sfsi_via_millen_et_al_2020(dw, hz, design_drift=0.025)
    assert np.isclose(dw.delta_d, 0.282773, rtol=0.001), dw.delta_d
    assert np.isclose(dw.v_base, 354451.990, rtol=0.001), dw.v_base
    assert np.isclose(dw.m_base, 4435939.0475, rtol=0.001), dw.m_base


def test_ddbd_wall_fixed():

    hz = eqdes.models.hazard.Hazard()
    ml.load_hazard_test_data(hz)
    sw = ml.initialise_single_wall_test_data()
    dw = eqdes.models.wall_building.DispBasedRCWall(sw)
    dw.preferred_bar_diameter = 0.032
    wall_dbd = eqdes.dbd.dbd_wall.design_rc_wall(dw, hz)

    assert isclose(wall_dbd.delta_d, 0.339295, rel_tol=0.001), wall_dbd.delta_d
    assert isclose(wall_dbd.mass_eff, 59429.632, rel_tol=0.001), wall_dbd.mass_eff
    assert isclose(wall_dbd.height_eff, 12.46885, rel_tol=0.001), wall_dbd.height_eff
    assert isclose(wall_dbd.mu, 1.1902, rel_tol=0.001), wall_dbd.mu
    assert isclose(wall_dbd.theta_y, 0.0168299, rel_tol=0.001), wall_dbd.theta_y
    assert isclose(wall_dbd.xi, 0.07259, rel_tol=0.001), wall_dbd.xi
    assert isclose(wall_dbd.eta, 0.86946, rel_tol=0.001), wall_dbd.eta
    assert isclose(wall_dbd.t_eff, 2.38184, rel_tol=0.001), wall_dbd.t_eff


if __name__ == '__main__':
    test_case_study_wall_pbd_wall_w_sfsi()
