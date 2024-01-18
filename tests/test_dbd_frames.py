
import numpy as np

import eqdes.models.foundation
import eqdes.models.frame_building
import eqdes.models.hazard
import eqdes.models.material
import eqdes.models.soil
import eqdes.nonlinear_foundation
from tests import models_for_testing as ml
from eqdes import dbd
from eqdes import models as dm
from eqdes import design_spectra
import sfsimodels as sm
import geofound as gf

from tests.checking_tools import isclose


def test_ddbd_frame_fixed_small():

    hz = eqdes.models.hazard.Hazard()
    ml.load_hazard_test_data(hz)
    fb = ml.initialise_frame_building_test_data()
    frame_ddbd = dbd.design_rc_frame(fb, hz)

    assert isclose(frame_ddbd.delta_d, 0.2400, rel_tol=0.001), frame_ddbd.delta_d
    assert isclose(frame_ddbd.mass_eff, 67841.581, rel_tol=0.001), frame_ddbd.mass_eff
    assert isclose(frame_ddbd.height_eff, 14.34915, rel_tol=0.001), frame_ddbd.height_eff
    assert isclose(frame_ddbd.mu, 1.689, rel_tol=0.001), frame_ddbd.mu
    assert isclose(frame_ddbd.theta_y, 0.0099, rel_tol=0.001), frame_ddbd.theta_y
    assert isclose(frame_ddbd.xi, 0.123399, rel_tol=0.001), frame_ddbd.xi
    assert isclose(frame_ddbd.eta, 0.69867, rel_tol=0.001), frame_ddbd.eta
    assert isclose(frame_ddbd.t_eff, 2.09646, rel_tol=0.001), frame_ddbd.t_eff


def test_ddbd_frame_consistent():
    """
    Test the DBD of a fixed base frame is the same as the SFSI frame when the soil is very stiff.
    :return:
    """

    fb = ml.initialise_frame_building_test_data()
    hz = eqdes.models.hazard.Hazard()
    sl = eqdes.models.soil.Soil()
    fd = eqdes.models.foundation.RaftFoundation()
    ml.load_hazard_test_data(hz)
    ml.load_soil_test_data(sl)
    ml.load_raft_foundation_test_data(fd)
    frame_ddbd = dbd.design_rc_frame(fb, hz)
    sl.override("g_mod", 1.0e10)  # make soil very stiff
    fd.height = 2.0  # add some height to the foundation
    frame_sfsi_dbd = dbd.design_rc_frame_w_sfsi_via_millen_et_al_2018(fb, hz, sl, fd, found_rot=1e-6)
    assert isclose(frame_sfsi_dbd.theta_f, 0.0, abs_tol=1e-5)

    assert isclose(frame_ddbd.delta_d, frame_sfsi_dbd.delta_d, rel_tol=0.01), frame_sfsi_dbd.delta_d
    assert isclose(frame_ddbd.mass_eff, frame_sfsi_dbd.mass_eff, rel_tol=0.001), frame_sfsi_dbd.mass_eff
    assert isclose(frame_ddbd.height_eff, frame_sfsi_dbd.height_eff - fd.height, rel_tol=0.001), frame_sfsi_dbd.height_eff
    assert isclose(frame_ddbd.mu, frame_sfsi_dbd.mu, rel_tol=0.001), frame_sfsi_dbd.mu
    assert isclose(frame_ddbd.theta_y, frame_sfsi_dbd.theta_y, rel_tol=0.001), frame_sfsi_dbd.theta_y
    assert isclose(frame_ddbd.xi, frame_sfsi_dbd.xi, rel_tol=0.001), frame_sfsi_dbd.xi
    assert isclose(frame_ddbd.eta, frame_sfsi_dbd.eta, rel_tol=0.001), frame_sfsi_dbd.eta
    assert isclose(frame_ddbd.t_eff, frame_sfsi_dbd.t_eff, rel_tol=0.001), frame_sfsi_dbd.t_eff


def test_ddbd_frame_fixed_large():
    n_storeys = 5
    n_bays = 1
    fb = eqdes.models.frame_building.FrameBuilding(n_storeys, n_bays)
    fb.material = eqdes.models.material.ReinforcedConcrete()
    hz = eqdes.models.hazard.Hazard()

    fb.interstorey_heights = 3.6 * np.ones(n_storeys)

    fb.bay_lengths = 6.0 * np.ones(n_bays)
    fb.set_beam_prop("depth", .6 * np.ones(n_bays))  # m      #varies vertically

    hz.z_factor = 0.3
    hz.r_factor = 1.0
    hz.n_factor = 1.0

    design_drift = 0.02

    fb.n_seismic_frames = 1
    fb.n_gravity_frames = 0

    fb.floor_length = sum(fb.bay_lengths)
    fb.floor_width = 14.0  # m

    fb.storey_masses = np.array([488.0, 488.0, 488.0, 488.0, 411.0]) * 1e3

    frame_ddbd = dbd.design_rc_frame(fb, hz, design_drift=design_drift)

    # StoreyForcesCheck1 = np.array([460402.85, 872342.25, 1235818.18, 1550830.66, 2158400.44])
    # assert isclose(frame_ddbd.v_base, 4889353.79)  # 6277794.38  # TODO: fix this
    # assert isclose(frame_ddbd.Storey_Forces, StoreyForcesCheck1)


def test_dbd_sfsi_frame_via_millen_et_al_2018():
    n_storeys = 5
    n_bays = 1
    fb = eqdes.models.frame_building.FrameBuilding(n_storeys, n_bays)
    fb.n_seismic_frames = 2
    fb.n_gravity_frames = 0
    fb.material = eqdes.models.material.ReinforcedConcrete()
    fb.bay_lengths = [5.]
    fb.floor_width = 5.
    fb.floor_length = 5.
    fb.storey_masses = 700 * fb.floor_area * np.ones(n_storeys)
    fb.interstorey_heights = 3.4 * np.ones(n_storeys)

    fb.tie_depth = 0.8 * np.ones(fb.n_bays)  # m
    fb.tie_width = 0.8 * np.ones(fb.n_bays)  # m
    fb.foundation_rotation = 1e-3
    fb.discrete_rotation_ratio = 1.0
    fb.AxialLoadRatio = 26  # Should calculate this
    fb.Base_moment_contribution = 0.6
    fb.beam_group_size = 1
    fb.set_beam_prop('depth', 0.4, repeat='all')

    # Foundation
    fd = eqdes.models.foundation.PadFoundation()
    fd.height = 0  # m
    fd.length = 5
    fd.width = 5.
    fd.depth = 0.8
    fd.n_pads_l = 2
    fd.n_pads_w = 2

    fd.pad.width = 1.4  # m
    fd.pad.length = 1.4  # m
    fd.pad.depth = 0.5  # m
    fd.pad.height = 0
    fd.set_pad_pos_in_length_dir_as_equally_spaced()
    fd.set_pad_pos_in_width_dir_as_equally_spaced()

    fd.mass = 0
    fd2 = fd.deepcopy()

    # Soil properties

    hz = eqdes.models.hazard.Hazard()
    hz.z_factor = 0.3
    hz.r_factor = 1.0
    hz.n_factor = 1.0
    hz.corner_acc_factor = 2.
    hz.corner_period = 1

    sl = eqdes.models.soil.Soil()
    sl.g_mod = 120e6  # Pa
    sl.poissons_ratio = 0.3  # Poisson's ratio of the soil
    sl.e_curr = 0.6  # %
    sl.phi = 35.
    sl.cohesion = 0
    sl.specific_gravity = 2.65

    design_drift = 0.02

    frame_ddbd = dbd.design_rc_frame_w_sfsi_via_millen_et_al_2018(fb, hz, sl, fd, design_drift=design_drift, verbose=0)
    assert np.isclose(frame_ddbd.delta_d, 0.08488596427), frame_ddbd.delta_d
    assert np.isclose(frame_ddbd.theta_f, 0.00501363, rtol=0.001), frame_ddbd.theta_f


def load_system(n_bays=2, n_storeys=6):
    hz = sm.SeismicHazard()
    hz.z_factor = 0.3  # Hazard factor
    hz.r_factor = 1.0  # Return period factor
    hz.n_factor = 1.0  # Near-fault factor
    hz.magnitude = 7.5  # Magnitude of earthquake
    hz.corner_period = 4.0  # s
    hz.corner_acc_factor = 0.55
    sp = sm.Soil()
    sp.g_mod = 25.0e6  # [Pa]
    sp.phi = 32.0  # [degrees]
    sp.unit_dry_weight = 17000  # [N/m3]
    sp.unit_sat_weight = 18000  # [N/m3]
    sp.unit_weight_water = 9800  # [N/m3]
    sp.cohesion = 0.0  # [Pa]
    sp.poissons_ratio = 0.22

    interstorey_height = 3.4  # m
    fb = sm.FrameBuilding(n_bays=n_bays, n_storeys=n_storeys)
    fb.material = sm.materials.ReinforcedConcreteMaterial()

    fb.interstorey_heights = interstorey_height * np.ones(n_storeys)
    fb.bay_lengths = 4.0 * np.ones(n_bays)
    fb.floor_length = np.sum(fb.bay_lengths) + 4  # m
    fb.floor_width = 12.0  # m
    fb.n_seismic_frames = 3
    fb.n_gravity_frames = 0
    fb.set_storey_masses_by_pressure(9e3)  # Pa
    col_loads = fb.get_column_vert_loads()
    fb.horz2vert_mass = 1
    fb.set_beam_prop('depth', 0.6, 'all')

    fd = sm.PadFoundation()
    fd.width = fb.floor_width  # m
    fd.length = fb.floor_length  # m
    fd.depth = 0.4 + 0.1 * fb.n_storeys  # m
    fd.height = 1.0  # m
    fd.mass = 0.0  # kg
    pad = gf.size_footing_for_capacity(sp, np.max(col_loads), method='salgado', fos=4., depth=fd.depth)
    fd.pad_length = pad.length
    fd.pad_width = pad.width
    fd.pad.depth = fd.depth
    fd.pad.height = fd.height
    tie_beams_sect = sm.sections.RCBeamSection()
    tie_beams_sect.depth = fd.height
    tie_beams_sect.width = fd.height
    tie_beams_sect.rc_mat = sm.materials.ReinforcedConcreteMaterial()
    tie_beams_sect.cracked_ratio = 0.6
    fd.tie_beam_sect_in_width_dir = tie_beams_sect
    fd.tie_beam_sect_in_length_dir = tie_beams_sect
    x = fb.get_column_positions()
    x[0] = fd.pad_length / 2
    x[-1] = fd.length - fd.pad_length / 2
    fd.pad_pos_in_length_dir = x
    fd.n_pads_w = fb.n_frames
    fd.set_pad_pos_in_width_dir_as_equally_spaced()

    return fb, fd, sp, hz


def test_dbd_sfsi_frame_via_millen_et_al_2020():

    fb, fd, sp, hz = load_system(n_storeys=3, n_bays=2)
    designed_frame = dbd.design_rc_frame_w_sfsi_via_millen_et_al_2020(fb, hz, sp, fd, verbose=0)
    print('delta_ss: ', designed_frame.delta_ss)
    print('delta_f: ', designed_frame.delta_f)
    print(designed_frame.axial_load_ratio)
    assert np.isclose(designed_frame.axial_load_ratio, 6.155229455)
    assert np.isclose(designed_frame.delta_ss, 0.13399949371, rtol=0.001)
    assert np.isclose(designed_frame.delta_f, 0.000780144597, rtol=0.001), designed_frame.delta_f


if __name__ == '__main__':
    test_dbd_sfsi_frame_via_millen_et_al_2020()
    # test_calculate_rotation_via_millen_et_al_2020()
