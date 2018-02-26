
import numpy as np

from tests import models_for_testing as ml
from eqdes import ddbd
from eqdes import models as dm

from tests.checking_tools import isclose


def test_ddbd_frame_fixed_small():

    fb = dm.FrameBuilding()
    hz = dm.Hazard()
    ml.load_hazard_test_data(hz)
    ml.load_large_frame_building_test_data(fb)
    frame_ddbd = ddbd.dbd_frame(fb, hz)

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

    fb = dm.FrameBuilding()
    hz = dm.Hazard()
    sl = dm.Soil()
    fd = dm.RaftFoundation()
    ml.load_hazard_test_data(hz)
    ml.load_large_frame_building_test_data(fb)
    ml.load_soil_test_data(sl)
    ml.load_raft_foundation_test_data(fd)
    frame_ddbd = ddbd.dbd_frame(fb, hz)
    sl.g_mod = 1.0e10  # make soil very stiff
    fd.height = 2.0  # add some height to the foundation
    frame_sfsi_dbd = ddbd.dbd_sfsi_frame(fb, hz, sl, fd, found_rot=1e-6)
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

    fb = dm.FrameBuilding()
    hz = dm.Hazard()

    n_storeys = 5
    fb.interstorey_heights = 3.6 * np.ones(n_storeys)
    n_bays = 1
    fb.bay_lengths = 6.0 * np.ones(n_bays)
    fb.beam_depth = .6 * np.ones(n_storeys)  # m      #varies vertically

    hz.z_factor = 0.3
    hz.r_factor = 1.0
    hz.n_factor = 1.0

    design_drift = 0.02

    fb.n_seismic_frames = 1
    fb.n_gravity_frames = 0

    fb.floor_length = sum(fb.bay_lengths)
    fb.floor_width = 14.0  # m

    fb.storey_masses = np.array([488.0, 488.0, 488.0, 488.0, 411.0]) * 1e3

    frame_ddbd = ddbd.dbd_frame(fb, hz, design_drift=design_drift)

    # StoreyForcesCheck1 = np.array([460402.85, 872342.25, 1235818.18, 1550830.66, 2158400.44])
    # assert isclose(frame_ddbd.v_base, 4889353.79)  # 6277794.38  # TODO: fix this
    # assert isclose(frame_ddbd.Storey_Forces, StoreyForcesCheck1)


def test_dbd_sfsi_frame():
    fb = dm.FrameBuilding()
    fb.n_bays = 1
    fb.raft_height = 0  # m
    fb.footing_mass = 0
    fb.raft_foundation = 0
    fb.pad_depth = 0.8 * np.ones((fb.n_bays + 1))
    fb.pad_width = 2.0 * np.ones((fb.n_bays + 1))  # m
    fb.tie_depth = 0.8 * np.ones(fb.n_bays)  # m
    fb.tie_width = 0.8 * np.ones(fb.n_bays)  # m
    fb.foundation_rotation = 1e-3
    fb.discrete_rotation_ratio = 1.0
    fb.AxialLoadRatio = 26  # Should calculate this
    fb.Base_moment_contribution = 0.6
    fb.beam_group_size = 1

    # Soil properties
    fb.Soil_G_modulus = 80e6  # Pa
    fb.Soil_Poissons_ratio = 0.3  # Poisson's ratio of the soil
    fb.Soil_DR = 60  # %
    fb.Soil_phi = 35


def test_ddbd_wall_fixed():

    fb = dm.WallBuilding()
    hz = dm.Hazard()
    ml.load_hazard_test_data(hz)
    ml.load_wall_building_test_data(fb)
    wall_dbd = ddbd.wall(fb, hz)

    assert isclose(wall_dbd.delta_d, 0.339295, rel_tol=0.001), wall_dbd.delta_d
    assert isclose(wall_dbd.mass_eff, 59429.632, rel_tol=0.001), wall_dbd.mass_eff
    assert isclose(wall_dbd.height_eff, 12.46885, rel_tol=0.001), wall_dbd.height_eff
    assert isclose(wall_dbd.mu, 1.1902, rel_tol=0.001), wall_dbd.mu
    assert isclose(wall_dbd.theta_y, 0.0168299, rel_tol=0.001), wall_dbd.theta_y
    assert isclose(wall_dbd.xi, 0.07259, rel_tol=0.001), wall_dbd.xi
    assert isclose(wall_dbd.eta, 0.86946, rel_tol=0.001), wall_dbd.eta
    assert isclose(wall_dbd.t_eff, 2.38184, rel_tol=0.001), wall_dbd.t_eff

if __name__ == '__main__':
    test_ddbd_frame_fixed_large()
