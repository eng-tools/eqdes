
import numpy as np

from tests import models_for_testing as ml
from eqdes import dbd
from eqdes import models as dm
from eqdes import design_spectra
import sfsimodels as sm
import geofound as gf

from tests.checking_tools import isclose


def test_ddbd_frame_fixed_small():

    hz = dm.Hazard()
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
    hz = dm.Hazard()
    sl = dm.Soil()
    fd = dm.RaftFoundation()
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
    fb = dm.FrameBuilding(n_storeys, n_bays)
    fb.material = dm.ReinforcedConcrete()
    hz = dm.Hazard()

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
    fb = dm.FrameBuilding(n_storeys, n_bays)
    fb.n_seismic_frames = 2
    fb.n_gravity_frames = 0
    fb.material = dm.ReinforcedConcrete()
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
    fd = dm.PadFoundation()
    fd.height = 0  # m
    fd.length = 5
    fd.width = 5.
    fd.depth = 0.8

    fd.pad.width = 1.4  # m
    fd.pad.length = 1.4  # m
    fd.pad.depth = 0.5  # m
    fd.pad.height = 0
    fd.n_pads_l = 2
    fd.n_pads_w = 2
    fd.mass = 0
    fd2 = fd.deepcopy()

    # Soil properties

    hz = dm.Hazard()
    hz.z_factor = 0.3
    hz.r_factor = 1.0
    hz.n_factor = 1.0
    hz.corner_acc_factor = 2.
    hz.corner_period = 1

    sl = dm.Soil()
    sl.g_mod = 120e6  # Pa
    sl.poissons_ratio = 0.3  # Poisson's ratio of the soil
    sl.e_curr = 0.6  # %
    sl.phi = 35.
    sl.cohesion = 0
    sl.specific_gravity = 2.65

    design_drift = 0.02

    frame_ddbd = dbd.design_rc_frame_w_sfsi_via_millen_et_al_2018(fb, hz, sl, fd, design_drift=design_drift, verbose=2)
    assert np.isclose(frame_ddbd.delta_d, 0.08488596), frame_ddbd.delta_d
    assert np.isclose(frame_ddbd.theta_f, 0.0050136357), frame_ddbd.theta_f


def to_be_test_ddbd_sfsi_wall_from_millen_pdf_paper_2018():
    fb = dm.FrameBuilding()
    sl = dm.Soil()
    fd = dm.PadFoundation()
    hz = dm.Hazard()
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
    pad = gf.size_footing_for_capacity(sp, np.max(col_loads), method='salgado', fos=3., depth=fd.depth)
    tie_beams_sect = sm.sections.RCBeamSection()
    tie_beams_sect.depth = fd.height
    tie_beams_sect.width = fd.height
    tie_beams_sect.rc_mat = sm.materials.ReinforcedConcreteMaterial()
    tie_beams_sect.cracked_ratio = 0.6
    fd.tie_beam_sect_in_width_dir = tie_beams_sect
    fd.tie_beam_sect_in_length_dir = tie_beams_sect

    if fd.ftype == "pad":
        fd.n_pads_l = fb.n_cols  # Number of pads in length direction
        fd.n_pads_w = fb.n_frames  # Number of pads in width direction
    fd.pad_length = pad.length
    fd.pad_width = pad.width
    fd.pad.depth = fd.depth
    fd.pad.height = fd.height

    return fb, fd, sp, hz


def test_dbd_sfsi_frame_via_millen_et_al_2020():

    fb, fd, sp, hz = load_system(n_storeys=3, n_bays=2)
    designed_frame = dbd.design_rc_frame_w_sfsi_via_millen_et_al_2020(fb, hz, sp, fd, verbose=0)
    print('delta_ss: ', designed_frame.delta_ss)
    print('delta_f: ', designed_frame.delta_f)
    print(designed_frame.axial_load_ratio)
    assert np.isclose(designed_frame.axial_load_ratio, 4.54454554)
    assert np.isclose(designed_frame.delta_ss, 0.1352084271)
    assert np.isclose(designed_frame.delta_f, 0.0007801446), designed_frame.delta_f


def test_case_study_wall_pbd_wall():
    def create(save=0, show=1):
        n_storeys = 6
        wb = dm.WallBuilding(n_storeys)
        wb.wall_width = 0.3  # m
        wb.wall_depth = 3.4  # m
        wb.interstorey_heights = 3.4 * np.ones(n_storeys)  # m
        wb.n_walls = 1
        wb.floor_length = 20 / 2  # m
        wb.floor_width = 12 / 2  # m
        g_load = 6000.  # Pa
        q_load = 3000.  # Pa
        eq_load_factor = 0.4
        floor_pressure = g_load + eq_load_factor * q_load
        wb.set_storey_masses_by_pressure(floor_pressure)

        fd = dm.RaftFoundation()
        fd.height = 1.3
        fd.length = 5.6  # m # from HDF
        fd.width = 2.25  # m # from HDF
        fd.depth = 0.0  # TODO: check this
        fd.mass = 0.0

        # soil properties from HDF
        sl = dm.Soil()
        sl.g_mod = 40e6  # Pa
        sl.poissons_ratio = 0.3
        sl.phi = 36.0  # degrees
        # sl.phi_r = np.radians(sl.phi)
        sl.cohesion = 0.0
        sl.unit_dry_weight = 18000.  # TODO: check this

        # hazard
        hz = dm.Hazard()
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

        # dw = dbd.wall(wb, hz, design_drift=0.025)
        dw = dbd.design_rc_wall(wb, hz, sl, fd, design_drift=0.025)
        print(dw.delta_d)


def test_ddbd_wall_fixed():

    hz = dm.Hazard()
    ml.load_hazard_test_data(hz)
    wb = ml.initialise_wall_building_test_data()
    wall_dbd = dbd.design_rc_wall(wb, hz)

    assert isclose(wall_dbd.delta_d, 0.339295, rel_tol=0.001), wall_dbd.delta_d
    assert isclose(wall_dbd.mass_eff, 59429.632, rel_tol=0.001), wall_dbd.mass_eff
    assert isclose(wall_dbd.height_eff, 12.46885, rel_tol=0.001), wall_dbd.height_eff
    assert isclose(wall_dbd.mu, 1.1902, rel_tol=0.001), wall_dbd.mu
    assert isclose(wall_dbd.theta_y, 0.0168299, rel_tol=0.001), wall_dbd.theta_y
    assert isclose(wall_dbd.xi, 0.07259, rel_tol=0.001), wall_dbd.xi
    assert isclose(wall_dbd.eta, 0.86946, rel_tol=0.001), wall_dbd.eta
    assert isclose(wall_dbd.t_eff, 2.38184, rel_tol=0.001), wall_dbd.t_eff


def test_calculate_rotation_via_millen_et_al_2020():
    mom = 200.
    k_rot = 1000.0e2
    psi = 0.4
    h_eff = 3.0
    l_in = 3.0
    n_load = 300.
    n_cap = 3000.
    theta = dbd.calc_fd_rot_via_millen_et_al_2020_alt_form(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
    assert np.isclose(theta, 0.0053710398), theta
    theta = dbd.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
    assert np.isclose(theta, 0.0053710398), theta

    n_load = 2000.
    n_cap = 3000.
    l_in = 5.0
    mom = 100.
    theta = dbd.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
    assert np.isclose(theta, 0.0011910855), theta

    # very large moment
    mom = 3000.
    theta = dbd.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
    assert theta is None

    # n_load equal to n_cap
    mom = 10
    n_load = 300.
    n_cap = 300.
    theta = dbd.calc_fd_rot_via_millen_et_al_2020_alt_form(k_rot, l_in, n_load, n_cap, psi, mom, h_eff)
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

    theta_w_tbs = dbd.calc_fd_rot_via_millen_et_al_2020_w_tie_beams(k_rot, l_in, n_load, n_cap, psi, mom,
                                                                          h_eff,
                                                                          k_tbs=k_tbs)

    mom_tbs = theta_w_tbs * k_tbs
    rots_adj = dbd.calc_fd_rot_via_millen_et_al_2020(k_rot, l_in, n_load, n_cap, psi, mom - mom_tbs, h_eff)
    assert np.isclose(theta_w_tbs, rots_adj, rtol=0.01)


if __name__ == '__main__':
    test_dbd_sfsi_frame_via_millen_et_al_2020()
    # test_calculate_rotation_via_millen_et_al_2020()
