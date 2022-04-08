
import numpy as np

from tests import models_for_testing as ml
from eqdes import dbd
from eqdes import dba
from eqdes import models as dm
from eqdes import design_spectra
import sfsimodels as sm
import geofound as gf
import eqdes

from tests.checking_tools import isclose


def test_ddbd_frame_fixed_small():

    hz = dm.Hazard()
    ml.load_hazard_test_data(hz)
    fb = ml.initialise_frame_building_test_data()
    frame_dbd = dbd.design_rc_frame(fb, hz)
    otm_max_approx = np.sum(frame_dbd.storey_forces * frame_dbd.heights)

    assert isclose(frame_dbd.delta_d, 0.2400, rel_tol=0.001), frame_dbd.delta_d
    assert isclose(frame_dbd.mass_eff, 67841.581, rel_tol=0.001), frame_dbd.mass_eff
    assert isclose(frame_dbd.height_eff, 14.34915, rel_tol=0.001), frame_dbd.height_eff
    assert isclose(frame_dbd.mu, 1.689, rel_tol=0.001), frame_dbd.mu
    assert isclose(frame_dbd.theta_y, 0.0099, rel_tol=0.001), frame_dbd.theta_y
    assert isclose(frame_dbd.xi, 0.123399, rel_tol=0.001), frame_dbd.xi
    assert isclose(frame_dbd.eta, 0.69867, rel_tol=0.001), frame_dbd.eta
    assert isclose(frame_dbd.t_eff, 2.09646, rel_tol=0.001), frame_dbd.t_eff

    af = dba.assess_rc_frame(frame_dbd, hz, theta_max=frame_dbd.design_drift, otm_max=otm_max_approx)

    assert isclose(af.delta_d, 0.2400, rel_tol=0.001), af.delta_d
    assert isclose(af.mass_eff, 67841.581, rel_tol=0.001), af.mass_eff
    assert isclose(af.height_eff, 14.34915, rel_tol=0.001), af.height_eff
    assert isclose(af.mu, 1.6509, rel_tol=0.001), af.mu
    assert isclose(af.theta_y, 0.0099, rel_tol=0.001), af.theta_y
    assert isclose(af.xi, 0.1201, rel_tol=0.001), af.xi
    assert isclose(af.eta, 0.70683, rel_tol=0.001), af.eta
    assert isclose(af.t_eff, 2.025, rel_tol=0.001), af.t_eff


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
    otm_max_approx = np.sum(frame_ddbd.storey_forces * frame_ddbd.heights)
    af_sfsi = dba.assess_rc_frame_w_sfsi_via_millen_et_al_2020(fb, hz, sl, fd, theta_max=frame_ddbd.design_drift, otm_max=otm_max_approx)
    assert isclose(af_sfsi.theta_f, 0.0, abs_tol=1e-5)

    assert isclose(frame_ddbd.delta_d, af_sfsi.delta_ss, rel_tol=0.05), (frame_ddbd.delta_d, af_sfsi.delta_ss)
    assert isclose(frame_ddbd.mass_eff, af_sfsi.mass_eff, rel_tol=0.05), af_sfsi.mass_eff
    assert isclose(frame_ddbd.height_eff, af_sfsi.height_eff - fd.height, rel_tol=0.05), af_sfsi.height_eff
    assert isclose(frame_ddbd.mu, af_sfsi.mu, rel_tol=0.05), af_sfsi.mu
    assert isclose(frame_ddbd.theta_y, af_sfsi.theta_y, rel_tol=0.05), af_sfsi.theta_y
    assert isclose(frame_ddbd.xi, af_sfsi.xi, rel_tol=0.03), af_sfsi.xi
    assert isclose(frame_ddbd.eta, af_sfsi.eta, rel_tol=0.03), af_sfsi.eta
    assert isclose(frame_ddbd.t_eff, af_sfsi.t_eff, rel_tol=0.05), (af_sfsi.t_eff, frame_ddbd.t_eff)


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
    fd.pad_length = pad.length
    fd.pad_width = pad.width
    fd.pad.depth = fd.depth
    fd.pad.height = fd.height
    tie_beams = sm.std.create_rc_beam(depth=fd.height, width=fd.height, n_sects=1)
    tie_beams.set_section_prop('cracked_ratio', 0.6)
    fd.tie_beam_in_width_dir = tie_beams
    fd.tie_beam_in_length_dir = tie_beams
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
    assert np.isclose(designed_frame.axial_load_ratio, 4.54454554), designed_frame.axial_load_ratio
    assert np.isclose(designed_frame.delta_ss, 0.134327, rtol=0.001), designed_frame.delta_ss
    assert np.isclose(designed_frame.delta_f, 0.000780144597, rtol=0.001), designed_frame.delta_f

    ps = eqdes.moment_equilibrium.assess(designed_frame, designed_frame.storey_forces)
    moment_beams_cl = ps[0]
    moment_column_bases = ps[1]
    axial_seismic = ps[2]
    eqdes.moment_equilibrium.set_beam_face_moments_from_centreline_demands(fb, moment_beams_cl, centre_sect=True)
    eqdes.moment_equilibrium.set_column_base_moments_from_demands(fb, moment_column_bases)
    otm_max = eqdes.moment_equilibrium.calc_otm_capacity(fb)

    otm_max_approx = np.sum(designed_frame.storey_forces[1:] * designed_frame.heights)

    af = dba.assess_rc_frame_w_sfsi_via_millen_et_al_2020(fb, hz, sp, fd, theta_max=designed_frame.design_drift, otm_max=otm_max)


if __name__ == '__main__':
    test_dbd_sfsi_frame_via_millen_et_al_2020()