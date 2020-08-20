
from eqdes import dbd_tools as dt
from tests.checking_tools import isclose


def test_effective_stiffness():
    assert isclose(dt.effective_stiffness(300, 2.0), 2959.76, rel_tol=0.0001)
    assert isclose(dt.effective_stiffness(200, 1.0), 7892.7048, rel_tol=0.0001)
    assert isclose(dt.effective_stiffness(100., 2.0), 986.588, rel_tol=0.0001), dt.effective_stiffness(100., 2.0)


def test_equivalent_viscous_damping():

    assert isclose(dt.equivalent_viscous_damping(3.0, mtype="concrete", btype="frame"), 0.16992, rel_tol=0.001)


def test_design_base_shear():
    assert dt.design_base_shear(2.0, 2.0) == 4.0


def test_calculate_storey_forces():
    masses = [2., 2.]
    displacements = [0.5, 0.5]
    v_base = 10.0
    forces = dt.calculate_storey_forces(masses, displacements, v_base, btype='frame')
    assert forces[0] == 4.5
    assert forces[1] == 5.5


def test_cal_higher_mode_factor():
    assert dt.cal_higher_mode_factor(2, btype="frame") == 1.0
    assert dt.cal_higher_mode_factor(17, btype="frame") == 0.85
    assert dt.cal_higher_mode_factor(10, btype="frame") == 0.94


def test_bilinear_load_factor():
    assert dt.bilinear_load_factor(ductility_current=1.0, ductility_max=3.0, r=0.0) == 1.0
    assert dt.bilinear_load_factor(ductility_current=3.0, ductility_max=3.0, r=0.0) == 1.0
    assert dt.bilinear_load_factor(ductility_current=3.0, ductility_max=3.0, r=0.01) == 1.0
    assert dt.bilinear_load_factor(ductility_current=1.0, ductility_max=3.0, r=0.01) == 0.98
    assert dt.bilinear_load_factor(ductility_current=0.5, ductility_max=3.0, r=0.01) == 0.49
    assert dt.bilinear_load_factor(ductility_current=2.0, ductility_max=3.0, r=0.01) == 0.99


def test_ductility():
    assert dt.ductility(10, 1) == 10
    assert dt.ductility(10, 5) == 2
    assert dt.ductility(10, 4) == 2.5
    assert dt.ductility(1, 1) == 1
    assert dt.ductility(0.2, 1) == 0.2


def test_conc_frame_yield_drift():
    assert dt.conc_frame_yield_drift(100, 100000, 10, 0.5) == 0.01


def test_displacement_from_effective_period():
    assert dt.displacement_from_effective_period(0.5, 4, 3, 3) == 2.0
    assert dt.displacement_from_effective_period(0.5, 4, 1.5, 3) == 1.0
    # assert dt.displacement_from_effective_period(0.5, 4, 5, 3) == 2.0


def test_add_foundation():
    ss_heights = [3.4, 6.8, 10.2]
    ss_masses = [40e3, 40e3, 40e3]
    fd_height = 1.0
    fd_mass = 20e3
    heights, masses = dt.add_foundation(ss_heights, ss_masses, fd_height, fd_mass)
    assert heights[0] == 1.0
    assert heights[1] == 4.4
    assert masses[0] == 20e3
    assert masses[1] == 40e3


def test_equivalent_sdof_sfsi():
    hm_factor = 1.0
    heights = [3.4, 6.8, 10.2]
    masses = [40e3, 40e3, 40e3]
    theta_c = 0.02
    displacements = dt.displacement_profile_frame(theta_c, heights, hm_factor)
    delta_fb, mass_eff_fb, height_eff_fb = dt.equivalent_sdof(masses, displacements, heights)
    fd_height = 1.0
    fd_mass = 20.0e3
    theta_f = 0.0
    heights_w_f, masses_w_f = dt.add_foundation(heights, masses, fd_height, fd_mass)
    displacements_sfsi = dt.displacement_profile_frame(theta_c, heights_w_f, hm_factor, foundation=True,
                                                    fd_height=fd_height, theta_f=theta_f)
    delta_sfsi, mass_eff_sfsi, height_eff_sfsi = dt.equivalent_sdof(masses_w_f, displacements_sfsi, heights_w_f)
    print(delta_fb, delta_sfsi)
    assert isclose(delta_fb, delta_sfsi)


if __name__ == '__main__':
    # test_effective_stiffness()
    # test_equivalent_viscous_damping()
    test_equivalent_sdof_sfsi()
