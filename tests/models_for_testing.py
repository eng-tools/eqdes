
import numpy as np
from eqdes import models as em


def load_hazard_test_data(hz):
    """
    Sample data for the Hazard object

    :param hz: sfsimodels.models.Hazard Object
    :return:
    """
    # hazard
    hz.z_factor = 0.3  # Hazard factor
    hz.r_factor = 1.0  # Return period factor
    hz.n_factor = 1.0  # Near-fault factor
    hz.magnitude = 7.5  # Magnitude of earthquake
    hz.corner_period = 4.0  # s
    hz.corner_acc_factor = 0.55


def initialise_hazard_test_data():
    hz = em.Hazard()
    load_hazard_test_data(hz)
    return hz


def initialise_frame_building_test_data():
    """
    Sample data for the FrameBuilding object

    :param fb:
    :return:
    """
    number_of_storeys = 6
    number_of_bays = 3
    fb = em.FrameBuilding(n_storeys=number_of_storeys, n_bays=number_of_bays)
    fb.material = em.ReinforcedConcrete()
    interstorey_height = 3.4  # m
    masses = 40.0e3  # kg

    fb.interstorey_heights = interstorey_height * np.ones(number_of_storeys)
    fb.floor_length = 18.0  # m
    fb.floor_width = 16.0  # m
    fb.storey_masses = masses * np.ones(number_of_storeys)  # kg

    fb.bay_lengths = [6., 6.0, 6.0]
    fb.set_beam_prop("depth", [.5, .5, .5])
    fb.n_seismic_frames = 3
    fb.n_gravity_frames = 0
    fb.horz2vert_mass = 1
    return fb


def initialise_wall_building_test_data():
    """
    Sample data for the WallBuilding object

    :return: WallBuilding
    """
    number_of_storeys = 6
    wb = em.WallBuilding(number_of_storeys)
    wb.material = em.ReinforcedConcrete()

    interstorey_height = 3.4  # m
    masses = 40.0e3  # kg

    wb.interstorey_heights = interstorey_height * np.ones(number_of_storeys)
    wb.floor_length = 18.0  # m
    wb.floor_width = 16.0  # m
    wb.storey_masses = masses * np.ones(number_of_storeys)  # kg

    wb.n_walls = 4
    wb.wall_depth = 2.0
    return wb


def load_soil_test_data(sl):
    """
    Sample data for the Soil object

    :param sp: Soil Object
    :return:
    """
    # soil
    sl.g_mod = 60.0e6  # [Pa]
    sl.phi = 30  # [degrees]
    sl.relative_density = .40  # [decimal]
    sl.unit_dry_weight = 17000  # [N/m3]
    sl.cohesion = 10.0  # [Pa]
    sl.poissons_ratio = 0.22


def initialise_soil_test_data():
    sl = em.Soil()
    load_soil_test_data(sl)
    return sl


def load_raft_foundation_test_data(fd):
    """
    Sample data for the Foundation object

    :param fd: Foundation Object
    :return:
    """
    # foundation
    fd.width = 16.0  # m
    fd.length = 18.0  # m
    fd.depth = 0.0  # m
    fd.mass = 0.0
    fd.height = 1.0


def initialise_foundation_test_data():
    fd = em.RaftFoundation()
    load_raft_foundation_test_data(fd)
    return fd