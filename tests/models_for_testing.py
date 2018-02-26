
import numpy as np


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


# def load_building_test_data(bd):
#     """
#     Sample data for the Building object
#     :param bd:
#     :return:
#     """
#     number_of_storeys = 8
#     interstorey_height = 3.4  # m
#     masses = 40.0e3  # kg
#
#     bd.interstorey_heights = interstorey_height * np.ones(number_of_storeys)
#     bd.floor_length = 18.0  # m
#     bd.floor_width = 16.0  # m
#     bd.storey_masses = np.array([masses])  # kg


def load_large_building_test_data(bd):
    """
    Sample data for the Building object
    :param bd:
    :return:
    """
    number_of_storeys = 6
    interstorey_height = 3.4  # m
    masses = 40.0e3  # kg

    bd.interstorey_heights = interstorey_height * np.ones(number_of_storeys)
    bd.floor_length = 18.0  # m
    bd.floor_width = 16.0  # m
    bd.storey_masses = masses * np.ones(number_of_storeys)  # kg


def load_frame_building_test_data(fb):
    """
    Sample data for the FrameBuilding object
    :param fb:
    :return:
    """
    load_large_building_test_data(fb)

    fb.bay_lengths = [6., 6.0, 6.0]
    fb.beam_depths = [.5]
    fb.n_seismic_frames = 3
    fb.n_gravity_frames = 0


def load_wall_building_test_data(fb):
    """
    Sample data for the WallBuilding object
    :param fb:
    :return:
    """
    load_large_building_test_data(fb)

    fb.n_walls = 4
    fb.wall_depth = 2.0


def load_large_frame_building_test_data(fb):
    """
    Sample data for the FrameBuilding object
    :param fb:
    :return:
    """
    load_large_building_test_data(fb)

    fb.bay_lengths = [6., 6.0, 6.0]
    fb.beam_depths = [.5]
    fb.n_seismic_frames = 3
    fb.n_gravity_frames = 0


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
