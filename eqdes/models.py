import numpy as np

from sfsimodels import models as sm
from sfsimodels import output as mo
import geofound
from eqdes import nonlinear_foundation as nf
from eqdes import dbd_tools as dt
from eqdes.extensions.exceptions import DesignError


class Soil(sm.Soil):
    required_inputs = ["g_mod",
                      "phi",
                      "unit_weight"
                      ]


class Hazard(sm.SeismicHazard):
    required_inputs = ["corner_disp",
                      "corner_period",
                      "z_factor",
                      "r_factor"
                      ]


class FrameBuilding(sm.FrameBuilding):
    required_inputs = ["interstorey_heights",
                      "floor_length",
                      "floor_width",
                      "storey_masses",
                      "bay_lengths",
                      "beam_depths",
                      "n_seismic_frames",
                      "n_gravity_frames"
                      ]

    def __init__(self, n_storeys, n_bays):
        super(FrameBuilding, self).__init__(n_storeys, n_bays)  # run parent class initialiser function


def to_table(obj, table_name="fb-table"):
    para = mo.output_to_table(obj, olist="all")
    para = mo.add_table_ends(para, 'latex', table_name, table_name)
    return para


class WallBuilding(sm.WallBuilding):
    required_inputs = [
        'floor_length',
        'floor_width',
        'interstorey_heights',
        'n_storeys',
        "n_walls",
        "wall_depth"
    ]


class ReinforcedConcrete(sm.materials.ReinforcedConcreteMaterial):
    required_inputs = [
            'fy',
            'e_mod_steel'
    ]


class RaftFoundation(sm.RaftFoundation):
    required_inputs = [
        "width",
        "length",
        "depth",
        "height",
        "density",
        "i_ww",
        "i_ll"
    ]


class PadFoundation(sm.PadFoundation):
    required_inputs = [
        "width",
        "length",
        "depth",
        "height",
        "density",
        "i_ww",
        "i_ll",
        "n_pads_l",
        "n_pads_w",
        "pad_length",
        "pad_width",
    ]


class DesignedRCFrame(FrameBuilding):
    method = "standard"

    hz = Hazard()

    # outputs
    design_drift = 0.0
    delta_d = 0.0
    mass_eff = 0.0
    height_eff = 0.0
    theta_y = 0.0
    mu = 0.0
    xi = 0.0
    eta = 0.0
    t_eff = 0.0
    v_base = 0.0
    storey_forces = 0.0

    def __init__(self, fb, hz, verbose=0):
        super(DesignedRCFrame, self).__init__(fb.n_storeys, fb.n_bays)  # run parent class initialiser function
        self.__dict__.update(fb.__dict__)
        self.hz.__dict__.update(hz.__dict__)
        self.verbose = verbose
        self.av_beam = np.average(self.beam_depths)
        self.av_bay = np.average(self.bay_lengths)
        assert fb.material.type == 'rc_material'
        self.concrete = fb.material
        self.fye = 1.1 * self.concrete.fy
        self.storey_mass_p_frame = self.storey_masses / self.n_seismic_frames
        self.storey_forces = np.zeros((1, len(self.storey_masses)))
        self.hm_factor = dt.cal_higher_mode_factor(self.n_storeys, btype="frame")
        self._extra_class_variables = ["method"]
        self.inputs += self._extra_class_variables
        self.beam_group_size = 2

    def get_column_base_moments(self):
        cols = self.columns[0, :]
        return np.array([col.sections[0].mom_cap for col in cols])

    def get_beam_face_moments(self, signs=('p', 'p')):
        m_face = [[] for i in range(self.n_storeys)]
        beams = self.beams
        for ns in range(self.n_storeys):
            for nb in range(self.n_bays):
                m_face[ns].append([getattr(beams[ns][nb].sections[0], f'mom_cap_{signs[0]}'),
                                   getattr(beams[ns][nb].sections[-1], f'mom_cap_{signs[1]}')])

        return np.array(m_face)


class DesignedRCWall(WallBuilding):
    method = "standard"
    preferred_bar_diameter = 0.032

    hz = Hazard()

    # outputs
    phi_material = 0.0
    epsilon_y = 0.0
    fu = 0.0
    design_drift = 0.0
    delta_d = 0.0
    mass_eff = 0.0
    height_eff = 0.0
    theta_y = 0.0
    mu = 0.0
    xi = 0.0
    eta = 0.0
    t_eff = 0.0
    v_base = 0.0
    storey_forces = 0.0

    def __init__(self, wb, hz, verbose=0):
        super(DesignedRCWall, self).__init__(wb.n_storeys)  # run parent class initialiser function
        self.__dict__.update(wb.__dict__)
        self.hz.__dict__.update(hz.__dict__)
        self.verbose = verbose
        assert wb.material.type == 'rc_material'
        self.concrete = wb.material
        self.fye = 1.1 * self.concrete.fy
        self.storey_mass_p_wall = self.storey_masses / self.n_walls
        self.storey_forces = np.zeros((1, len(self.storey_masses)))
        self.hm_factor = dt.cal_higher_mode_factor(self.n_storeys, btype="wall")
        self._extra_class_variables = ["method"]
        self.inputs += self._extra_class_variables

    def static_dbd_values(self):
        # Material strain limits check
        self.phi_material = 0.072 / self.wall_depth  # Eq 6.10b
        self.fye = 1.1 * self.concrete.fy
        self.epsilon_y = self.fye / self.concrete.e_mod_steel
        self.fu = 1.40 * self.fye  # Assumed, see pg 141


class AssessedRCFrame(FrameBuilding):
    method = "standard"
    post_yield_stiffness_ratio = 0.05

    hz = Hazard()

    # inputs
    otm_max = 0.0
    theta_max = 0.0

    # outputs
    assessed_drift = 0.0
    delta_max = 0.0
    mass_eff = 0.0
    height_eff = 0.0
    theta_y = 0.0
    max_mu = 0.0
    mu = 0.0
    xi = 0.0
    eta = 0.0
    t_eff = 0.0
    v_base = 0.0
    storey_forces = 0.0

    def __init__(self, fb, hz, verbose=0):
        super(AssessedRCFrame, self).__init__(n_bays=fb.n_bays, n_storeys=fb.n_storeys)  # run parent class initialiser function
        assert fb.material.type == 'rc_material'
        self.concrete = fb.material
        self.__dict__.update(fb.__dict__)
        self.hz.__dict__.update(hz.__dict__)
        self.verbose = verbose
        self.av_beam = np.average(self.beam_depths)
        self.av_bay = np.average(self.bay_lengths)
        self.fye = 1.1 * self.concrete.fy
        self.storey_mass_p_frame = self.storey_masses / self.n_seismic_frames
        self.storey_forces = np.zeros((1, len(self.storey_masses)))
        self.hm_factor = dt.cal_higher_mode_factor(self.n_storeys, btype="frame")
        self._extra_class_variables = ["method"]
        self.inputs += self._extra_class_variables


class DesignedSFSIRCFrame(DesignedRCFrame):

    sl = sm.Soil()
    fd = sm.RaftFoundation()
    total_weight = 0.0
    theta_f = 0.0
    axial_load_ratio = 0.0
    theta_pseudo_up = 0.0

    def __init__(self, fb, hz, sl, fd, ip_axis='length', horz2vert_mass=None):
        super(DesignedSFSIRCFrame, self).__init__(fb, hz)  # run parent class initialiser function
        self.sl.__dict__.update(sl.__dict__)
        # self.fd.__dict__.update(fd.__dict__)
        self.fd = fd.deepcopy()
        self.k_f0_shear = geofound.stiffness.calc_shear_via_gazetas_1991(self.sl, self.fd, ip_axis=ip_axis)
        self.k_f_0 = geofound.stiffness.calc_rotational_via_gazetas_1991(self.sl, self.fd, ip_axis=ip_axis)
        if self.fd.ftype == "raft":
            self.alpha = 4.0
        else:
            self.alpha = 3.0

        self.zeta = 1.5
        if horz2vert_mass is not None:
            self.horz2vert_mass = horz2vert_mass
        self.beam_group_size = 2

    def static_values(self):
        self.total_weight = self.horz2vert_mass * (sum(self.storey_masses) + self.fd.mass) * self.g
        if hasattr(self.fd, 'pad'):
            self.soil_q = geofound.capacity_salgado_2008(sl=self.sl, fd=self.fd.pad)
        else:
            self.soil_q = geofound.capacity_salgado_2008(sl=self.sl, fd=self.fd)

        # Deal with both raft and pad foundations
        bearing_capacity = nf.bearing_capacity(self.fd.area, self.soil_q)
        weight_per_frame = self.horz2vert_mass * sum(self.storey_masses) / (self.n_seismic_frames + self.n_gravity_frames) * self.g
        self.axial_load_ratio = bearing_capacity / self.total_weight
        self.fd_bearing_capacity = bearing_capacity

        self.theta_pseudo_up = nf.calculate_pseudo_uplift_angle(self.total_weight, self.fd.width, self.k_f_0,
                                                                self.axial_load_ratio, self.alpha, self.zeta)


def designed_frame_table(fb, table_name="df-table"):
    para = mo.output_to_table(fb, olist="all")
    para += mo.output_to_table(fb.fd)
    para += mo.output_to_table(fb.sl)
    para += mo.output_to_table(fb.hz)
    para = mo.add_table_ends(para, 'latex', table_name, table_name)
    return para

#
# class Soil(object):
#     pass


class DesignedSFSIRCWall(DesignedRCWall):

    sl = sm.Soil()
    fd = sm.RaftFoundation()
    total_weight = 0.0
    theta_f = 0.0
    axial_load_ratio = 0.0
    bearing_capacity = 0.0
    theta_pseudo_up = 0.0

    def __init__(self, wb, hz, sl, fd):
        super(DesignedSFSIRCWall, self).__init__(wb, hz)  # run parent class initialiser function
        self.sl.__dict__.update(sl.__dict__)
        self.fd.__dict__.update(fd.__dict__)
        self.k_f0_shear = geofound.stiffness.calc_shear_via_gazetas_1991(self.sl, self.fd, ip_axis='length')

        if self.fd.ftype == "raft":
            self.alpha = 4.0
        else:
            self.alpha = 3.0
        self.k_f_0 = geofound.stiffness.calc_rotational_via_gazetas_1991(self.sl, self.fd, ip_axis='length')
        self.zeta = 1.5

    def static_values(self):
        self.total_weight = (sum(self.storey_masses) + self.fd.mass) * self.g
        soil_q = geofound.capacity_salgado_2008(sl=self.sl, fd=self.fd)

        # Deal with both raft and pad foundations
        self.bearing_capacity = nf.bearing_capacity(self.fd.area, soil_q)
        self.axial_load_ratio = self.bearing_capacity / self.total_weight

        self.theta_pseudo_up = nf.calculate_pseudo_uplift_angle(self.total_weight, self.fd.width, self.k_f_0,
                                                                self.axial_load_ratio, self.alpha, self.zeta)


def design_wall_to_table(dw, table_name="df-table"):
    para = mo.output_to_table(dw, olist="all")
    para += mo.output_to_table(dw.fd)
    para += mo.output_to_table(dw.sl)
    para += mo.output_to_table(dw.hz)
    para = mo.add_table_ends(para, 'latex', table_name, table_name)
    return para


class AssessedSFSIRCFrame(AssessedRCFrame):
    sl = sm.Soil()
    fd = sm.Foundation()
    total_weight = 0.0
    theta_f = 0.0
    axial_load_ratio = 0.0
    theta_pseudo_up = 0.0

    def __init__(self, fb, hz, sl, fd, ip_axis='length', horz2vert_mass=None):
        super(AssessedSFSIRCFrame, self).__init__(fb, hz)  # run parent class initialiser function
        self.sl.__dict__.update(sl.__dict__)
        if fd.ftype == "raft":
            self.fd = sm.RaftFoundation()
        if fd.ftype == "pad":
            self.fd = sm.PadFoundation()
        self.fd.__dict__.update(fd.__dict__)
        self.k_f0_shear = geofound.stiffness.calc_shear_via_gazetas_1991(self.sl, self.fd, ip_axis=ip_axis)
        self.k_f_0 = geofound.stiffness.calc_rotational_via_gazetas_1991(self.sl, self.fd, ip_axis=ip_axis)
        if self.fd.ftype == "raft":
            #self.k_f_0 = nf.rotational_stiffness(self.fd.width, self.fd.length, self.sl.g_mod, self.sl.poissons_ratio)
            self.alpha = 4.0
        else:
            #self.k_f_0 = nf.rotational_stiffness(self.fd.width, self.fd.length, self.sl.g_mod, self.sl.poissons_ratio) / 2
            self.alpha = 3.0

        self.zeta = 1.5
        if horz2vert_mass is not None:
            self.horz2vert_mass = horz2vert_mass
        assert fb.material.type == 'rc_material'
        self.concrete = fb.material


    def static_values(self):
        self.total_weight = (sum(self.storey_masses) + self.fd.mass) * self.g * self.horz2vert_mass
        if hasattr(self.fd, 'pad_length'):
            pad = sm.PadFoundation()
            pad.length = self.fd.pad_length
            pad.width = self.fd.pad_width
            pad.height = self.fd.height
            pad.depth = self.fd.depth
            self.soil_q = geofound.capacity_salgado_2008(sl=self.sl, fd=pad)
        else:
            self.soil_q = geofound.capacity_salgado_2008(sl=self.sl, fd=self.fd)
        # Add new function to foundations, bearing_capacity_from_sfsimodels,
        # Deal with both raft and pad foundations
        bearing_capacity = nf.bearing_capacity(self.fd.area, self.soil_q)
        weight_per_frame = sum(self.storey_masses) / (self.n_seismic_frames + self.n_gravity_frames) * self.g
        self.axial_load_ratio = bearing_capacity / self.total_weight
        if self.axial_load_ratio < 1.0:
            raise DesignError("Static failure expected. Axial load ratio: %.3f" % self.axial_load_ratio)

        self.theta_pseudo_up = nf.calculate_pseudo_uplift_angle(self.total_weight, self.fd.width, self.k_f_0,
                                                                self.axial_load_ratio, self.alpha, self.zeta)

def to_table(aw, table_name="af-table"):
    para = mo.output_to_table(aw, olist="inputs")
    para += mo.output_to_table(aw.fd, prefix="Foundation ")
    para += mo.output_to_table(aw.sl, prefix="Soil ")
    para += mo.output_to_table(aw.hz, prefix="Hazard ")
    para = mo.add_table_ends(para, 'latex', table_name, table_name)
    return para
