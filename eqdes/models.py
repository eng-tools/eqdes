import numpy as np

from sfsimodels import models as sm
from sfsimodels import output as mo
from geofound import capacity
from geofound import stiffness
from eqdes import nonlinear_foundation as nf
from eqdes import ddbd_tools as dt
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

    def to_table(self, table_name="fb-table"):
        para = mo.output_to_table(self, olist="all")
        para += mo.output_to_table(self.hz)
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

    def to_table(self, table_name="wb-table"):
        para = mo.output_to_table(self, olist="all")
        para = mo.add_table_ends(para, 'latex', table_name, table_name)
        return para


class Concrete(sm.material.Concrete):
    required_inputs = [
            'fy',
            'youngs_steel'
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
        "i_ll"
    ]


class DesignedFrame(FrameBuilding):
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
        super(DesignedFrame, self).__init__()  # run parent class initialiser function
        self.__dict__.update(fb.__dict__)
        self.hz.__dict__.update(hz.__dict__)
        self.verbose = verbose
        self.av_beam = np.average(self.beam_depths)
        self.av_bay = np.average(self.bay_lengths)
        self.fye = 1.1 * self.concrete.fy
        self.storey_mass_p_frame = self.storey_masses / self.n_seismic_frames
        self.storey_forces = np.zeros((1, len(self.storey_masses)))
        self.hm_factor = dt.cal_higher_mode_factor(self.n_storeys, btype="frame")

    @property
    def inputs(self):
        input_list = super(DesignedFrame, self).inputs
        new_inputs = [
        "method",
    ]
        return input_list + new_inputs


class DesignedWall(WallBuilding):
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
        super(DesignedWall, self).__init__()  # run parent class initialiser function
        self.__dict__.update(wb.__dict__)
        self.hz.__dict__.update(hz.__dict__)
        self.verbose = verbose
        self.fye = 1.1 * self.concrete.fy
        self.storey_mass_p_wall = self.storey_masses / self.n_walls
        self.storey_forces = np.zeros((1, len(self.storey_masses)))
        self.hm_factor = dt.cal_higher_mode_factor(self.n_storeys, btype="wall")

    @property
    def inputs(self):
        input_list = super(DesignedWall, self).inputs
        new_inputs = [
        "method",
    ]
        return input_list + new_inputs

    def static_dbd_values(self):
        # Material strain limits check
        self.phi_material = 0.072 / self.wall_depth  # Eq 6.10b
        self.fye = 1.1 * self.concrete.fy
        self.epsilon_y = self.fye / self.concrete.youngs_steel
        self.fu = 1.40 * self.fye  # Assumed, see pg 141


class AssessedFrame(FrameBuilding):
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
        super(AssessedFrame, self).__init__()  # run parent class initialiser function
        self.__dict__.update(fb.__dict__)
        self.hz.__dict__.update(hz.__dict__)
        self.verbose = verbose
        self.av_beam = np.average(self.beam_depths)
        self.av_bay = np.average(self.bay_lengths)
        self.fye = 1.1 * self.concrete.fy
        self.storey_mass_p_frame = self.storey_masses / self.n_seismic_frames
        self.storey_forces = np.zeros((1, len(self.storey_masses)))
        self.hm_factor = dt.cal_higher_mode_factor(self.n_storeys, btype="frame")

    @property
    def inputs(self):
        input_list = super(AssessedFrame, self).inputs
        new_inputs = [
        "method",
    ]
        return input_list + new_inputs


class DesignedSFSIFrame(DesignedFrame):

    sl = sm.Soil()
    fd = sm.RaftFoundation()
    total_weight = 0.0
    theta_f = 0.0
    axial_load_ratio = 0.0
    theta_pseudo_up = 0.0

    def __init__(self, fb, hz, sl, fd):
        super(DesignedSFSIFrame, self).__init__(fb, hz)  # run parent class initialiser function
        self.sl.__dict__.update(sl.__dict__)
        self.fd.__dict__.update(fd.__dict__)
        self.k_f0_shear = stiffness.shear_stiffness(self.fd.width, self.fd.length, self.sl.g_mod, self.sl.poissons_ratio)

        if self.fd.ftype == "raft":
            self.k_f_0 = stiffness.rotational_stiffness(self.sl, self.fd)
            self.alpha = 4.0
        else:
            self.k_f_0 = stiffness.rotational_stiffness(self.sl, self.fd)
            self.alpha = 3.0

        self.zeta = 1.5

    def static_values(self):
        self.total_weight = (sum(self.storey_masses) + self.fd.mass) * self.g
        soil_q = capacity.salgado_capacity(sl=self.sl, fd=self.fd)

        # Deal with both raft and pad foundations
        bearing_capacity = nf.bearing_capacity(self.fd.area, soil_q)
        weight_per_frame = sum(self.storey_masses) / (self.n_seismic_frames + self.n_gravity_frames) * self.g
        self.axial_load_ratio = bearing_capacity / self.total_weight

        self.theta_pseudo_up = nf.calculate_pseudo_uplift_angle(self.total_weight, self.fd.width, self.k_f_0,
                                                                self.axial_load_ratio, self.alpha, self.zeta)

    def to_table(self, table_name="df-table"):
        para = mo.output_to_table(self, olist="all")
        para += mo.output_to_table(self.fd)
        para += mo.output_to_table(self.sl)
        para += mo.output_to_table(self.hz)
        para = mo.add_table_ends(para,'latex', table_name, table_name)
        return para


class AssessedSFSIFrame(AssessedFrame):
    sl = sm.Soil()
    fd = sm.Foundation()
    total_weight = 0.0
    theta_f = 0.0
    axial_load_ratio = 0.0
    theta_pseudo_up = 0.0

    def __init__(self, fb, hz, sl, fd):
        super(AssessedSFSIFrame, self).__init__(fb, hz)  # run parent class initialiser function
        self.sl.__dict__.update(sl.__dict__)
        if fd.ftype == "raft":
            self.fd = sm.RaftFoundation()
        if fd.ftype == "pad":
            self.fd = sm.PadFoundation()
        self.fd.__dict__.update(fd.__dict__)
        self.k_f0_shear = stiffness.shear_stiffness(self.fd.width, self.fd.length, self.sl.g_mod, self.sl.poissons_ratio)

        if self.fd.ftype == "raft":
            self.k_f_0 = stiffness.rotational_stiffness(self.sl, self.fd)
            #self.k_f_0 = nf.rotational_stiffness(self.fd.width, self.fd.length, self.sl.g_mod, self.sl.poissons_ratio)
            self.alpha = 4.0
        else:
            self.k_f_0 = stiffness.rotational_stiffness(self.sl, self.fd)
            #self.k_f_0 = nf.rotational_stiffness(self.fd.width, self.fd.length, self.sl.g_mod, self.sl.poissons_ratio) / 2
            self.alpha = 3.0

        self.zeta = 1.5

    def static_values(self):
        self.total_weight = (sum(self.storey_masses) + self.fd.mass) * self.g
        self.soil_q = capacity.salgado_capacity(sl=self.sl, fd=self.fd)
        # Add new function to foundations, bearing_capacity_from_sfsimodels,
        # Deal with both raft and pad foundations
        bearing_capacity = nf.bearing_capacity(self.fd.area, self.soil_q)
        weight_per_frame = sum(self.storey_masses) / (self.n_seismic_frames + self.n_gravity_frames) * self.g
        self.axial_load_ratio = bearing_capacity / self.total_weight
        if self.axial_load_ratio < 1.0:
            raise DesignError("Static failure expected. Axial load ratio: %.3f" % self.axial_load_ratio)

        self.theta_pseudo_up = nf.calculate_pseudo_uplift_angle(self.total_weight, self.fd.width, self.k_f_0,
                                                                self.axial_load_ratio, self.alpha, self.zeta)

    def to_table(self, table_name="af-table"):
        para = mo.output_to_table(self, olist="all")
        para += mo.output_to_table(self.fd, prefix="Foundation ")
        para += mo.output_to_table(self.sl, prefix="Soil ")
        para += mo.output_to_table(self.hz, prefix="Hazard ")
        para = mo.add_table_ends(para, 'latex', table_name, table_name)
        return para
