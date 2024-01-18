import geofound
import numpy as np
import sfsimodels as sm
from sfsimodels import output as mo

from eqdes import dbd_tools as dt, nonlinear_foundation as nf
from eqdes.extensions.exceptions import DesignError
from eqdes.models.hazard import Hazard


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
        self.method = None
        self.inputs = [item for item in self.inputs]
        self.inputs += self._extra_class_variables
        self.beam_group_size = 2


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
        self.method = None
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
        self.fd.k_h_0 = geofound.stiffness.calc_shear_via_gazetas_1991(self.sl, self.fd, ip_axis=ip_axis)
        self.fd.k_m_0 = geofound.stiffness.calc_rotational_via_gazetas_1991(self.sl, self.fd, ip_axis=ip_axis)
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
        self.fd.n_ult = bearing_capacity

        self.theta_pseudo_up = nf.calculate_pseudo_uplift_angle(self.total_weight, self.fd.width, self.fd.k_m_0,
                                                                self.axial_load_ratio, self.alpha, self.zeta)


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
        self.fd_bearing_capacity = bearing_capacity
        if self.axial_load_ratio < 1.0:
            raise DesignError("Static failure expected. Axial load ratio: %.3f" % self.axial_load_ratio)

        self.theta_pseudo_up = nf.calculate_pseudo_uplift_angle(self.total_weight, self.fd.width, self.k_f_0,
                                                                self.axial_load_ratio, self.alpha, self.zeta)
