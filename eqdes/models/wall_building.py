import geofound
import numpy as np
import sfsimodels as sm
from sfsimodels import output as mo

from eqdes import dbd_tools as dt, nonlinear_foundation as nf
from eqdes.models.hazard import Hazard


class WallBuilding(sm.WallBuilding):
    required_inputs = [
        'floor_length',
        'floor_width',
        'interstorey_heights',
        'n_storeys',
        "n_walls",
        "wall_depth"
    ]


class DesignedRCWall(sm.SingleWall):
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

    def __init__(self, sw, hz, verbose=0):
        super(DesignedRCWall, self).__init__(sw.n_storeys)  # run parent class initialiser function
        self.__dict__.update(sw.__dict__)
        self.hz.__dict__.update(hz.__dict__)
        self.verbose = verbose
        assert sw.material.type == 'rc_material'
        self.concrete = sw.material
        self.phi_material = 0.072 / self.wall_depth  # Eq 6.10b
        self.fye = 1.1 * self.concrete.fy
        self.epsilon_y = self.fye / self.concrete.e_mod_steel
        self.fu = 1.40 * self.fye  # Assumed, see pg 141
        self.storey_mass = self.storey_masses
        self.storey_forces = np.zeros((1, len(self.storey_masses)))
        self.hm_factor = dt.cal_higher_mode_factor(self.n_storeys, btype="wall")
        self._extra_class_variables = ["method"]
        self.method = None
        self.inputs += self._extra_class_variables

    def static_dbd_values(self):
        pass
        # Material strain limits check


class DesignedSFSIRCWall(DesignedRCWall):

    sl = sm.Soil()
    fd = sm.RaftFoundation()
    total_weight = 0.0
    theta_f = 0.0
    axial_load_ratio = 0.0
    bearing_capacity = 0.0
    theta_pseudo_up = 0.0
    k_f0_shear = 0.0
    alpha = 0.0
    zeta = 0.0
    k_m_0 = 0.0

    def __init__(self, wb, hz, sl, fd):
        super(DesignedSFSIRCWall, self).__init__(wb, hz)  # run parent class initialiser function
        self.sl.__dict__.update(sl.__dict__)
        self.fd.__dict__.update(fd.__dict__)

    def static_values(self):
        self.total_weight = sum(self.storey_n_loads) + self.fd.mass * self.g
        soil_q = geofound.capacity_salgado_2008(sl=self.sl, fd=self.fd)

        # Deal with both raft and pad foundations
        self.fd.n_ult = self.fd.area * soil_q
        self.axial_load_ratio = self.bearing_capacity / self.total_weight

        self.fd.k_h_0 = geofound.stiffness.calc_shear_via_gazetas_1991(self.sl, self.fd, ip_axis='length')

        if self.fd.ftype == "raft":
            self.alpha = 4.0
        else:
            self.alpha = 3.0
        self.fd.k_m_0 = geofound.stiffness.calc_rotational_via_gazetas_1991(self.sl, self.fd, ip_axis='length')
        self.zeta = 1.5
        self.theta_pseudo_up = nf.calculate_pseudo_uplift_angle(self.total_weight, self.fd.width, self.k_m_0,
                                                                self.axial_load_ratio, self.alpha, self.zeta)


def design_wall_to_table(dw, table_name="df-table"):
    para = mo.output_to_table(dw, olist="all")
    para += mo.output_to_table(dw.fd)
    para += mo.output_to_table(dw.sl)
    para += mo.output_to_table(dw.hz)
    para = mo.add_table_ends(para, 'latex', table_name, table_name)
    return para


class AssessedRCWall(sm.SingleWall):
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

    def __init__(self, sw, hz, verbose=0):
        super(AssessedRCWall, self).__init__(sw.n_storeys)  # run parent class initialiser function
        self.__dict__.update(sw.__dict__)
        self.hz.__dict__.update(hz.__dict__)
        self.verbose = verbose
        assert sw.material.type == 'rc_material'
        self.concrete = sw.material
        self.phi_material = 0.072 / self.wall_depth  # Eq 6.10b
        self.fye = 1.1 * self.concrete.fy
        self.epsilon_y = self.fye / self.concrete.e_mod_steel
        self.fu = 1.40 * self.fye  # Assumed, see pg 141
        self.storey_mass = self.storey_masses
        self.storey_forces = np.zeros((1, len(self.storey_masses)))
        self.hm_factor = dt.cal_higher_mode_factor(self.n_storeys, btype="wall")
        self._extra_class_variables = ["method"]
        self.method = None
        self.inputs += self._extra_class_variables

