from eqdes import dbd
from eqdes import models as em
import numpy as np
import eqdes

def create():
    hz = em.Hazard()
    hz.z_factor = 0.3  # Hazard factor
    hz.r_factor = 1.0  # Return period factor
    hz.n_factor = 1.0  # Near-fault factor
    hz.magnitude = 7.5  # Magnitude of earthquake
    hz.corner_period = 4.0  # s
    hz.corner_acc_factor = 0.55

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

    df = dbd.design_rc_frame(fb, hz, 0.02)
    ps = eqdes.moment_equilibrium.assess(df, df.storey_forces)
    moment_beams_cl = ps[0]
    moment_column_bases = ps[1]
    axial_seismic = ps[2]
    print(moment_beams_cl)
    eqdes.moment_equilibrium.set_beam_face_moments_from_centreline_demands(df, moment_beams_cl)
    eqdes.moment_equilibrium.set_column_base_moments_from_demands(df, moment_column_bases)
    otm_max = eqdes.moment_equilibrium.calc_otm_capacity(df)
    approx_otm = np.sum(df.storey_forces * df.heights)
    print(otm_max, approx_otm)


if __name__ == '__main__':
    create()