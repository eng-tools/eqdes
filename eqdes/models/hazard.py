import sfsimodels as sm


class Hazard(sm.SeismicHazard):
    required_inputs = ["corner_disp",
                      "corner_period",
                      "z_factor",
                      "r_factor"
                      ]
