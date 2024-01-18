import sfsimodels as sm


class Soil(sm.Soil):
    required_inputs = ["g_mod",
                      "phi",
                      "unit_weight"
                      ]
