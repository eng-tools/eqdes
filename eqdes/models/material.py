import sfsimodels as sm


class ReinforcedConcrete(sm.materials.ReinforcedConcreteMaterial):
    required_inputs = [
            'fy',
            'e_mod_steel'
    ]
