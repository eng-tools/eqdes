__author__ = 'maximmillen'

from eqdes import models as dm


def test_model_inputs():
    p_models = [dm.Hazard(),
                dm.FrameBuilding(1, 1),
                dm.WallBuilding(1),
                dm.Soil(),
                dm.Concrete()]
    for model in p_models:
        for parameter in model.required_inputs:
            assert hasattr(model, parameter), parameter
