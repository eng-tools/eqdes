__author__ = 'maximmillen'

import eqdes.models.frame_building
import eqdes.models.hazard
import eqdes.models.material
import eqdes.models.soil
import eqdes.models.wall_building
from eqdes import models as dm
from tests import conftest


def test_model_inputs():
    p_models = [eqdes.models.hazard.Hazard(),
                eqdes.models.frame_building.FrameBuilding(1, 1),
                eqdes.models.wall_building.WallBuilding(1),
                eqdes.models.soil.Soil(),
                eqdes.models.material.ReinforcedConcrete()]
    for model in p_models:
        for parameter in model.required_inputs:
            assert hasattr(model, parameter), parameter


def test_initialse_designed_walls():
    wb = conftest.wb_test
    hz = conftest.hz_test
    sl = conftest.sl_test
    fd = conftest.fd_test

    dw = eqdes.models.wall_building.DispBasedRCWall(wb, sl, fd)
    dw.design_drift = 0.025
    assert dw.sl.unit_dry_weight == sl.unit_dry_weight


if __name__ == '__main__':
    test_initialse_designed_walls()