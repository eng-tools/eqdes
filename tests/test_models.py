__author__ = 'maximmillen'

from eqdes import models as dm
from tests import conftest


def test_model_inputs():
    p_models = [dm.Hazard(),
                dm.FrameBuilding(1, 1),
                dm.WallBuilding(1),
                dm.Soil(),
                dm.Concrete()]
    for model in p_models:
        for parameter in model.required_inputs:
            assert hasattr(model, parameter), parameter


def test_initialse_designed_walls():
    wb = conftest.wb_test
    hz = conftest.hz_test
    sl = conftest.sl_test
    fd = conftest.fd_test

    dw = dm.DesignedSFSIWall(wb, hz, sl, fd)
    dw.design_drift = 0.025
    assert dw.sl.unit_dry_weight == sl.unit_dry_weight


if __name__ == '__main__':
    test_initialse_designed_walls()