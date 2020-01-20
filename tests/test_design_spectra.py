from eqdes import design_spectra
import pytest
import numpy as np


@pytest.mark.parametrize('t,sc,expected', [
    (0.1, 'A', 2.35),
    (0.1, 'B', 2.35),
    (0.5, 'B', 1.6),
    (0.8, 'B', 1.12),
    (2.0, 'B', 0.53),
    (3.5, 'B', 0.26),
    (0.1, 'C', 2.93),
    (0.5, 'C', 2.00),
    (0.8, 'C', 1.41),
    (2.0, 'C', 0.66),
    (3.5, 'C', 0.32),
    (0.1, 'D', 3.00),
    (0.5, 'D', 3.00),
    (0.8, 'D', 2.29),
    (2.0, 'D', 1.07),
    (3.5, 'D', 0.52),
    (0.1, 'E', 3.00),
    (0.5, 'E', 3.00),
    (0.8, 'E', 3.00),
    (2.0, 'E', 1.66),
    (3.5, 'E', 0.81),
])
def test_get_ch_nzs1170(t, sc, expected):

    val = design_spectra.get_ch_nzs1170(t, sc)
    assert np.isclose(val, expected, atol=0.01), (t, sc)


def test_get_ch_nzs1170_for_static():
    t = 0.2
    method = 'static'
    sc = 'A'
    expected = 1.89
    val = design_spectra.get_ch_nzs1170(t, sc, method)
    assert np.isclose(val, expected)
    t = 0.2
    method = 'static'
    sc = 'C'
    expected = 2.36
    val = design_spectra.get_ch_nzs1170(t, sc, method)
    assert np.isclose(val, expected)
    # zeroth value
    t = 0.0
    method = 'static'
    sc = 'C'
    expected = 2.36
    val = design_spectra.get_ch_nzs1170(t, sc, method)
    assert np.isclose(val, expected), (val, expected)
    # outside scope
    t = 0.9
    method = 'static'
    sc = 'C'
    expected = 1.29
    val = design_spectra.get_ch_nzs1170(t, sc, method)
    assert np.isclose(val, expected, atol=0.02), (val, expected)
    # soil D
    t = 0.2
    method = 'static'
    sc = 'D'
    expected = 3.0
    val = design_spectra.get_ch_nzs1170(t, sc, method)
    assert np.isclose(val, expected)
    # multiple
    t = np.array([0.2, 0.7])
    method = 'static'
    sc = 'D'
    expected = [3.0, 2.53]
    val = design_spectra.get_ch_nzs1170(t, sc, method)
    assert np.isclose(val[0], expected[0]), (val[0], expected[0])
    assert np.isclose(val[1], expected[1], atol=0.01)


if __name__ == '__main__':
    test_get_ch_nzs1170_for_static()
