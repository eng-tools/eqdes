import numpy as np
from eqdes import ei_finder


def test_EI_S1():
    n = 1
    mass = 10.0
    T_target = 1.2
    L = 1.0
    mdof = ei_finder.gen_target_system(n, mass, L, T_target, verbose=0)
    EI = mdof.ei
    print('EI: ', EI)
    T = 2 * np.pi * (mass * L ** 3 / (3 * EI)) ** 0.5
    print('T_target: ', T_target)
    print('T_obtained: ', T)
    assert np.isclose(T_target, T)


def test_K_terms():
    n = 2
    L = 1.5
    B = 2 * L
    # Solution from Chopra pg 364
    K_sol = np.array([[12, 3 * B, -12, 3 * B],
                      [3 * B, B ** 2, -3 * B, B ** 2 / 2],
                      [-12, -3 * B, 24, 0],
                      [3 * B, B ** 2 / 2, 0, 2 * B ** 2]])
    print('K_sol: ')
    print(K_sol)
    K = ei_finder.gen_k_terms(n, L)
    print('K_terms: ')
    print(K)
    rem =(K - K_sol)
    print('rem: ', rem)
    assert np.isclose(np.sum(abs(rem)), 0.0)

