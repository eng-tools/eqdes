'''
Created on May 26, 2015

@author: maximmillen
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig


def gen_k_terms(n, ele_l, normed=False):
    if hasattr(ele_l, '__len__'):
        assert len(ele_l) == n
    else:
        ele_l = np.ones(n) * ele_l

    K = np.zeros((2 * n, 2 * n))
    nl = 1
    for i in range(n):
        if normed:
            nl = ele_l[i] ** 3
        kmat = np.array([12.0, 6.0 * ele_l[i], -12.0, 6.0 * ele_l[i]]) / nl
        tmat = np.array([6.0 * ele_l[i], 4.0 * ele_l[i] ** 2, -6.0 * ele_l[i], 2 * ele_l[i] ** 2]) / nl
        kmat2 = np.array([-12.0, -6.0 * ele_l[i], 12.0, -6.0 * ele_l[i]]) / nl
        tmat2 = np.array([6.0 * ele_l[i], 2 * ele_l[i] ** 2, -6.0 * ele_l[i], 4.0 * ele_l[i] ** 2]) / nl
        for j in range(4):
            try:
                K[2 * i][j + 2 * i] += kmat[j]
                K[2 * i + 1][j + 2 * i] += tmat[j]
            except IndexError:
                pass
            if i == 0:
                continue
            else:
                K[2 * i][j + 2 * (i - 1)] += kmat2[j]
                K[2 * i + 1][j + 2 * (i - 1)] += tmat2[j]

    return K


class MDOF(object):
    def __init__(self, ei, periods, phi, m_mat, k_mat):
        self.ei = ei
        self.periods = periods
        self.phi = phi
        self.m_mat = m_mat
        self.k_mat = k_mat


def gen_target_system(n, mass, heights, T_target, **kwargs):
    verbose = kwargs.get('verbose', 0)
    EI_trial = kwargs.get('EI_trial', 100.0)

    M = np.zeros((n, n))
    for i in range(n):
        M[i][i] = mass
    if hasattr(heights, '__len__'):
        assert len(heights) == n
    else:
        heights = heights * np.ones(n)
    K = gen_k_terms(n, heights, normed=True)
    if verbose:
        print('K_terms: \n', K)

    K = K * EI_trial
    K11 = np.zeros((n, n))
    K12 = np.zeros((n, n))
    K21 = np.zeros((n, n))
    K22 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K11[i][j] = K[2 * i][2 * j]
            K12[i][j] = K[2 * i][2 * j + 1]
            K21[i][j] = K[2 * i + 1][2 * j]
            K22[i][j] = K[2 * i + 1][2 * j + 1]

    K22_inv = np.array(np.matrix(K22).I)

    Kt = K22_inv.dot(K21)
    Kt2 = K12.dot(Kt)
    Kc = K11 - Kt2  # condensed matrix

    [L, phi] = eig(Kc, M)
    if verbose:
        print('K22_inv: \n', K22_inv)
        print('Kt: \n', Kt)
        print('Kc:\n', Kc)
        print('The mass matrix:')
        print(M)
        print('modes: \n', phi)
    #
    # #Append base node to the eigen vectors
    # phi=list(phi)
    # phi.insert(0,np.zeros((len(phi))))
    # M = np.insert(M,0,np.zeros((len(phi))))

    periods = []
    w = []
    for i in range(len(phi)):
        K_s = abs(np.transpose(phi[:, i]).dot(Kc).dot(phi[:, i]))
        M_s = abs(np.transpose(phi[:, i]).dot(M).dot(phi[:, i]))
        w.append(np.sqrt(K_s / M_s))
        if verbose:
            print('KS: ', K_s)

        periods.append(2 * np.pi / w[i])
    periods = np.array(periods)
    T_ratio = max(periods) / T_target
    EI = EI_trial * T_ratio ** 2
    periods = periods / T_ratio
    if verbose:
        print('Time periods:', periods)

    mdof = MDOF(EI, periods, phi, M, Kc)
    print('Not working - possibly because base is not fixed?')

    return mdof


def get_moment_demands(mdof, t_spectra, a_spectra):
        # spect_data = np.loadtxt(Apath + 'SelectedAverage.txt', delimiter=',')
        # T_spectra = spect_data[0]
        # D_spectra = spect_data[1]
        # A_spectra = spect_data[2]
        T_spectra = t_spectra
        A_spectra = a_spectra
        Ch = []
        
        for i in range(len(mdof.periods)):
            Ch.append(np.interp(mdof.periods[i], T_spectra, A_spectra))
            # Ch.append(Spectrum.Force(Time_period[0],Soil_type))

        print("Ch: ", Ch)

        # Calculating participating weight:
        r_modal_vector = np.ones((len(mdof.phi), 1))
        Weight_total = n * mass * 9.8
        percentage_weight = np.zeros((len(mdof.phi)))
        Base_shear = np.zeros((len(mdof.phi)))
        V = np.zeros((len(mdof.phi), len(mdof.phi)))
        delta = np.zeros((len(mdof.phi), len(mdof.phi)))
        gravity = 9.8
        print('phi')
        print(mdof.phi)
        for i in range(n):
            # phi_star = np.column(phi, -i - 1)
            phi_star_trans = mdof.phi[:, -i - 1]
            phi_star = np.array(np.matrix(phi_star_trans).T)
            # print phi_star
            Lstar_vector = phi_star_trans.dot(mdof.m_mat)
            Lstar = phi_star_trans.dot(mdof.m_mat).dot(r_modal_vector)
            print('Lstar:', Lstar)
            Mstar = phi_star_trans.dot(mdof.m_mat).dot(phi_star)
            participating_weight = Lstar ** 2 * gravity / Mstar
            percentage_weight[i] = participating_weight / Weight_total
            Base_shear[i] = Ch[i] * participating_weight * Weight_total
            V[i][:] = Base_shear[i] * Lstar_vector / Lstar
            delta_m = Ch[i] * gravity * mdof.periods[i] ** 2 / (4 * np.pi ** 2)
            phi_sq = mdof.phi[i] ** 2
            delta[i][:] = mdof.phi[i] * sum(mdof.phi[i].dot(mdof.m_mat) * delta_m) / sum(phi_sq.dot(mdof.m_mat))

        # NOTE:The mode shapes have been reversed to give the percentage weights as first column is first mode
        print('PW:', percentage_weight)

        # Maximum values form modal analysis:
        max_base_shear = sum(Base_shear)
        max_storey_forces = np.zeros(n)
        V_trans = np.array(np.matrix(V).T)
        M_is = np.zeros(n)

        heights = mdof.heights
        for i in range(len(V_trans)):
            max_storey_forces[i] = np.sqrt(sum(V_trans[i] ** 2))
            M_is[i] = np.sqrt(sum((V_trans[i] * heights[i]) ** 2))
        
        print('Base shear: \n', Base_shear)
        print('Maximum storey forces: \n', max_storey_forces)
        Moment = sum(M_is)
        return Moment


def get_static_deflections(mdof, forces):
    k_inv = np.array(np.matrix(mdof.k_mat).I)
    forces = np.array(forces)
    # l = k_inv * forces[:, np.newaxis]
    return np.cross(k_inv, forces)



if __name__ == '__main__':
    n = 2
    mass = 10.0
    T_target = 2.0
    L = 1.0
    mdof = gen_target_system(n, mass, L, T_target, verbose=1, EI_trial=7818.0)
    print(mdof.ei)
    print(mdof.periods)
    ds = get_static_deflections(mdof, [100, 100])
    print('ds: ', ds)
    pass
