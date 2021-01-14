'''
Created on Feb 21, 2014

@author: mmi46
'''
import numpy as np


def getTeff(Z_Haz, R_Haz, N_Haz, Soil_type, Disp_eff, **kwargs):
    '''
    Returns the effective period of the structure
     using the values from NZS1170.5
    :param BO:
    :param Disp_eff:
    '''
    gravity = kwargs.get('gravity', 9.8)
    if Soil_type == 'C':
        T_C = 3.0
        D_C = (3.96 * Z_Haz * R_Haz *
               N_Haz / (2 * np.pi) ** 2 * gravity)
    elif Soil_type == 'D':
        T_C = 3.0
        D_C = (6.42 * Z_Haz * R_Haz *
               N_Haz / (2 * np.pi) ** 2 * gravity)
    elif Soil_type == 'E':
        T_C = 3.0
        D_C = (9.96 * Z_Haz * R_Haz * \
               N_Haz / (2 * np.pi) ** 2 * gravity)

    if Disp_eff > D_C:
        'Print displacement ductility not achieveable'
        T_eff = 0
    else:
        T_eff = T_C * Disp_eff / D_C

    return T_eff


def calculate_z(corner_disp, site_class_nzs):
    """
    calculates the value of Z for design for a given corner_disp.
    NOTE: corner_disp must be at 3s
    """
    if site_class_nzs == 'C':
        i = 0
    elif site_class_nzs == 'D':
        i = 1
    elif site_class_nzs == 'E':
        i = 2
    else:
        raise KeyError("site_class_nzs must be 'C', 'D' or 'E'.")

    nzs_val = [3.96, 6.42, 9.96]
    Z = corner_disp * 4 * np.pi ** 2 / nzs_val[i] / 9.8
    return Z


def get_ch_nzs1170(t, site_class_nzs, method='nith'):
    """
    Spectral shape factor based on Equations from NZS1170.5 Supp

    Eqs.
    Ch(0) = V0
    Ch(T) = V0 + V1(T/0.1) at T<T1
    Ch(T) = V2 at T<T2
    Ch(T) = V3a * (V3b / T) ** 0.75 at T<T3
    Ch(T) = V4 / T at T < T4
    Ch(T) = V5 / T ** 2
    if method == 'static':
        Ch(T) = VS
    """
    ch_dd = {
        "A": {
            "V0": 1.0,
            "T1": 0.1,
            "V1": 1.35,
            "T2": 0.3,
            "V2": 2.35,
            "T3": 1.5,
            "V3a": 1.6,
            "V3b": 0.5,
            "T4": 3.,
            "V4": 1.05,
            "V5": 3.15,
            "VS": 1.89
        },
        "B": {
            "V0": 1.0,
            "T1": 0.1,
            "V1": 1.35,
            "T2": 0.3,
            "V2": 2.35,
            "T3": 1.5,
            "V3a": 1.6,
            "V3b": 0.5,
            "T4": 3.,
            "V4": 1.05,
            "V5": 3.15,
            "VS": 1.89
        },
        "C": {
            "V0": 1.33,
            "T1": 0.1,
            "V1": 1.6,
            "T2": 0.3,
            "V2": 2.93,
            "T3": 1.5,
            "V3a": 2.0,
            "V3b": 0.5,
            "T4": 3.,
            "V4": 1.32,
            "V5": 3.96,
            "VS": 2.36
        },
        "D": {
            "V0": 1.12,
            "T1": 0.0,  # previous supplement =0.1s
            "V1": 1.88,
            "T2": 0.56,
            "V2": 3.0,
            "T3": 1.5,
            "V3a": 2.4,
            "V3b": 0.75,
            "T4": 3.,
            "V4": 2.14,
            "V5": 6.42,
            "VS": 3.0,
        },
        "E": {
            "V0": 1.12,
            "T1": 0.0,  # previous supplement =0.1s
            "V1": 1.88,
            "T2": 1.0,
            "V2": 3.0,
            "T3": 1.5,
            "V3a": 3.0,
            "V3b": 1.0,
            "T4": 3.,
            "V4": 3.32,
            "V5": 9.96,
            "VS": 3.0,
        }
    }

    cd = ch_dd[site_class_nzs]
    assert np.min(t) >= 0.0
    conds = [t <= cd['T1'],
             (t > cd['T1']) & (t <= cd['T2']),
             (t > cd['T2']) & (t <= cd['T3']),
             (t > cd['T3']) & (t <= cd['T4']),
             t > cd['T4']]

    funcs = [lambda t: cd['V0'] + cd['V1'] * (t/0.1),
             lambda t: cd['V2'],
             lambda t: cd['V3a'] * (cd['V3b'] / t) ** 0.75,
             lambda t: cd['V4'] / t,
             lambda t: cd['V5'] / t ** 2,
             ]
    vals = np.piecewise(t, conds, funcs)
    if method == 'static':
        vals = np.where(t < 0.4, cd['VS'], vals)
    return vals

sae = {
    "A": {
        "S": 1.0,
        "TB": 0.15,
        "TC": 0.4,
        "TD": 2.0
    },
    "B": {
        "S": 1.2,
        "TB": 0.15,
        "TC": 0.5,
        "TD": 2.0
    },
    "C": {
        "S": 1.15,
        "TB": 0.2,
        "TC": 0.6,
        "TD": 2.0
    },
    "D": {
        "S": 1.35,
        "TB": 0.2,
        "TC": 0.8,
        "TD": 2.0
    },
    "E": {
        "S": 1.4,
        "TB": 0.15,
        "TC": 0.5,
        "TD": 2.0
    }
}


def eurocode_sa(t, sc):
    """
    Eurocode site response spectrum Part 1 CL 3.2.2.2
    :param t:
    :param sc:
    :return:
    """
    eta = 1.0  # for 5% damping CL 3.2.2.2
    if 0 < t <= sae[sc]["TB"]:
        sa = sae[sc]["S"] * (1 + t / sae[sc]['TB'] * eta * 2.5 - 1)
    elif sae[sc]["TB"] < t <= sae[sc]["TC"]:
        sa = sae[sc]["S"] * eta * 2.5
    elif sae[sc]["TC"] < t <= sae[sc]["TD"]:
        sa = sae[sc]["S"] * eta * 2.5 * sae[sc]['TC'] / t
    elif sae[sc]["TD"] < t <= 4.0:
        sa = sae[sc]["S"] * eta * 2.5 * (sae[sc]['TC'] * sae[sc]['TD']) / t ** 2
    else:
        # beyond the scope of the standard
        raise NotImplementedError
    return sa


