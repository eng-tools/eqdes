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


def calculate_Z(D_C, Soil_type):
    '''
    calculates the value of Z for design for a given D_C.
    NOTE: D_C must be at 3s
    '''
    if Soil_type == 'C': i = 0
    if Soil_type == 'D': i = 1
    if Soil_type == 'E': i = 2

    NZS_val = [3.96, 6.42, 9.96]
    Z = D_C * 4 * np.pi ** 2 / NZS_val[i] / 9.8
    return Z


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
