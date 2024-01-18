import sfsimodels as sm


class RaftFoundation(sm.RaftFoundation):
    k_m_0 = None
    k_h_0 = None
    n_ult = None
    alpha = None
    zeta = None
    theta_pseudo_up = None
    axial_load_ratio = None
    required_inputs = [
        "width",
        "length",
        "depth",
        "height",
        "density",
        "i_ww",
        "i_ll"
    ]
    theta_p = None
    theta_r = None


class PadFoundation(sm.PadFoundation):
    required_inputs = [
        "width",
        "length",
        "depth",
        "height",
        "density",
        "i_ww",
        "i_ll",
        "n_pads_l",
        "n_pads_w",
        "pad_length",
        "pad_width",
    ]
