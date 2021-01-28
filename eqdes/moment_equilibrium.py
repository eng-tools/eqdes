import numpy as np
from eqdes import fns
from eqdes.extensions.exceptions import DesignError


def assess(fb, storey_forces, mom_ratio=0.6, verbose=0):
    """
    Distribute the applied loads to a frame structure

    Parameters
    ----------
    fb: FrameBuilding object
    mom_ratio: float
        ratio of overturning moment that is resisted by column base hinges
    verbose:
        level of verbosity

    Returns
    -------
    [beam moments, column base moments, seismic axial loads in exterior columns]
    """
    if hasattr(fb, 'column_depth') and np.std(fb.column_depth) > 1e-2:
        print('Does not work with odd column depths')
        print(fb.column_depth)
        raise NotImplementedError

    mom_running = 0
    mom_storey = np.zeros(fb.n_storeys)
    v_storey = np.zeros(fb.n_storeys)
    for i in range(fb.n_storeys):

        if i == 0:
            v_storey[-1 - i] = storey_forces[-1 - i]
        else:
            v_storey[-1 - i] = v_storey[-i] + storey_forces[-1 - i]
        mom_storey[-1 - i] = (v_storey[-1 - i] * fb.interstorey_heights[-1 - i] + mom_running)
        mom_running = mom_storey[-1 - i]

    cumulative_total_shear = sum(v_storey)
    base_shear = sum(storey_forces)

    # Column_base_moment_total=mom_storey[0]*Base_moment_contribution
    column_base_moment_total = base_shear * mom_ratio * fb.interstorey_heights[0]
    moment_column_bases = (column_base_moment_total / fb.n_bays * np.ones((fb.n_bays + 1)))
    moment_column_bases[0] = moment_column_bases[0] / 2
    moment_column_bases[-1] = moment_column_bases[-1] / 2

    axial_seismic = (mom_storey[0] - column_base_moment_total) / sum(fb.bay_lengths)
    if verbose == 1:
        print('Storey shear forces: \n', v_storey)
        print('Moments', mom_storey)
        print('Total overturning moment: ', mom_storey[0])
        print('column_base_moment_total: ', column_base_moment_total)
        print('Seismic axial: ', axial_seismic)

    beam_shear_force = np.zeros(fb.n_storeys)

    for i in range(int(np.ceil(fb.n_storeys / fb.beam_group_size))):
        group_shear = np.average(
            v_storey[i * fb.beam_group_size:(i + 1) * fb.beam_group_size]) / cumulative_total_shear * axial_seismic
        if verbose > 1:
            print('group shear: ', group_shear)
        for j in range(int(fb.beam_group_size)):
            if i * fb.beam_group_size + j == fb.n_storeys:
                if verbose:
                    print('odd number of storeys')
                break
            beam_shear_force[i * fb.beam_group_size + j] = group_shear

    if (sum(beam_shear_force) - axial_seismic) / axial_seismic > 1e-2:
        raise DesignError('Beam shear force incorrect!')

    moment_beams_cl = np.zeros((fb.n_storeys, fb.n_bays, 2))
    if fb.bay_lengths[0] != fb.bay_lengths[-1]:
        print('Design not developed for irregular frames!')
    else:
        for i in range(fb.n_storeys):
            for j in range(fb.n_bays):
                moment_beams_cl[i][j] = beam_shear_force[i] * fb.bay_lengths[0] * 0.5 * np.array([1, -1])

    if verbose > 0:
        print('Seismic beam shear force: \n', beam_shear_force)
        print('Beam centreline moments: \n', moment_beams_cl)
    
    return moment_beams_cl, moment_column_bases, axial_seismic


def set_beam_face_moments_from_centreline_demands(df, moment_beams_cl, centre_sect=False):  # TODO: currently beam moment are centreline!
    import sfsimodels as sm
    assert isinstance(df, sm.FrameBuilding)
    nsects = 2
    es = 1
    if centre_sect:
        nsects = 3
        es = 2
    for ns in range(df.n_storeys):
        beams = df.get_beams_at_storey(ns)
        for beam in beams:
            existing_sects = beam.sections
            if len(existing_sects) != nsects:
                beam.sections = [existing_sects[0].deepcopy() for x in range(nsects)]
        # for nb in range(df.n_bays):
    # Assumes symmetric
    df.set_beam_prop('mom_cap_p', moment_beams_cl[:, :, :1], sections=[0], repeat='none')
    df.set_beam_prop('mom_cap_p', -moment_beams_cl[:, :, 1:], sections=[es], repeat='none')
    # Assumes symmetric
    df.set_beam_prop('mom_cap_n', -moment_beams_cl[:, :, :1], sections=[0], repeat='none')
    df.set_beam_prop('mom_cap_n', moment_beams_cl[:, :, 1:], sections=[es], repeat='none')

def set_beam_dimensions_from_yield_moment_and_strain(df, eps_yield, e_mod, f_crack=0.4, d_fact=2.1):
    """d_fact=2.1 from Priestley book"""
    import sfsimodels as sm
    assert isinstance(df, sm.FrameBuilding)
    for ns in range(df.n_storeys):
        beams = df.get_beams_at_storey(ns)
        for beam in beams:
            sects = beam.sections
            widths = []
            for sect in sects:
                phi_y = d_fact * eps_yield / sect.depth
                if hasattr(sect, 'mom_cap_p'):
                    mom_cap = (sect.mom_cap_p - sect.mom_cap_n) / 2
                    i_cracked = mom_cap / phi_y / e_mod
                    i_full = i_cracked / f_crack
                    widths.append(i_full / sect.depth ** 3 * 12)
            width = np.mean(widths)
            beam.set_section_prop('width', width, sections=list(range(len(sects))))
            beam.set_section_prop('e_mod', e_mod, sections=list(range(len(sects))))
            beam.set_section_prop('f_crack', f_crack, sections=list(range(len(sects))))


def set_column_base_moments_from_demands(df, moment_column_bases):
    import sfsimodels as sm
    assert isinstance(df, sm.FrameBuilding)

    columns = df.columns[0]
    for i, column in enumerate(columns):
        existing_sects = column.sections
        if len(existing_sects) != 2:
            column.sections = [existing_sects[0].deepcopy(),  # TODO: should be RCColumnSection
                           existing_sects[0].deepcopy()]
        column.sections[0].mom_cap = moment_column_bases[i]
        column.sections[0].inputs += ['mom_cap']


def calc_otm_capacity(df, mcbs=None):  # and account for tie beams !!! and m_foots=None, h_foot=0
    if mcbs is None:
        mcbs = fns.get_column_base_moments(df)
    m_f_beams = fns.get_beam_face_moments(df, signs=('p', 'n'))
    v_beams = -np.diff(m_f_beams[:, :]).reshape((df.n_storeys, df.n_bays)) / df.bay_lengths[np.newaxis, :]
    # Assume contra-flexure at centre of beam
    a_loads = np.zeros((df.n_storeys, df.get_n_cols()))
    a_loads[:, :-1] += v_beams
    a_loads[:, 1:] += -v_beams
    col_axial_loads = np.sum(a_loads, axis=0)
    x_cols = df.get_column_positions()
    otm_beams = -np.sum(x_cols * col_axial_loads)
    otm_total = otm_beams + np.sum(mcbs)
    return otm_total


def calc_seismic_axial_load_limit(df):  # and account for tie beams !!! and m_foots=None, h_foot=0
    m_col_bases = fns.get_column_base_moments(df)
    m_f_beams = fns.get_beam_face_moments(df, signs=('p', 'n'))
    v_beams = -np.diff(m_f_beams[:, :]).reshape((df.n_storeys, df.n_bays)) / df.bay_lengths[np.newaxis, :]
    # Assume contra-flexure at centre of beam
    a_loads = np.zeros((df.n_storeys, df.get_n_cols()))
    a_loads[:, :-1] += v_beams
    a_loads[:, 1:] += -v_beams
    col_axial_loads = np.sum(a_loads, axis=0)
    return col_axial_loads