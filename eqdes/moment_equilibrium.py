import numpy as np

from eqdes.extensions.exceptions import DesignError


def assess(fb, storey_forces, mom_ratio=0.6, verbose=0):
    """
    Distribute the applied loads to a frame structure
    :param fb: FrameBuilding object
    :param mom_ratio: ratio of overturning moment that is resisted by column base hinges
    :param verbose: level of verbosity
    :return: beam moments, column base moments, seismic axial loads in exterior columns
    """
    if hasattr(fb, 'column_depth') and np.std(fb.column_depth) > 1e-2:
        print('Does not work with odd column depths')
        print(fb.column_depth)
        raise NotImplementedError

    height = np.zeros(len(fb.storey_heights))
    for i in range(len(fb.storey_heights)):
        if i == 0:
            height[i] = fb.storey_heights[i]
        else:
            height[i] = height[i - 1] + fb.storey_heights[i]

    mom_running = 0
    mom_storey = np.zeros(len(fb.storey_heights))
    v_storey = np.zeros(len(fb.storey_heights))
    for i in range(len(fb.storey_heights)):

        if i == 0:
            v_storey[-1 - i] = storey_forces[-1 - i]
        else:
            v_storey[-1 - i] = v_storey[-i] + storey_forces[-1 - i]
        mom_storey[-1 - i] = (v_storey[-1 - i] *
            fb.storey_heights[-1 - i] + mom_running)
        mom_running = mom_storey[-1 - i]

    cumulative_total_shear = sum(v_storey)
    base_shear = sum(storey_forces)

    # Column_base_moment_total=mom_storey[0]*Base_moment_contribution
    column_base_moment_total = base_shear * mom_ratio * fb.storey_heights[0]
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



