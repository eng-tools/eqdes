import numpy as np


def get_column_base_moments(fb):
    cols = fb.columns[0, :]
    return np.array([col.sections[0].mom_cap for col in cols])


def get_beam_face_moments(fb, signs=('p', 'p')):
    m_face = [[] for i in range(fb.n_storeys)]
    beams = fb.beams
    for ns in range(fb.n_storeys):
        for nb in range(fb.n_bays):
            m_face[ns].append([getattr(beams[ns][nb].sections[0], f'mom_cap_{signs[0]}'),
                               getattr(beams[ns][nb].sections[-1], f'mom_cap_{signs[1]}')])

    return np.array(m_face)
