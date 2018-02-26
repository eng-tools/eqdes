
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    To check equality of floats
    From https://www.python.org/dev/peps/pep-0485/#proposed-implementation
    :param a:
    :param b:
    :param rel_tol:
    :param abs_tol:
    :return:
    """
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
