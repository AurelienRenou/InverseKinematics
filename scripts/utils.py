import biorbd
import numpy as np


def get_range_q(biorbd_model: biorbd.Model) -> tuple[np.ndarray, np.ndarray]:
    """
    Give the ranges of the model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model

    Returns
    -------
    q_range_min, q_range_max: tuple[np.ndarray, np.ndarray]
        The range min and max of the q for each dof
    """
    q_range_max = []
    q_range_min = []
    for i in range(biorbd_model.nbSegment()):
        segment = biorbd_model.segment(i)
        q_range_max += [q_range.max() for q_range in segment.QRanges()]
        q_range_min += [q_range.min() for q_range in segment.QRanges()]
    return np.array(q_range_min), np.array(q_range_max)


