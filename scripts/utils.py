import biorbd
import numpy as np


def get_range_q(biorbd_model: biorbd.Model):
    """


    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model

    Returns
    -------
    range_q
        The range of the q for each dof
    """
    ranges = np.zeros((biorbd_model.nbQ(), 2))
    for dof_nb in range(biorbd_model.nbDof()):
        seg_id, dof_id = get_segment_and_dof_id_from_global_dof(biorbd_model, dof_nb)
        range_min = biorbd_model.segment(seg_id).QRanges()[dof_id].min()
        range_max = biorbd_model.segment(seg_id).QRanges()[dof_id].max()
        ranges[dof_nb, 0] = range_min
        ranges[dof_nb, 1] = range_max
    return ranges


def get_range_max_q(biorbd_model: biorbd.Model):
    """


    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model

    Returns
    -------
    range_q
        The range of the q for each dof
    """
    ranges = np.zeros((biorbd_model.nbQ(), 1))
    for dof_nb in range(biorbd_model.nbDof()):
        seg_id, dof_id = get_segment_and_dof_id_from_global_dof(biorbd_model, dof_nb)
        range_max = biorbd_model.segment(seg_id).QRanges()[dof_id].max()
        ranges[dof_nb] = range_max
    return ranges


def get_range_min_q(biorbd_model: biorbd.Model):
    """


    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model

    Returns
    -------
    range_q
        The range of the q for each dof
    """
    ranges = np.zeros((biorbd_model.nbQ(), 1))
    for dof_nb in range(biorbd_model.nbDof()):
        seg_id, dof_id = get_segment_and_dof_id_from_global_dof(biorbd_model, dof_nb)
        range_min = biorbd_model.segment(seg_id).QRanges()[dof_id].min()
        ranges[dof_nb] = range_min
    return ranges


def get_segment_and_dof_id_from_global_dof(biorbd_model: biorbd.Model, global_dof: int):
    """
    Allow the users to get the segment id which correspond to a dof of the model and the id of this dof in the segment

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    global_dof: int
        The global id of the dof in the model

    Returns
    -------
    seg_id: int
        The id of the segment which correspond to the dof
    count_dof: int
         The dof id in this segment
    """
    for j, seg in enumerate(biorbd_model.segments()):
        complete_seg_name = biorbd_model.nameDof()[global_dof].to_string()  # We get "Segment_Name_DofName"
        seg_name = complete_seg_name.replace("_RotX", "")  # We remove "_DofName"
        seg_name = seg_name.replace("_RotY", "")
        seg_name = seg_name.replace("_RotZ", "")
        seg_name = seg_name.replace("_TransX", "")
        seg_name = seg_name.replace("_TransY", "")
        seg_name = seg_name.replace("_TransZ", "")

        if seg.name().to_string() == seg_name:
            seg_name_new = seg_name
            seg_id = j

    dof = biorbd_model.nameDof()[global_dof].to_string().replace(seg_name_new, "")
    dof = dof.replace("_", "")  # we remove the _ "_DofName"
    count_dof = 0
    while biorbd_model.segment(seg_id).nameDof(count_dof).to_string() != dof:
        count_dof += 1

    return seg_id, count_dof
