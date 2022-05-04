from ezc3d import c3d
import numpy as np
from scipy.interpolate import interp1d
import biorbd


class C3dData:
    """
    The base class for managing c3d file

    Attributes
    ----------
    c3d: ezc3d.c3d
        The c3d file
    marker_names: list
        The list of all marker names in the biorbd model.
    trajectories: ndarray
        The position of the markers
    Methods
    -------
    get_marker_trajectories(loaded_c3d: c3d, marker_names: list) -> np.ndarray
        Get markers trajectories
    get_indices(self)
        Get the indices of start and end
    get_final_time(self)
        Get the final time of c3d
    """

    def __init__(self, file_path: str, biorbd_model: biorbd.Model):
        self.c3d = c3d(file_path)
        self.marker_names = [biorbd_model.markerNames()[i].to_string() for i in range(len(biorbd_model.markerNames()))]
        self.trajectories = self.get_marker_trajectories()
        self.nb_frames = self.c3d["parameters"]["POINT"]["FRAMES"]["value"][0]

    def get_marker_trajectories(self, marker_names: list = None) -> np.ndarray:
        """
        get markers trajectories
        """
        marker_names = self.marker_names if marker_names is None else marker_names
        # LOAD C3D FILE
        points = self.c3d["data"]["points"]
        labels_markers = self.c3d["parameters"]["POINT"]["LABELS"]["value"]

        # GET THE MARKERS POSITION (X, Y, Z) AT EACH POINT
        markers = np.zeros((3, len(marker_names), len(points[0, 0, :])))

        for i, name in enumerate(marker_names):
            markers[:, i, :] = points[:3, labels_markers.index(name), :] * 1e-3
        return markers

    def get_indices(self):
        idx_start = 0 + 1
        idx_stop = len(self.trajectories[0, 0, :])
        return [idx_start, idx_stop]

    def get_final_time(self):
        """
        find phase duration
        """
        # todo: plz shrink the function
        freq = self.c3d["parameters"]["POINT"]["RATE"]["value"][0]
        index = self.get_indices()
        return [(1 / freq * (index[i + 1] - index[i] + 1)) for i in range(len(index) - 1)][0]
