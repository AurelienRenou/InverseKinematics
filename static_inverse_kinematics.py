from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import biorbd
import bioviz
from scripts.utils import get_range_q
from scripts.load_experimental_data import C3dData
import scipy


class StaticInverseKinematics:
    """
    The class for generate inverse kinemacs from c3d files

    Attributes:
    ----------
    biorbd_model_path: str
        The biorbd model path
    c3d_path_file: str
        The c3d file path
    biorbd_model: biorbd.Model
        The biorbd model loaded
    c3d_data: C3dData
        The Data from c3d file
    list_model_markers: list[str]
        The list of markers' name
    xp_markers: np.array
        The position of the markers from the c3d
    nbQ: int
        The number of dof in the model
    nb_frames: int
        The number of frame in the c3d
    nb_markers: int
        The number of markers in the model
    model_markers: np.array

    q: np.array
        The values of the q to makes markers' position from c3d and model match
    bounds_min: np.array
        The min range of the model Q
    bounds_max: np.array
        The max range of the model Q

    Methods
    -------
    _marker_diff(self, q: np.ndarray, xp_markers: np.ndarray)
        Compute the difference between the marker position in the model and the position in the data.
    _marker_jac(self, q: np.ndarray, xp_markers: np.ndarray)
        Generate the Jacobian matrix for each frame.
    optimize(self, n_frame: int, method: str, bounds: tuple() = None)
        Uses least_square function to minimize the difference between markers' positions of model and c3d.
    solve(self, method: str = "lm", full: bool = False)
        Solve the inverse kinematics by using least square methode from scipy.
    animate(self)
        Animate the result of solve with bioviz.

    """

    def __init__(
            self,
            biorbd_model_path: str,
            c3d_path_file: str,
    ):
        self.biorbd_model_path = biorbd_model_path
        self.c3d_path_file = c3d_path_file

        self.biorbd_model = biorbd.Model(self.biorbd_model_path)
        self.c3d_data = C3dData(self.c3d_path_file, self.biorbd_model)

        self.list_model_markers = [
            self.biorbd_model.markerNames()[i].to_string() for i in range(len(self.biorbd_model.markerNames()))
        ]
        self.xp_markers = self.c3d_data.trajectories
        self.nbQ = self.biorbd_model.nbQ()
        self.nb_frames = self.c3d_data.nb_frames
        self.nb_markers = self.biorbd_model.nbMarkers()

        self.model_markers = np.zeros((3, self.nb_markers))  # We initialize this attributes
        self.q = np.zeros((self.biorbd_model.nbQ(), self.nb_frames))
        self.bounds_min, self.bounds_max = np.squeeze(get_range_q(self.biorbd_model))

    def _marker_diff(self, q: np.ndarray, xp_markers: np.ndarray):
        """
        Compute the difference between the marker position in the model and the position in the data
        Arguments:
        ---------
        q: np.ndarray
            The q values
        xp_markers: np.ndarray
            The position of the markers from the c3d during a certain frame
        """
        mat_pos_markers = self.biorbd_model.technicalMarkers(q)
        vect_pos_markers = np.zeros(3 * self.nb_markers)

        for m in range(self.nb_markers):
            vect_pos_markers[m * 3: (m + 1) * 3] = mat_pos_markers[m].to_array()

        # return the vector of the squared differences between the model and the data
        return vect_pos_markers - np.reshape(xp_markers.T, (self.nb_markers * 3,))

    def _marker_jac(self, q: np.ndarray, xp_markers: np.ndarray):
        """
        Generate the Jacobian matrix for each frame.
        Arguments:
        ---------
        q: np.ndarray
            The q values
        xp_markers: np.ndarray
            The position of the markers from the c3d during a certain frame
        """
        mat_jac = self.biorbd_model.technicalMarkersJacobian(q)

        jac = np.zeros((3 * self.nb_markers, self.nbQ))
        for m in range(self.nb_markers):
            jac[m * 3: (m + 1) * 3, :] = mat_jac[m].to_array()

        return jac

    def optimize(self, n_frame: int, method: str, bounds: tuple() = None):
        print(f" ****   Frame {n_frame}  ****")
        x0 = self.q[:, n_frame - 1]
        if bounds is None:
            sol = scipy.optimize.least_squares(
                    fun=self._marker_diff,
                    args=([self.xp_markers[:, :, n_frame]]),
                    jac=self._marker_jac,
                    x0=x0,
                    method=method,
                    xtol=1e-6,
                    tr_options=dict(disp=False),
                )
        else:
            sol = scipy.optimize.least_squares(
                fun=self._marker_diff,
                args=([self.xp_markers[:, :, n_frame]]),
                bounds=bounds,
                jac=self._marker_jac,
                x0=x0,
                method=method,
                xtol=1e-6,
                tr_options=dict(disp=False),
            )
        self.q[:, n_frame] = sol.x

    def solve(self, method: str = "lm", full: bool = False):
        """
        Solve the inverse kinematics by using least square methode from scipy
        Parameters:
        ----------
        method: str
            The method used by least_square

        """

        if full:
            if method == "trf":
                for ii in range(0, self.nb_frames):
                    self.optimize(ii, method, (self.bounds_min, self.bounds_max))
            elif method == "lm":
                for ii in range(0, self.nb_frames):
                    self.optimize(ii, method)
            else:
                raise ValueError('This method is not implemented please use "trf" or "lm" as argument')

        else:
            self.optimize(0, "trf")

            if method == "trf" or method == "lm":
                for ii in range(1, self.nb_frames):
                    self.optimize(ii, method)
            else:
                raise ValueError('This method is not implemented please use "trf" or "lm" as argument')

        print("Inverse Kinematics done for all frames")

    def animate(self):
        """
        Animate the result of solve with bioviz.
        """
        b = bioviz.Viz(loaded_model=self.biorbd_model, show_muscles=False)
        b.load_experimental_markers(self.xp_markers)
        b.load_movement(self.q)
        b.exec()
