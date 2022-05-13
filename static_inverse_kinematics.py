import numpy as np
from scipy.optimize import minimize
import biorbd
import bioviz
from utils import get_range_q, get_unit_division_factor
import scipy
from ezc3d import c3d


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
    c3d: ezc3d.c3d
        The c3d loaded
    marker_names: list[str]
        The list of markers' name
    xp_markers: np.array
        The position of the markers from the c3d
    nb_q: int
        The number of dof in the model
    nb_frames: int
        The number of frame in the c3d
    nb_markers: int
        The number of markers in the model
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
        self.c3d = c3d(self.c3d_path_file)

        self.marker_names = [
            self.biorbd_model.markerNames()[i].to_string() for i in range(len(self.biorbd_model.markerNames()))
        ]
        self.xp_markers = self.get_marker_trajectories()
        self.nb_q = self.biorbd_model.nbQ()
        self.nb_frames = self.c3d["parameters"]["POINT"]["FRAMES"]["value"][0]
        self.nb_markers = self.biorbd_model.nbMarkers()

        self.q = np.zeros((self.nb_q, self.nb_frames))
        self.bounds = get_range_q(self.biorbd_model)

    def get_marker_trajectories(self) -> np.ndarray:
        """
        get markers trajectories
        """

        # LOAD C3D FILE
        points = self.c3d["data"]["points"]
        labels_markers = self.c3d["parameters"]["POINT"]["LABELS"]["value"]

        # GET THE MARKERS POSITION (X, Y, Z) AT EACH POINT
        markers = np.zeros((3, len(self.marker_names), len(points[0, 0, :])))

        for i, name in enumerate(self.marker_names):
            markers[:, i, :] = points[:3, labels_markers.index(name), :] / get_unit_division_factor(self.c3d)
        return markers

    def _marker_diff(self, q: np.ndarray, xp_markers: np.ndarray):
        """
        Compute the difference between the marker position in the model and the position in the data

        Parameters:
        -----------
        q: np.ndarray
            The q values
        xp_markers: np.ndarray
            The position of the markers from the c3d during a certain frame

        Return:
        ------
            The difference vector between markers' position in the model and in the c3d
        """
        mat_pos_markers = self.biorbd_model.technicalMarkers(q)
        vect_pos_markers = np.zeros(3 * self.nb_markers)

        for m in range(self.nb_markers):
            vect_pos_markers[m * 3 : (m + 1) * 3] = mat_pos_markers[m].to_array()

        # return the vector of the squared differences between the model and the data
        return vect_pos_markers - np.reshape(xp_markers.T, (self.nb_markers * 3,))

    def _marker_jac(self, q: np.ndarray, xp_markers: np.ndarray):
        """
        Generate the Jacobian matrix for each frame.

        Parameters:
        -----------
        q: np.ndarray
            The q values
        xp_markers: np.ndarray
            The position of the markers from the c3d during a certain frame

        Return:
        ------
            The Jacobian matrix
        """
        mat_jac = self.biorbd_model.technicalMarkersJacobian(q)

        jac = np.zeros((3 * self.nb_markers, self.nb_q))
        for m in range(self.nb_markers):
            jac[m * 3 : (m + 1) * 3, :] = mat_jac[m].to_array()

        return jac

    def solve(self, method: str = "lm", full: bool = False):
        """
        Solve the inverse kinematics by using least_square method from scipy

        Parameters:
        ----------
        method: str
            The method used by least_square to optimize the difference.

            If method = 'lm', the 'trf' method will be used for the first frame, in order to respect the bounds of the model.
            Then, the 'lm' method will be used for the following frames.
            If method = 'trf', the 'trf' method will be used for all the frames.
            If method = 'only_lm', the 'lm' method will be used for all the frames.

            In least_square:
                -‘trf’ : Trust Region Reflective algorithm, particularly suitable for large sparse problems
                        with bounds.
                        Generally robust method.
                -‘lm’ : Levenberg-Marquardt algorithm as implemented in MINPACK.
                        Doesn’t handle bounds and sparse Jacobians.
                        Usually the most efficient method for small unconstrained problems.

        """
        initial_bounds = (-np.inf, np.inf) if method == "only_lm" else self.bounds
        inital_method = "lm" if method == "only_lm" else "trf"

        bounds = self.bounds if method == "trf" else (-np.inf, np.inf)
        method = "lm" if method == "only_lm" else method

        if method != "lm" and method != "trf" and method != "only_lm":
            raise ValueError('This method is not implemented please use "trf", "lm" or "only_lm" as argument')

        else:
            for ii in range(0, self.nb_frames):
                print(f" ****   Frame {ii} / {self.nb_frames-1} ****")
                x0 = np.random.random(self.nb_q) * 0.1 if ii == 0 else self.q[:, ii - 1]
                sol = scipy.optimize.least_squares(
                    fun=self._marker_diff,
                    args=([self.xp_markers[:, :, ii]]),
                    bounds=initial_bounds if ii == 0 else bounds,
                    jac=self._marker_jac,
                    x0=x0,
                    method=inital_method if ii == 0 else method,
                    xtol=1e-6,
                    tr_options=dict(disp=False),
                )
                self.q[:, ii] = sol.x
        print("Inverse Kinematics done for all frames")

    def animate(self):
        """
        Animate the result of solve with bioviz.
        """
        b = bioviz.Viz(loaded_model=self.biorbd_model, show_muscles=False)
        b.load_experimental_markers(self.xp_markers)
        b.load_movement(self.q)
        b.exec()
