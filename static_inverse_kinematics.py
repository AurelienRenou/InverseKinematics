from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import biorbd
import bioviz
from scripts.utils import get_range_q
from scripts.load_experimental_data import C3dData


class StaticInverseKinematics:
    """
    The class for generate inverse kinemacs from c3d files
    """

    def __init__(
        self,
        biorbd_model_path,
        c3d_path_file,
    ):
        self.biorbd_model_path = biorbd_model_path
        self.c3d_path_file = c3d_path_file

        self.biorbd_model = biorbd.Model(self.biorbd_model_path)
        self.c3d_data = C3dData(self.c3d_path_file, self.biorbd_model)

        self.list_model_markers = [
            self.biorbd_model.markerNames()[i].to_string() for i in range(len(self.biorbd_model.markerNames()))
        ]
        self.markers_data = self.c3d_data.trajectories
        self.nb_frames = self.c3d_data.nb_frames
        self.nb_markers = self.biorbd_model.nbMarkers()
        self.model_markers = np.zeros((3, self.nb_markers))
        self.q = np.zeros((self.biorbd_model.nbQ(), self.nb_frames))

    def solve(self):
        def _marker_quad_diff(q, model_xp):
            """
            Compute the difference between the marker position in the model and the position in the data
            """
            for m in range(self.nb_markers):
                self.model_markers[:, m] = self.biorbd_model.markers(q)[m].to_array()

            # return the sum of the squared differences between the model and the data
            return np.sum(np.linalg.norm(self.model_markers - model_xp, axis=0) ** 2)

        # self._marker_quad_diff()
        bounds = tuple(get_range_q(self.biorbd_model))

        for ii in range(self.nb_frames):
            print(f" ****   Frame {ii}  ****")
            x0 = np.zeros((self.biorbd_model.nbQ())) if ii == 0 else self.q[:, ii - 1]
            method = "trust-constr" if ii == 0 else "SLSQP"
            sol = minimize(
                fun=_marker_quad_diff,
                args=self.markers_data[:, :, ii],
                bounds=bounds,
                x0=x0,
                method=method,
                tol=1e-6,
                options=dict(disp=False),
            )
            self.q[:, ii] = sol.x
        print(f" Inverse Kinematics done for all frames")

    def animate(self):
        b = bioviz.Viz(loaded_model=self.biorbd_model, show_muscles=False)
        b.load_experimental_markers(self.markers_data)
        b.load_movement(self.q)
        b.exec()
