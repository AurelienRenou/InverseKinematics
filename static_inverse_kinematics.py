from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import biorbd
import bioviz
from scripts.utils import get_range_q, get_range_max_q, get_range_min_q
from scripts.load_experimental_data import C3dData
import scipy


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

    def _marker_diff(self, q, model_xp):
        """
        Compute the difference between the marker position in the model and the position in the data
        """
        mat_pos_markers = self.biorbd_model.technicalMarkers(q)
        vect_pos_markers = np.zeros(3*self.nb_markers)

        for m in range(self.nb_markers):
            vect_pos_markers[m * 3:(m + 1) * 3] = mat_pos_markers[m].to_array()

        # return the sum of the squared differences between the model and the data
        return vect_pos_markers - np.reshape(model_xp.T, (19*3,))  # todo: 19

    def _marker_jac(self, q, model_xp):
        """

        """
        mat_Jac = self.biorbd_model.technicalMarkersJacobian(q)

        Jac = np.zeros((3 * self.nb_markers, self.biorbd_model.nbQ())) # todo:nb_q in self, modify the name
        for m in range(self.nb_markers):
            Jac[m*3:(m+1)*3, :] = mat_Jac[m].to_array()

        return Jac

    def solve(self, method: str = "lm"):
        bounds_max = np.squeeze(get_range_max_q(self.biorbd_model))
        bounds_min = np.squeeze(get_range_min_q(self.biorbd_model))
        c3d_markers = self.c3d_data.marker_names
        model_markers = [i.to_string() for i in self.biorbd_model.markerNames()]

        if method == "trf":
            for ii in range(1, self.nb_frames):
                print(f" ****   Frame {ii}  ****")
                x0 = self.q[:, ii - 1]
                sol = scipy.optimize.least_squares(
                    fun=self._marker_diff,
                    args=[self.markers_data[:, :, ii]],
                    # bounds=(bounds_min, bounds_max),
                    jac=self._marker_jac,
                    x0=x0,
                    method=method,
                    xtol=1e-6,
                    tr_options=dict(disp=False),
                )
                self.q[:, ii] = sol.x

        else:
            if method != "lm":
                print("method = lm")
            # The first frame use the trf method in order to be sure to respect the bounds
            x0 = np.random.random((self.biorbd_model.nbQ())) * 0.01
            sol = scipy.optimize.least_squares(
                fun=self._marker_diff,
                args=[self.markers_data[:, :, 0]],
                bounds=(bounds_min, bounds_max),
                jac=self._marker_jac,
                x0=x0,
                method="trf",
                xtol=1e-6,
                tr_options=dict(disp=False),
            )
            self.q[:, 0] = sol.x

            for ii in range(1, self.nb_frames):
                print(f" ****   Frame {ii}  ****")

                x0 = self.q[:, ii - 1]
                sol = scipy.optimize.least_squares(
                    fun=self._marker_diff,
                    args=[self.markers_data[:, :, ii]],
                    # bounds=(bounds_min, bounds_max),
                    jac=self._marker_jac,
                    x0=x0,
                    method="lm",
                    xtol=1e-6,
                    tr_options=dict(disp=False),
                )
                self.q[:, ii] = sol.x
        print(f" Inverse Kinematics done for all frames")

    def animate(self):
        b = bioviz.Viz(loaded_model=self.biorbd_model, show_muscles=False)
        b.load_experimental_markers(self.markers_data)
        b.load_movement(self.q)
        b.exec()
