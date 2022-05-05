from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import biorbd
import biorbd_casadi
from casadi import MX, sumsqr, norm_2, Function, jacobian
from load_experimental_data import C3dData
from utils import get_range_q

biorbd_model_path = Path("/home/puchaud/Projets_Python/bioptim_exo/models/wu_converted_definitif_without_modif.bioMod")
c3d_path_file = Path("/home/puchaud/Projets_Python/bioptim_exo/data/F0_aisselle_05.c3d")

biorbd_model = biorbd.Model(str(biorbd_model_path.absolute()))
biorbd_model_casadi = biorbd_casadi.Model(str(biorbd_model_path.absolute()))

list_model_markers = [biorbd_model.markerNames()[i].to_string() for i in range(len(biorbd_model.markerNames()))]
biorbd_model.markers()

biorbd_model.markers(np.ones((biorbd_model.nbQ())))[0].to_array()

cd3_data = C3dData(str(c3d_path_file.absolute()), biorbd_model)
markers_data = cd3_data.trajectories
nb_frames = cd3_data.nb_frames


def marker_quad_diff(q, biorbd_model: biorbd.Model, model_xp: np.array):
    """
    Compute the difference between the marker position in the model and the position in the data

    Parameters
    ----------
    q : numpy.array
        The state of the model
    """
    nb_markers = biorbd_model.nbMarkers()
    if biorbd_model.__module__ == "biorbd.biorbd":
        model_markers = np.zeros((3, nb_markers))
        for m in range(nb_markers):
            model_markers[:, m] = biorbd_model.markers(q)[m].to_array()

        # return the sum of the squared differences between the model and the data
        return np.sum(np.linalg.norm(model_markers - model_xp, axis=0) ** 2)

    elif biorbd_model.__module__ == "biorbd_casadi.biorbd":
        model_markers = MX.zeros((3, nb_markers))
        sum_norm2 = MX.zeros(1)
        for m in range(nb_markers):
            sum_norm2 += norm_2(biorbd_model.markers(q)[m].to_mx() - model_xp[:, m])
        # return the sum of the squared differences between the model and the data
        J = sumsqr(sum_norm2)
        return Function("marker_quad_diff", [q_sym], [J]).expand()


def jac_marker_quad_diff(q, biorbd_model: biorbd.Model, model_xp: np.array):
    """
    Compute the jacobian of the difference between the marker position in the model and the position in the data

    Parameters
    ----------
    q : numpy.array
        The state of the model
    """
    nb_markers = biorbd_model.nbMarkers()
    if biorbd_model.__module__ == "biorbd_casadi.biorbd":
        model_markers = MX.zeros((3, nb_markers))
        sum_norm2 = MX.zeros(1)
        for m in range(nb_markers):
            sum_norm2 += norm_2(biorbd_model.markers(q)[m].to_mx() - model_xp[:, m])
        # return the sum of the squared differences between the model and the data
        J = sumsqr(sum_norm2)
        return Function("marker_quad_diff", [q_sym], [jacobian(J, q).T]).expand()
    else:
        raise NotImplementedError("Not implemented for this model")


def DM_to_array(q, func):
    return np.squeeze(func(q).toarray())


q_sym = MX.sym("q", biorbd_model.nbQ(), 1)
m_sym = MX.sym("m", 3, biorbd_model.nbMarkers())

J_func = marker_quad_diff(q_sym, biorbd_model_casadi, markers_data[:, :, 0])
jac_func = jac_marker_quad_diff(q_sym, biorbd_model_casadi, markers_data[:, :, 0])
bounds = tuple(get_range_q(biorbd_model))

q = np.zeros((biorbd_model.nbQ(), nb_frames))
for ii in range(nb_frames):
    print(f" ****   Frame {ii}  ****")
    x0 = np.zeros((biorbd_model.nbQ())) if ii == 0 else q[:, ii - 1]
    J_func = lambda q: DM_to_array(q, marker_quad_diff(q_sym, biorbd_model_casadi, markers_data[:, :, ii]))
    jac_func = lambda q: DM_to_array(q, jac_marker_quad_diff(q_sym, biorbd_model_casadi, markers_data[:, :, ii]))

    sol = minimize(
        fun=J_func, bounds=bounds, x0=x0, jac=jac_func, method="trust-constr", tol=1e-6, options=dict(disp=True)
    )
    q[:, ii] = sol.x
print(f" Inverse Kinematics done for all frames")
