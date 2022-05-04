from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import biorbd
from load_experimental_data import C3dData
from utils import get_range_q

biorbd_model_path = Path('/home/puchaud/Projets_Python/bioptim_exo/models/wu_converted_definitif_without_modif.bioMod')
c3d_path_file = Path('/home/puchaud/Projets_Python/bioptim_exo/data/F0_aisselle_05.c3d')

biorbd_model = biorbd.Model(str(biorbd_model_path.absolute()))
list_model_markers = [biorbd_model.markerNames()[i].to_string() for i in range(len(biorbd_model.markerNames()))]
biorbd_model.markers()

biorbd_model.markers(np.ones((biorbd_model.nbQ())))[0].to_array()

cd3_data = C3dData(str(c3d_path_file.absolute()), biorbd_model)
markers_data = cd3_data.trajectories
nb_frames = cd3_data.nb_frames


def marker_quad_diff(q, biorbd_model, model_xp: np.array):
    """
    Compute the difference between the marker position in the model and the position in the data

    Parameters
    ----------
    q : numpy.array
        The state of the model
    """
    nb_markers = biorbd_model.nbMarkers()
    model_markers = np.zeros((3, nb_markers))
    for m in range(nb_markers):
        model_markers[:, m] = biorbd_model.markers(q)[m].to_array()

    # return the sum of the squared differences between the model and the data
    return np.sum(np.linalg.norm(model_markers - model_xp, axis=0) ** 2)


marker_quad_diff(np.ones((biorbd_model.nbQ())), biorbd_model, markers_data[:, :, 1])

bounds = tuple(get_range_q(biorbd_model))

q = np.zeros((biorbd_model.nbQ(), nb_frames))
for ii in range(nb_frames):
    print(f" ****   Frame {ii}  ****")
    x0 = np.zeros((biorbd_model.nbQ())) if ii == 0 else q[:, ii - 1]
    sol = minimize(fun=marker_quad_diff,
                   args=(biorbd_model, markers_data[:, :, ii]),
                   bounds=bounds,
                   x0=x0,
                   method='trust-constr', tol=1e-6, options=dict(disp=True))
    q[:, ii] = sol.x
print(f" Inverse Kinematics done for all frames")

