from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import biorbd_casadi as biorbd
from casadi import MX, SX, sumsqr, norm_2, Function, nlpsol
from load_experimental_data import C3dData
from utils import get_range_q
from typing import Union

biorbd_model_path = Path("/home/puchaud/Projets_Python/bioptim_exo/models/wu_converted_definitif_without_modif.bioMod")
c3d_path_file = Path("/home/puchaud/Projets_Python/bioptim_exo/data/F0_aisselle_05.c3d")

biorbd_model = biorbd.Model(str(biorbd_model_path.absolute()))

cd3_data = C3dData(str(c3d_path_file.absolute()), biorbd_model)
markers_data = cd3_data.trajectories
nb_frames = cd3_data.nb_frames

options = {
    "ipopt.tol": 1e-06,
    "ipopt.dual_inf_tol": 1.0,
    "ipopt.constr_viol_tol": 0.0001,
    "ipopt.compl_inf_tol": 0.0001,
    "ipopt.acceptable_tol": 1e-06,
    "ipopt.acceptable_dual_inf_tol": 10000000000.0,
    "ipopt.acceptable_constr_viol_tol": 0.01,
    "ipopt.acceptable_compl_inf_tol": 0.01,
    "ipopt.max_iter": 1000,
    "ipopt.hessian_approximation": "exact",
    "ipopt.limited_memory_max_history": 50,
    "ipopt.linear_solver": "mumps",
    "ipopt.mu_init": 0.1,
    "ipopt.warm_start_init_point": "no",
    "ipopt.warm_start_mult_bound_push": 0.001,
    "ipopt.warm_start_slack_bound_push": 0.001,
    "ipopt.warm_start_bound_push": 0.001,
    "ipopt.warm_start_slack_bound_frac": 0.001,
    "ipopt.warm_start_bound_frac": 0.001,
    "ipopt.bound_push": 0.01,
    "ipopt.bound_frac": 0.01,
    "ipopt.print_level": 1,
}


def marker_quad_diff(q: Union[MX, SX], biorbd_model: biorbd.Model, model_xp: Union[MX, SX]):
    """
    Compute the difference between the marker position in the model and the position in the data

    Parameters
    ----------
    q : Union[MX, SX]
        The state of the model
    biorbd_model : biorbd.Model
        The model
    model_xp : Union[MX, SX]
        Experimental marker position
    """
    nb_markers = biorbd_model.nbMarkers()
    model_markers = MX.zeros((3, nb_markers))
    sum_norm2 = MX.zeros(1)
    for m in range(nb_markers):
        sum_norm2 += norm_2(biorbd_model.markers(q)[m].to_mx() - model_xp[:, m])
    # return the sum of the squared differences between the model and the data
    J = sumsqr(sum_norm2)
    return Function("marker_quad_diff", [q_sym, model_xp], [J]).expand()


q_sym = MX.sym("q", biorbd_model.nbQ(), 1)
m_sym = MX.sym("m", 3, biorbd_model.nbMarkers())

J_func = marker_quad_diff(q_sym, biorbd_model, m_sym)

x_bounds_min = get_range_q(biorbd_model)[:, 0]
x_bounds_max = get_range_q(biorbd_model)[:, 1]

x_init = np.zeros(biorbd_model.nbQ())

ipopt_nlp = dict(x=q_sym, g=[])

ipopt_limits = dict(lbx=x_bounds_min, ubx=x_bounds_max, lbg=[], ubg=[], x0=x_init)

q = np.zeros((biorbd_model.nbQ(), nb_frames))
for ii in range(nb_frames):
    print(f" ****   Frame {ii}  ****")
    ipopt_nlp["f"] = J_func(q_sym, markers_data[:, :, ii])
    ipopt_limits["x0"] = q[:, ii - 1] if ii > 0 else x_init
    ik_solver = nlpsol("nlpsol", "ipopt", ipopt_nlp, options)
    out = {"sol": ik_solver.call(ipopt_limits)}
    q[:, ii] = np.squeeze(out["sol"]["x"].toarray())
print(f" Inverse Kinematics done for all frames")
