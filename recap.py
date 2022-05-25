"""
This script generate a recap of the Inverse Kinematics Class
"""
import static_inverse_kinematics as sik
import pandas as pd


model_path = "example/wu_converted_definitif_inverse_kinematics.bioMod"
c3d_path = "example/F0_aisselle_05.c3d"

df = pd.DataFrame(
    columns=[
        "method",
        "number_of_frame",
        "frequency",
        "residuals",
        "nb_iteration_diff",
        "nb_iteration_jac",
    ]
)

method_list = ["trf", "lm", "only_lm"]

for method in method_list:
    ik = sik.StaticInverseKinematics(model_path, c3d_path)
    ik.solve(method=method)
    ik.get_sol()
    cur_dict = dict(
        method=method,
        number_of_frame=ik.c3d["parameters"]["POINT"]["FRAMES"]["value"][0],
        frequency=ik.c3d["parameters"]["POINT"]["RATE"]["value"][0],
        residuals=ik.output["residuals"],
        nb_iteration_diff=ik.output["nb_iteration_diff"],
        nb_iteration_jac=ik.output["nb_iteration_jac"],
    )
    row_df = pd.DataFrame([cur_dict])
    df = pd.concat([df, row_df], ignore_index=True)

df.to_csv(f"RecapInverseKinematics")
