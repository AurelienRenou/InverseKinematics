import static_inverse_kinematics as sik

model_path = "wu_converted_definitif_inverse_kinematics.bioMod"
c3d_path = "F0_aisselle_05.c3d"

ik = sik.StaticInverseKinematics(model_path, c3d_path)
ik.solve()
ik.animate()
