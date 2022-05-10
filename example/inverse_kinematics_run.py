import static_inverse_kinematics as sik
import time


model_path = "wu_converted_definitif_inverse_kinematics.bioMod"
c3d_path = "F0_aisselle_05.c3d"

ik = sik.StaticInverseKinematics(model_path, c3d_path)
start2 = time.time()
ik.solve()
end2 = time.time()
print("The time used to execute with least square is given below")
print(end2 - start2)

ik.animate()
