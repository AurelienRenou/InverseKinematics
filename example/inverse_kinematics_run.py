import static_inverse_kinematics as sik
import time

model_path = "wu_converted_definitif_inverse_kinematics.bioMod"
c3d_path = "F0_aisselle_05.c3d"
c3d_path_nan = "F0_aisselle_05_nan.c3d"

ik = sik.StaticInverseKinematics(model_path, c3d_path)
start1 = time.time()
ik.solve()
end1 = time.time()
print("The time used to execute with least square is given below")
print(end1 - start1)

ik.animate()

ik = sik.StaticInverseKinematics(model_path, c3d_path)
start2 = time.time()
ik.solve("lm")
end2 = time.time()
print("The time used with 'lm' to execute with least square is given below")
print(end2 - start2)

ik.animate()

ik = sik.StaticInverseKinematics(model_path, c3d_path)
start3 = time.time()
ik.solve("only_lm")
end3 = time.time()
print("The time used with 'only_lm' to execute with least square is given below")
print(end3 - start3)

ik.animate()

ik = sik.StaticInverseKinematics(model_path, c3d_path)
start4 = time.time()
ik.solve("trf")
end4 = time.time()
print("The time used with 'trf' to execute with least square is given below")
print(end4 - start4)

ik.animate()

ik = sik.StaticInverseKinematics(model_path, c3d_path_nan)
start_nan = time.time()
ik.solve()
end_nan = time.time()
print("The time used to execute with least square is given below, if the c3d has nan values")
print(end_nan - start_nan)

ik.animate()
