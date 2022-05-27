import inverse_kinematics as ik
import time
import numpy as np
from ezc3d import c3d

model_path = "wu_converted_definitif_inverse_kinematics.bioMod"
c3d_path = "F0_aisselle_05.c3d"
c3d_path_nan = "F0_aisselle_05_nan.c3d"
c3d = c3d(c3d_path_nan)
markers_data = np.zeros((3, 19, 426))

my_ik = ik.InverseKinematics(model_path, markers_data)
start_markers = time.time()
my_ik.solve()
end_markers = time.time()
print("The time used to execute with least square and with a table of markers is given below")
print(end_markers - start_markers)

my_ik = ik.InverseKinematics(model_path, c3d)
start_c3d = time.time()
my_ik.solve()
end_c3d = time.time()
print("The time used to execute with least square and with a c3d file instead of a c3d path of markers is given below")
print(end_c3d - start_c3d)

my_ik = ik.InverseKinematics(model_path, c3d_path)
start1 = time.time()
my_ik.solve()
end1 = time.time()
print("The time used to execute with least square is given below")
print(end1 - start1)

my_ik = ik.InverseKinematics(model_path, c3d_path)
start2 = time.time()
my_ik.solve(method="lm")
end2 = time.time()
print("The time used with 'lm' to execute with least square is given below")
print(end2 - start2)


my_ik = ik.InverseKinematics(model_path, c3d_path)
start3 = time.time()
my_ik.solve(method="only_lm")
end3 = time.time()
print("The time used with 'only_lm' to execute with least square is given below")
print(end3 - start3)


my_ik = ik.InverseKinematics(model_path, c3d_path)
start4 = time.time()
my_ik.solve(method="trf")
end4 = time.time()
print("The time used with 'trf' to execute with least square is given below")
print(end4 - start4)


my_ik = ik.InverseKinematics(model_path, c3d_path_nan)
start_nan = time.time()
my_ik.solve()
end_nan = time.time()
print("The time used to execute with least square is given below, if the c3d has nan values")
print(end_nan - start_nan)

my_ik.animate()
