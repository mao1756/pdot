import false_geodesics as geod
import numpy as np

n = 8
V1, E1 = geod.ngone(n)  # polygone régulier de taille n


V2 = np.array([[0.0, 0.0], [1, 1], [-2, 4]])
E2 = np.array([[0, 1], [1, 2], [2, 0]])
V2, E2 = geod.normalize_curve(
    V2, E2
)  # normalise les longueurs pour bien avoir un périmètre égal à 1


Vt, Et, mut = geod.geodesic(V1, E1, V2, E2, 1 / 3, 1)
# Vt, Et = geod.minkowski_sum(V1,E1, V2,E2,1/3)


# Affichage des polygones
geod.print_curve(V1, E1, 1)
geod.print_curve(V2, E2, 2)
geod.print_curve(Vt, Et, 3)


geod.plt.show()

geod.anime_geodesic(V1, E1, V2, E2, 1)  # peut poser des problèmes (sous linux)


##
import false_geodesics as fg
import dynamic_prox9 as geod
import numpy as np
from scipy.io import savemat


V0 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.2], [0.0, 0.2]])
# V1=np.array([[0.,0.],[1.,0.1],[1.1,0.3], [0.,0.2]])
E0 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
V0, E0 = fg.normalize_curve(V0, E0)  # normalise les longueurs


Y = np.matrix([[0.0, 0.0], [0.9, -0.1], [0.9, 0.3], [0.0, 0.2]])
theta = np.pi / 8
R = np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
Y = (R * (Y.transpose())).transpose()
V1 = np.array(Y)
E1 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
V1, E1 = fg.normalize_curve(V1, E1)  # normalise les longueurs

N = 150
P = 50
f0 = np.zeros((N,))
f0[0] = 0.41666667
ind = np.rint(N / 4)
ind = ind.astype(int)
f0[ind] = 0.08333333
ind = np.rint(N / 2)
ind = ind.astype(int)
f0[ind] = 0.41666667
ind = np.rint(3 * N / 4)
ind = ind.astype(int)
f0[ind] = 0.08333333


f1 = np.zeros((N,))
ind = np.rint(N * 1.17809725 / (2 * np.pi))
ind = ind.astype(int)
f1[ind] = 0.16590096
ind = np.rint(N * 2.85955079 / (2 * np.pi))
ind = ind.astype(int)
f1[ind] = 0.37557428
ind = np.rint(N * 4.31968990 / (2 * np.pi))
ind = ind.astype(int)
f1[ind] = 0.08295048
ind = np.rint(N * 5.7798290 / (2 * np.pi))
ind = ind.astype(int)
f1[ind] = 0.37557428


# angles=np.arange(0,2*np.pi,2*np.pi/N)
#
# sig0=1
# f0=np.zeros((N,))
# f0[0:(np.int(N/2))]= np.exp(-np.power(angles[0:(np.int(N/2))] - np.pi/2, 2.) / (2 * np.power(sig0, 2.)))
# f0[(np.int(N/2)):] = f0[0:(np.int(N/2))]
# r=np.sum(f0)
# f0=N*f0/r
#
# sig1=0.3
# f1=np.zeros((N,))
# f1[0:(np.int(N/2))]= np.exp(-np.power(angles[0:(np.int(N/2))] - np.pi/6, 2.) / (2 * np.power(sig1, 2.)))
# f1[(np.int(N/2)):] = f1[0:(np.int(N/2))]
# r=np.sum(f1)
# f1=N*f1/r


U, j = geod.pd_solver(f0, f1, 0.5, 0.25, 0.6, 50, 1500)
rho = U[1]  # permet d'avoir les mesures d'interpolation entre f0 et f2
m = U[0]


savemat(
    "matching_constrained_Wasserstein_bis.mat",
    {"f0": f0, "f1": f1, "rho_path": rho, "m_path": m, "Ener": j},
)


##
import false_geodesics as fg
import dynamic_prox9 as geod
import numpy as np
from scipy.io import savemat
import time

N = 150
P = 50
f0 = np.zeros((N,))
f0[0] = 1
ind = np.rint(N / 4)
ind = ind.astype(int)
f0[ind] = 1
ind = np.rint(N / 3)
ind = ind.astype(int)
f0[ind] = 1
ind = np.rint(2 * N / 3)
ind = ind.astype(int)
f0[ind] = 1
ind = np.rint(3 * N / 4)
ind = ind.astype(int)
f0[ind] = 1
r = np.sum(f0)
f0 = N * f0 / r


angles = np.arange(0, 2 * np.pi, 2 * np.pi / N)

f1 = np.zeros((N,))
f1[0 : (np.int(N / 6))] = 1.0
f1[(np.int(N / 3)) : (np.int(N / 2))] = 1.0
f1[(np.int(4 * N / 6)) : (np.int(5 * N / 6))] = 1.0
r = np.sum(f1)
f1 = N * f1 / r

start = time.time()
U, j = geod.pd_solver(f0, f1, 0.5, 0.25, 0.6, 50, 10)
end = time.time()
print(end - start)
rho = U[1]  # permet d'avoir les mesures d'interpolation entre f0 et f1
m = U[0]


savemat(
    "matching_Wasserstein_Reuleaux3.mat",
    {"f0": f0, "f1": f1, "rho_path": rho, "m_path": m, "Ener": j},
)


## Example Mao
# import false_geodesics as fg
import dynamic_prox9 as geod
import numpy as np
from scipy.io import savemat
import time

N = 40
P = 50
angles = np.arange(0, 2 * np.pi, 2 * np.pi / N)

# f0=np.ones((N,))
# r=np.sum(f0)
f0 = 0.2 * np.ones((N,))
ind = np.rint(N / 8)
ind = ind.astype(int)
f0[ind] = 3
ind = np.rint(3 * N / 8)
ind = ind.astype(int)
f0[ind] = 3
ind = np.rint(5 * N / 8)
ind = ind.astype(int)
f0[ind] = 3
ind = np.rint(7 * N / 8)
ind = ind.astype(int)
f0[ind] = 3
f0[0] = f0[0] - np.dot(f0, np.cos(angles))
ind = np.rint(N / 4)
ind = ind.astype(int)
f0[ind] = f0[ind] - np.dot(f0, np.sin(angles))
r = np.sum(f0)
# f0=N*f0/r
f0 = f0 / r


f1 = 0.2 * np.ones((N,))
f1[0] = 4
ind = np.rint(N / 4)
ind = ind.astype(int)
f1[ind] = 2
ind = np.rint(N / 2)
ind = ind.astype(int)
f1[ind] = 4
ind = np.rint(3 * N / 4)
ind = ind.astype(int)
f1[ind] = 2
f1[0] = f1[0] - np.dot(f1, np.cos(angles))
ind = np.rint(N / 4)
ind = ind.astype(int)
f1[ind] = f1[ind] - np.dot(f1, np.sin(angles))
r = np.sum(f1)
# f1=N*f1/r
f1 = f1 / r

start = time.time()
U, j = geod.pd_solver(f0, f1, 0.5, 0.25, 0.6, P, 750)
# U,j = geod.pd_unbalanced_solver(f0,f1,0.5,0.25,0.6,P,750)
end = time.time()
print(end - start)
rho = U[1]  # permet d'avoir les mesures d'interpolation entre f0 et f1
m = U[0]


# savemat('matching_CWasserstein_N_40_P_50.mat',{'f0':f0,'f1':f1,'rho_path':rho,'m_path':m,'Ener':j})
savemat(
    "matching_Mao_Wasserstein3_N_40_P_50.mat",
    {"f0": f0, "f1": f1, "rho_path": rho, "m_path": m, "Ener": j},
)
