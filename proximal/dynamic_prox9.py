import numpy as np
import pylab as plt

# import scipy.integrate as integrate
# import scipy

# import seaborn as sns
# import ot
# import ot.plot
import pol_deg3

# import pdb

# from cmath import *

"""
Define different functions and prox algorithms
"""


def mid_int(ms, fs):
    N, P = ms.shape
    midm = np.zeros((N, P))
    midp = np.zeros((N, P))
    midm[:-1, :] = (ms[:-1, :] + ms[1:, :]) / 2
    midm[-1, :] = (ms[-1, :] + ms[0, :]) / 2
    midp = (fs[:, :-1] + fs[:, 1:]) / 2
    return (midm, midp)


def mid_intconj(mc, fc):
    N = mc.shape[0]
    P = fc.shape[1]
    ms = np.zeros((N, P))
    fs = np.zeros((N, P + 1))
    ms[0, :] = 1 / 2 * (mc[0, :] + mc[-1, :])
    ms[1:, :] = 1 / 2 * (mc[:-1, :] + mc[1:, :])
    fs[:, 0] = 1 / 2 * fc[:, 0]
    fs[:, -1] = 1 / 2 * fc[:, -1]
    fs[:, 1:-1] = 1 / 2 * (fc[:, :-1] + fc[:, 1:])
    return (ms, fs)


div = lambda mb, fb, N, P: N * (mb[1:, :] - mb[:-1, :]) + P * (fb[:, 1:] - fb[:, :-1])


def fun_J(m, f):
    r = 0
    if f > 1e-5 and m > 1e-5:
        r = np.linalg.norm(m) ** 2 / (2 * f)
    return r


def fun_j(V):
    S = 0
    for M, F in zip(V[0], V[1]):
        for m, f in zip(M, F):
            S += fun_J(m, f)
    return S


def prox_J(m, f, gamma):
    a, b, c, d = (
        1,
        (2 * gamma - f),
        gamma**2 - 2 * gamma * f,
        -f * gamma**2 - gamma / 2 * np.linalg.norm(m) ** 2,
    )
    f_et = pol_deg3.maxSol(a, b, c, d)
    if f_et <= 0:
        return (0, 0)
    else:
        mu_fet = f_et * m / (f_et + gamma)
        return (mu_fet, f_et)


def prox_j(V, gamma):
    Mprox = np.zeros(V[0].shape)
    Fprox = np.zeros(V[1].shape)
    n, p = Fprox.shape
    for k in range(n):
        for l in range(p):
            Mprox[k, l], Fprox[k, l] = prox_J(V[0][k, l], V[1][k, l], gamma)
    return (Mprox, Fprox)


def prox_jconj(V, gamma):
    U = prox_j((V[0] / gamma, V[1] / gamma), 1 / gamma)
    return (V[0] - gamma * U[0], V[1] - gamma * U[1])


def projC(m, f, B, Ay, gamma):
    N = m.shape[0]
    P = f.shape[1] - 1

    xc = 1 / N * (np.arange(1, N + 1) - 1 / 2)
    tc = 1 / P * (np.arange(1, P + 1) - 1 / 2)
    xs = 1 / N * (np.arange(1, N + 2) - 1)
    ts = 1 / P * (np.arange(1, P + 2) - 1)

    U = np.zeros(P * N + N * (P + 1))
    for n in range(N):
        for p in range(P):
            U[n + p * N] = m[n, p]
    for n in range(N):
        for p in range(P + 1):
            U[P * N + n + p * N] = f[n, p]

    Uproj = U - np.dot(B, U) + Ay

    mproj = np.zeros(m.shape)
    fproj = np.zeros(f.shape)
    for n in range(N):
        for p in range(P):
            mproj[n, p] = Uproj[n + p * N]
    for n in range(N):
        for p in range(P + 1):
            fproj[n, p] = Uproj[P * N + n + p * N]

    return (mproj, fproj)


### PD solver ###


def pd_solver(f0, f1, sigma=1, tau=0.07, theta=0.8, P=20, L=50):
    # sigma = 10, tau = 1/(sigma*100)
    N = f0.size
    U = (np.zeros((N, P + 1)), np.zeros((N, P + 2)))
    for k in range(P + 1):
        U[1][:, k] = f0
    Gamma = U
    V = mid_int(U[0], U[1])
    # xc = 1/N*(np.arange(N)+1/2)
    xc = 1 / N * np.arange(N)
    A = np.zeros((N * (P + 1) + 2 * N + 2 * P, N * (P + 1) + N * (P + 2)))
    B = np.zeros((N * (P + 1) + 2 * N + 2 * P + 1, N * (P + 1) + 2 * N + 2 * P))

    #    for k in range(N*(P+1)):
    #        A[k,k]=-N/P
    #        A[k,(k+1)*(k+1<N*(P+1))]=N/P
    #        A[k,(P+1)*N+k]=-1
    #        A[k,(P+1)*N+N+k]=1
    for k in range(N):
        for l in range(P + 1):
            A[l * N + k, l * N + k] = -N / P
            A[l * N + k, l * N + ((k + 1) % N)] = N / P
            A[l * N + k, (P + 1) * N + l * N + k] = -1
            A[l * N + k, (P + 1) * N + (l + 1) * N + k] = 1

    for k in range(N):
        A[N * (P + 1) + k, N * (P + 1) + k] = 1
        A[N * (P + 1) + N + k, N * (P + 1) + N * (P + 1) + k] = 1
    for k in range(P):
        A[
            N * (P + 1) + 2 * N + k,
            N * (P + 1) + (k + 1) * N : N * (P + 1) + (k + 1) * N + N,
        ] = 1 * np.cos(2 * np.pi * xc)
        A[
            N * (P + 1) + 2 * N + P + k,
            N * (P + 1) + (k + 1) * N : N * (P + 1) + (k + 1) * N + N,
        ] = 1 * np.sin(2 * np.pi * xc)

    AtA = np.dot(A, A.T)
    B[:-1, :] = AtA
    B[-1, 0] = 1
    BtB = np.dot(B.T, B)
    #    Uhb,sb,Vhb = scipy.linalg.svd(BtB)

    #    sinv = 1/sb
    #    Sinv = np.diag(sinv)
    #    BtBinvsvd= np.dot(Vhb.T,np.dot(Sinv,Uhb.T))
    BtBinv = np.linalg.inv(BtB)

    Atilde = np.zeros((N * (P + 1) + 2 * N + 2 * P + 1, N * (P + 1) + N * (P + 2)))
    Atilde[:-1, :] = A

    C = np.dot(np.dot(A.T, BtBinv), np.dot(B.T, Atilde))

    y = np.zeros(N * (P + 1) + 2 * N + 2 * P + 1)
    y[N * (P + 1) : N * (P + 1) + N] = f0
    y[N * (P + 1) + N : N * (P + 1) + 2 * N] = f1

    Ay = np.dot(np.dot(A.T, BtBinv), np.dot(B.T, y))

    j = []
    for l in range(L):
        I = mid_int(Gamma[0], Gamma[1])
        V = prox_jconj((V[0] + sigma * I[0], V[1] + sigma * I[1]), sigma)

        Uprec = (U[0].copy(), U[1].copy())
        Iconj = mid_intconj(V[0], V[1])
        U = projC(Uprec[0] - tau * Iconj[0], Uprec[1] - tau * Iconj[1], C, Ay, tau)
        j.append(fun_j(mid_int(U[0], U[1])))
        Gamma = (U[0] + theta * (U[0] - Uprec[0]), U[1] + theta * (U[1] - Uprec[1]))

    return U, j


### afficher les courbes ###
def print_curve(V, E, i):
    n = V.shape[0]

    plt.figure(i, figsize=(7, 7))
    plt.clf()
    for k in range(n):
        x1 = V[int(E[k][0])][0]
        x2 = V[int(E[k][1])][0]
        y1 = V[int(E[k][0])][1]
        y2 = V[int(E[k][1])][1]
        plt.plot([x1, x2], [y1, y2])


def density_to_measure(f):
    N = f.shape[0]
    mu = np.zeros((N, 2))
    for n in range(N):
        mu[n, 0] = 2 * n * np.pi / N
        mu[n, 1] = f[n] / 300
    return mu


def measure_to_curve(mu):
    n = mu.shape[0]
    V = np.zeros((n, 2))
    E = np.zeros((n, 2))

    mu1 = mu.copy()
    mu1[:, 0] += (mu1[:, 0] <= 0) * 2 * np.pi
    mu_sorted = np.sort(mu1.view("i8,i8"), order=["f0"], axis=0).view(np.float)
    S = np.array([0.0, 0.0])

    for i in range(n):
        V[i] = S
        E[i, 0] = i
        E[i, 1] = (i + 1 < n) * (i + 1) + (i + 1 == n) * 0
        S = S + mu_sorted[i, 1] * np.array(
            [np.cos(mu_sorted[i, 0]), np.sin(mu_sorted[i, 0])]
        )

    return (V, E)


def anime_geodesic(v):
    for i in range(0, v.shape[1]):
        plt.figure(1, figsize=(7, 7))
        plt.clf()
        f = v[:, i]

        Vd, Ed = measure_to_curve(density_to_measure(f))
        Vdshape = Vd.shape[0]
        for k in range(Vdshape):
            x1 = Vd[int(Ed[k][0])][0]
            x2 = Vd[int(Ed[k][1])][0]
            y1 = Vd[int(Ed[k][0])][1]
            y2 = Vd[int(Ed[k][1])][1]
            plt.plot([x1, x2], [y1, y2])
        plt.pause(0.1)


def anime_geodesic_meas(v):
    n = np.arange(v.shape[0]) / v.shape[0]
    r = np.max(v) + 1
    for i in range(0, v.shape[1]):
        plt.figure(2, figsize=(7, 7))
        plt.clf()
        plt.ylim(0, r)
        plt.scatter(n, v[:, i])
        plt.pause(0.1)


def pd_unbalanced_solver(f0, f1, sigma=1, tau=0.07, theta=0.8, P=20, L=50):
    # sigma = 10, tau = 1/(sigma*100)
    N = f0.size
    U = (np.zeros((N, P + 1)), np.zeros((N, P + 2)), np.zeros((N, P + 1)))
    for k in range(P + 1):
        U[1][:, k] = f0
    Gamma = U
    V = mid_int(U[0], U[1])
    V = (V[0], V[1], U[2])

    # xc = 1/N*(np.arange(N)+1/2)
    xc = 1 / N * np.arange(N)
    A = np.zeros((N * (P + 1) + 2 * N + 2 * P, N * (P + 1) + N * (P + 2) + N * (P + 1)))
    B = np.zeros((N * (P + 1) + 2 * N + 2 * P + 1, N * (P + 1) + 2 * N + 2 * P))

    #    for k in range(N*(P+1)):
    #        A[k,k]=-N/P
    #        A[k,(k+1)*(k+1<N*(P+1))]=N/P
    #        A[k,(P+1)*N+k]=-1
    #        A[k,(P+1)*N+N+k]=1
    for k in range(N):
        for l in range(P + 1):
            A[l * N + k, l * N + k] = -N / (2 * np.pi)
            A[l * N + k, l * N + ((k + 1) % N)] = N / (2 * np.pi)
            A[l * N + k, (P + 1) * N + l * N + k] = -P
            A[l * N + k, (P + 1) * N + (l + 1) * N + k] = P
            A[l * N + k, (P + 1) * N + (P + 2) * N + l * N + k] = -1

    for k in range(N):
        A[N * (P + 1) + k, N * (P + 1) + k] = 1
        A[N * (P + 1) + N + k, N * (P + 1) + N * (P + 1) + k] = 1
    for k in range(P):
        A[
            N * (P + 1) + 2 * N + k,
            N * (P + 1) + (k + 1) * N : N * (P + 1) + (k + 1) * N + N,
        ] = 1 * np.cos(2 * np.pi * xc)
        A[
            N * (P + 1) + 2 * N + P + k,
            N * (P + 1) + (k + 1) * N : N * (P + 1) + (k + 1) * N + N,
        ] = 1 * np.sin(2 * np.pi * xc)

    AtA = np.matmul(A, A.T)
    B[:-1, :] = AtA
    B[-1, 0] = 1
    BtB = np.matmul(B.T, B)
    #    Uhb,sb,Vhb = scipy.linalg.svd(BtB)

    #    sinv = 1/sb
    #    Sinv = np.diag(sinv)
    #    BtBinvsvd= np.dot(Vhb.T,np.dot(Sinv,Uhb.T))
    BtBinv = np.linalg.inv(BtB)

    Atilde = np.zeros(
        (N * (P + 1) + 2 * N + 2 * P + 1, N * (P + 1) + N * (P + 2) + N * (P + 1))
    )
    Atilde[:-1, :] = A

    C = np.matmul(np.matmul(A.T, BtBinv), np.matmul(B.T, Atilde))

    y = np.zeros(N * (P + 1) + 2 * N + 2 * P + 1)
    y[N * (P + 1) : N * (P + 1) + N] = f0
    y[N * (P + 1) + N : N * (P + 1) + 2 * N] = f1

    Ay = np.matmul(np.matmul(A.T, BtBinv), np.matmul(B.T, y))

    j = []
    for l in range(L):
        I = mid_int(Gamma[0], Gamma[1])
        I = (I[0], I[1], Gamma[2])
        V = prox_jconj_unbalanced(
            (V[0] + sigma * I[0], V[1] + sigma * I[1], V[2] + sigma * I[2]), sigma
        )

        Uprec = (U[0].copy(), U[1].copy(), U[2].copy())
        Iconj = mid_intconj(V[0], V[1])
        U = projC_unbalanced(
            Uprec[0] - tau * Iconj[0],
            Uprec[1] - tau * Iconj[1],
            Uprec[2] - tau * V[2],
            C,
            Ay,
            tau,
        )
        j.append((fun_j_unbalanced(mid_int(U[0], U[1]), U[2])))
        Gamma = (
            U[0] + theta * (U[0] - Uprec[0]),
            U[1] + theta * (U[1] - Uprec[1]),
            U[2] + theta * (U[2] - Uprec[2]),
        )

    return U, j


def prox_jconj_unbalanced(V, sigma):
    Mprox = np.zeros(V[0].shape)
    Fprox = np.zeros(V[1].shape)
    Zetaprox = np.zeros(V[2].shape)
    n, p = Fprox.shape
    for k in range(n):
        for l in range(p):
            Mprox[k, l], Fprox[k, l], Zetaprox[k, l] = prox_J_unbalanced(
                V[0][k, l] / sigma, V[1][k, l] / sigma, V[2][k, l] / sigma, 1 / sigma
            )

    # U = prox_j((V[0]/gamma, V[1]/gamma), 1/gamma)
    return (V[0] - sigma * Mprox, V[1] - sigma * Fprox, V[2] - sigma * Zetaprox)


def prox_J_unbalanced(m, f, zeta, gamma):
    a, b, c, d = (
        1,
        (2 * gamma - f),
        gamma**2 - 2 * gamma * f,
        -f * gamma**2 - gamma / 2 * (m**2 + zeta**2),
    )
    f_et = pol_deg3.maxSol(a, b, c, d)
    if f_et <= 0:
        return (0, 0, 0)
    else:
        return (f_et * m / (f_et + gamma), f_et, f_et * zeta / (f_et + gamma))


def projC_unbalanced(m, f, zeta, B, Ay, gamma):
    N = m.shape[0]
    P = f.shape[1] - 1

    U = np.zeros(P * N + N * (P + 1) + N * (P + 1))
    for n in range(N):
        for p in range(P):
            U[n + p * N] = m[n, p]
    for n in range(N):
        for p in range(P + 1):
            U[P * N + n + p * N] = f[n, p]

    for n in range(N):
        for p in range(P):
            U[P * N + (P + 1) * N + n + p * N] = zeta[n, p]

    Uproj = U - np.dot(B, U) + Ay

    mproj = np.zeros(m.shape)
    fproj = np.zeros(f.shape)
    zetaproj = np.zeros(f.shape)
    for n in range(N):
        for p in range(P):
            mproj[n, p] = Uproj[n + p * N]
    for n in range(N):
        for p in range(P + 1):
            fproj[n, p] = Uproj[P * N + n + p * N]
    for n in range(N):
        for p in range(P + 1):
            zetaproj[n, p] = Uproj[P * N + (P + 1) * N + n + p * N]

    return (mproj, fproj, zetaproj)


def fun_j_unbalanced(V):
    N = V[0].shape[0]
    P = V[0].shape[1]
    S = 0
    for k in range(N):
        for p in range(P):
            if V[0][k, p] > 1e-6 and V[1][k, p] > 1e-6 and V[2][k, p] > 1e-6:
                S += (V[0][k, p] ** 2 + V[2][k, p] ** 2) / (2 * V[1][k, p])

    return S


r = 300
f0 = r * np.ones(300) / 300
f1 = np.zeros(300)
f1[0] = 1 / 4
f1[75] = 1 / 4
f1[150] = 1 / 4
f1[225] = 1 / 4
f1 = r * f1 / np.sum(f1)
f2 = np.zeros(300)
f2[40] = 1 / 4
f2[39] = 1 / 5
f2[38] = 1 / 6
f2[37] = 1 / 8
f2[41] = 1 / 5
f2[42] = 1 / 6
f2[43] = 1 / 8
f2[115] = 1 / 4
f2[114] = 1 / 5
f2[113] = 1 / 6
f2[112] = 1 / 8
f2[116] = 1 / 5
f2[117] = 1 / 6
f2[118] = 1 / 8
f2[190] = 1 / 4
f2[189] = 1 / 5
f2[188] = 1 / 6
f2[187] = 1 / 8
f2[191] = 1 / 5
f2[192] = 1 / 6
f2[193] = 1 / 8
f2[265] = 1 / 4
f2[264] = 1 / 5
f2[263] = 1 / 6
f2[262] = 1 / 8
f2[266] = 1 / 5
f2[267] = 1 / 6
f2[268] = 1 / 8
f2 = r * f2 / np.sum(f2)
f3 = np.zeros(300)
f3[15] = 1 / 3
f3[115] = 1 / 3
f3[215] = 1 / 3
f3 = r * f3 / np.sum(f3)
