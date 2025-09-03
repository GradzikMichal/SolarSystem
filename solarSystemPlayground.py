import math
import numpy as np
from numba import njit
import plotly.graph_objects as Gx


@njit
def solar_system():
    N = 50000
    h = 200000
    b = 9
    G = 6.6743 * 10 ** (-11)
    q = np.zeros((N + 1, b, 3))
    p = np.zeros((N + 1, b, 3))
    m = np.zeros(b)

    # positions
    q[0, 0, :] = [5.585 * 10 ** 8, 5.585 * 10 ** 8, 5.585 * 10 ** 8]
    q[0, 1, :] = [5.1979 * 10 ** 10, 7.6928 * 10 ** 9, -1.2845 * 10 ** 9]
    q[0, 2, :] = [-1.5041 * 10 ** 10, 9.708 * 10 ** 10, 4.4635 * 10 ** 10]
    q[0, 3, :] = [-1.1506 * 10 ** 9, -1.391 * 10 ** 11, -6.033 * 10 ** 10]
    q[0, 4, :] = [-4.8883 * 10 ** 10, -1.9686 * 10 ** 11, -8.8994 * 10 ** 10]
    q[0, 5, :] = [-8.1142 * 10 ** 11, 4.5462 * 10 ** 10, 3.9229 * 10 ** 10]
    q[0, 6, :] = [-4.278 * 10 ** 11, -1.3353 * 10 ** 12, -5.3311 * 10 ** 11]
    q[0, 7, :] = [2.7878 * 10 ** 12, 9.9509 * 10 ** 11, 3.9639 * 10 ** 8]
    q[0, 8, :] = [4.2097 * 10 ** 12, -1.3834 * 10 ** 12, -6.7105 * 10 ** 11]

    # momentum
    p[0, 0, :] = [-1.4663, 11.1238, 4.8370]
    p[0, 1, :] = [-1.5205 * 10 ** 4, 4.4189 * 10 ** 4, 2.518 * 10 ** 4]
    p[0, 2, :] = [-3.477 * 10 ** 4, -5.5933 * 10 ** 3, -316.8994]
    p[0, 3, :] = [2.9288 * 10 ** 4, -398.5759, -172.5873]
    p[0, 4, :] = [2.4533 * 10 ** 4, -2.7622 * 10 ** 3, -1.9295 * 10 ** 3]
    p[0, 5, :] = [-1.0724 * 10 ** 3, -1.1422 * 10 ** 4, -4.8696 * 10 ** 3]
    p[0, 6, :] = [8.7288 * 10 ** 3, -2.4369 * 10 ** 3, -1.3824 * 10 ** 3]
    p[0, 7, :] = [-2.4913 * 10 ** 3, 5.5197 * 10 ** 3, 2.4527 * 10 ** 3]
    p[0, 8, :] = [1.8271 * 10 ** 3, 4.7731 * 10 ** 3, 1.9082 * 10 ** 3]

    m[0] = 1.99 * 10 ** 30
    m[1] = 3.3 * 10 ** 23
    m[2] = 4.87 * 10 ** 24
    m[3] = 5.97 * 10 ** 26
    m[4] = 6.42 * 10 * 23
    m[5] = 1.9 * 10 ** 27
    m[6] = 5.68 * 10 ** 26
    m[7] = 8.68 * 10 ** 25
    m[8] = 1.02 * 10 ** 26

    a = np.ones((b, 1))
    ad = a * m
    M1 = ad.T
    M2 = []

    for i in range(b):
        for j in range(b):
            if i != j:
                M2.append(M1[j][i])
    M2 = np.array(M2)

    M3 = np.reshape(M2, (b, b - 1))
    M3 = np.column_stack((M3[0], M3[1], M3[2], M3[3], M3[4], M3[5], M3[6], M3[7], M3[8],))
    M = np.zeros((b, b - 1, 3))

    M[:, :, 0] = M3.T
    M[:, :, 1] = M3.T
    M[:, :, 2] = M3.T

    H = np.zeros((b, b, 3))
    H[:, :, 0] = np.eye(b)
    H[:, :, 1] = np.eye(b)
    H[:, :, 2] = np.eye(b)
    S45 = np.zeros((b, b - 1, 3))

    for i in range(N):
        s = p[i, :, :]
        t = q[i, :, :]
        print(i)
        for k in range(100):
            T = q[i, :, :] + h / 2 * (s + p[i, :, :])

            S1_help = (t + q[i, :, :]) * np.ones((b, 3))

            S1 = np.ones((b, b, 3))
            S1[:] = S1_help

            S2 = S1.transpose((1, 0, 2))
            S3 = S1 - S2
            S4 = []

            for matrix in range(len(S3)):
                S4_mat = []
                for row in range(len(S3[matrix])):
                    S4_help = []
                    for column in range(len(S3[row][row])):
                        if H[matrix][row][column] == 0:
                            S4_help.append(S3[matrix][row][column])
                    if len(S4_help) != 0:
                        S4_mat.append(S4_help)
                S4.append(S4_mat)

            S4 = np.array(S4)
            S42 = np.transpose(S4, (0, 2, 1))
            S43 = np.ones((b, b - 1))

            for matrix in range(len(S42)):
                row = len(S42[matrix])
                column = len(S42[matrix][0])
                for c in range(column):
                    norm = 0.0
                    for r in range(row):
                        norm += math.pow(S42[matrix][r][c], 2)
                    S43[matrix][c] = math.sqrt(norm)

            S44 = np.reshape(S43, (b, b - 1))

            S45[:, :, 0] = S44

            S45[:, :, 1] = S44
            S45[:, :, 2] = S44
            S5 = S4 / (S45 ** 3)

            S6 = np.multiply(M, S5)

            S7 = np.sum(S6, axis=1)

            S = p[i, :, :] + h * 4 * G * S7

            t = T

            s = S

        q[i + 1, :, :] = t
        p[i + 1, :, :] = s

    return q

q = solar_system()

figure = Gx.Figure(data=[Gx.Scatter3d(
    x=q[:, 0, 0],
    y=q[:, 0, 1],
    z=q[:, 0, 2],
    mode='lines',
    name="Sun",
)])

figure.add_trace(
    Gx.Scatter3d(
        x=q[:1000, 1, 0],
        y=q[:1000, 1, 1],
        z=q[:1000, 1, 2],
        mode='lines',
        name="Mercury"
    ))
figure.add_trace(
    Gx.Scatter3d(
        x=q[:5000, 2, 0],
        y=q[:5000, 2, 1],
        z=q[:5000, 2, 2],
        mode='lines',
        name="Venus"
    ))
figure.add_trace(
    Gx.Scatter3d(
        x=q[:10000, 3, 0],
        y=q[:10000, 3, 1],
        z=q[:10000, 3, 2],
        mode='lines',
        name="Earth"
    ))
figure.add_trace(
    Gx.Scatter3d(
        x=q[:15000, 4, 0],
        y=q[:15000, 4, 1],
        z=q[:15000, 4, 2],
        mode='lines',
        name="Mars",
    ))
figure.add_trace(
    Gx.Scatter3d(
        x=q[:20000, 5, 0],
        y=q[:20000, 5, 1],
        z=q[:20000, 5, 2],
        mode='lines',
        name="Jupiter"
    ))
figure.add_trace(
    Gx.Scatter3d(
        x=q[:25000, 6, 0],
        y=q[:25000, 6, 1],
        z=q[:25000, 6, 2],
        mode='lines',
        name="Saturn"
    ))
figure.add_trace(
    Gx.Scatter3d(
        x=q[:35000, 7, 0],
        y=q[:35000, 7, 1],
        z=q[:35000, 7, 2],
        mode='lines',
        name="Uranus"
    ))
figure.add_trace(
    Gx.Scatter3d(
        x=q[:, 8, 0],
        y=q[:, 8, 1],
        z=q[:, 8, 2],
        mode='lines',
        name="Neptune"
    ))

figure.show()
