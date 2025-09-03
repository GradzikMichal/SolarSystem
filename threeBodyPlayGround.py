import time

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as Gx
import plotly.io as pio

pio.renderers.default = "browser"


def kepler_three_body(q01, q02, q03, p01, p02, p03, m1, m2, m3, h, N):
    G = 6.6743 * 10 ** (-11)
    q1 = np.zeros((3, N + 1))
    q2 = np.zeros((3, N + 1))
    q3 = np.zeros((3, N + 1))
    p1 = np.zeros((3, N + 1))
    p2 = np.zeros((3, N + 1))
    p3 = np.zeros((3, N + 1))

    q1[:, 0] = q01

    p1[:, 0] = p01
    p2[:, 0] = p02
    q2[:, 0] = q02
    p3[:, 0] = p03
    q3[:, 0] = q03

    for i in range(N):
        s1 = p1[:, i]
        s2 = p2[:, i]
        s3 = p3[:, i]
        t1 = q1[:, i]
        t2 = q2[:, i]
        t3 = q3[:, i]
        for k in range(100):
            T1 = q1[:, i] + h / 2.0 * (s1 + p1[:, i])
            T2 = q2[:, i] + h / 2.0 * (s2 + p2[:, i])
            T3 = q3[:, i] + h / 2.0 * (s3 + p3[:, i])
            a = t1 + q1[:, i] - t2 - q2[:, i]

            b = t1 + q1[:, i] - t3 - q3[:, i]
            c = t2 + q2[:, i] - t3 - q3[:, i]
            S1 = p1[:, i] - 4 * h * G * (m2 * a / np.dot(a.T, a) ** (3 / 2.0) + m3 * b / np.dot(b.T, b) ** (3 / 2.0))
            S2 = p2[:, i] - 4 * h * G * (m3 * c / np.dot(c.T, c) ** (3 / 2.0) - m1 * a / np.dot(a.T, a) ** (3 / 2.0))
            S3 = p3[:, i] + 4 * h * G * (m1 * b / np.dot(b.T, b) ** (3 / 2.0) + m2 * c / np.dot(c.T, c) ** (3 / 2.0))
            s1 = S1
            s2 = S2
            s3 = S3
            t1 = T1
            t2 = T2
            t3 = T3

        p1[:, i + 1] = s1
        p2[:, i + 1] = s2
        p3[:, i + 1] = s3
        q1[:, i + 1] = t1
        q2[:, i + 1] = t2
        q3[:, i + 1] = t3
    x1 = []
    y1 = []
    z1 = []
    x2 = []
    y2 = []
    z2 = []
    x3 = []
    y3 = []
    z3 = []
    for in_q1 in q1:
        x1.append(in_q1[0])
        y1.append(in_q1[1])
        z1.append(in_q1[2])

    for in_q2 in q2:
        x2.append(in_q2[0])
        y2.append(in_q2[1])
        z2.append(in_q2[2])

    for in_q3 in q3:
        x3.append(in_q3[0])
        y3.append(in_q3[1])
        z3.append(in_q3[2])

    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.plot3D(q1[0, :], q1[1, :], q1[2, :], 'r')
    #ax.plot3D(q2[0, :], q2[1, :], q2[2, :], 'b')
    #ax.plot3D(q3[0, :], q3[1, :], q3[2, :], 'y')
    #plt.legend(['Body1', 'Body2', 'Body3'],
    #           loc='upper right')

    #plt.show()

    figure = Gx.Figure(data=[Gx.Scatter3d(
        x=q1[0, :],
        y=q1[1, :],
        z=q1[2, :],
        mode='lines',
    )])

    figure.add_trace(
        Gx.Scatter3d(
            x=q2[0, :],
            y=q2[1, :],
            z=q2[2, :],
            mode='lines',
        )
    )
    figure.add_trace(
        Gx.Scatter3d(
            x=q3[0, :],
            y=q3[1, :],
            z=q3[2, :],
            mode='lines'
        )
    )

    figure.show()


if __name__ == "__main__":
    # sun
    q1 = np.array([[5.5850 * 10 ** 8, 5.5850 * 10 ** 8, 5.5850 * 10 ** 8]])
    p1 = np.array([[-1.4663, 11.1238, 4.8370]])
    m1 = 1.99 * 10 ** 30

    # mercury
    q2 = np.array([[8.1979 * 10 ** 10, 8.6928 * 10 ** 9, -2.2845 * 10 ** 9]])
    p2 = np.array([[-1.5205 * 10 ** 4, 4.4189 * 10 ** 4, 2.5180 * 10 ** 4]])
    m2 = 3.3 * 10 ** 30

    # venus
    q4 = np.array([[-1.5041 * 10 ** 10, 9.7080 * 10 ** 10, 4.4635 * 10 ** 10]])
    p4 = np.array([[-3.4770 * 10 ** 4, -5.5933 * 10 ** 3, -316.8994]])
    m4 = 4.87 * 10 ** 24

    # earth
    q3 = np.array([[-1.1506 * 10 ** 9, -1.391 * 10 ** 11, -6.033 * 10 ** 10]])
    p3 = np.array([[2.9288 * 10 ** 4, -398.5759, -172.5873]])
    m3 = 5.97 * 10 ** 24

    h = 3200
    N = 8250

    kepler_three_body(q01=q1, q02=q2, q03=q3, p01=p1, p02=p2, p03=p3, m1=m1, m2=m2, m3=m3, h=h, N=N)
