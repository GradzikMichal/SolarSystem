import numpy as np
import plotly.graph_objects as Gx
import plotly.io as pio

pio.renderers.default = "browser"


def kepler_two_body(q01, q02, p01, p02, m1, m2, h, N):
    G = 6.6743 * 10 ** (-11)

    q1 = q01
    p1 = p01
    q2 = q02
    p2 = p02

    for i in range(N):
        s1 = p1[i]
        s2 = p2[i]
        t1 = q1[i]
        t2 = q2[i]

        for k in range(100):
            T1 = q1[i] + h / 2.0 * (s1 + p1[i])
            T2 = q2[i] + h / 2.0 * (s2 + p2[i])
            a = t1 + q1[i] - t2 - q2[i]
            S1 = p1[i] - 4 * h * G * m2 * a / np.dot(a.T, a) ** (3 / 2.0)
            S2 = p2[i] + 4 * h * G * m1 * a / np.dot(a.T, a) ** (3 / 2.0)
            s1 = S1
            s2 = S2
            t1 = T1
            t2 = T2
        p1 = np.append(p1, np.array([s1]), axis=0)
        p2 = np.append(p2, np.array([s2]), axis=0)
        q1 = np.append(q1, np.array([t1]), axis=0)
        q2 = np.append(q2, np.array([t2]), axis=0)

    x1 = []
    y1 = []
    z1 = []
    x2 = []
    y2 = []
    z2 = []
    for in_q1 in q1:
        x1.append(in_q1[0])
        y1.append(in_q1[1])
        z1.append(in_q1[2])

    for in_q2 in q2:
        x2.append(in_q2[0])
        y2.append(in_q2[1])
        z2.append(in_q2[2])

    figure = Gx.Figure(data=[Gx.Scatter3d(
        x=x1,
        y=y1,
        z=z1,
        mode='lines',
    )])
    figure.add_trace(
        Gx.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
        )
    )
    figure.add_trace(
        Gx.Scatter3d(
            x=x2,
            y=y2,
            z=z2,
            mode='lines',
        )
    )
    figure.show()


if __name__ == "__main__":
    q1 = np.array([[0, 0, 0]])
    q2 = np.array([[1, 0.5, 0]])
    p1 = np.array([[0.1, 0.2, 0]])
    p2 = np.array([[-0.1, 1.2, 0.5]])
    m1 = 1 / 2.6 * 10 ** 11
    m2 = 1.4 * 10 ** 6
    h = 0.01
    N = 2000
    kepler_two_body(q01=q1, q02=q2, p01=p1, p02=p2, m1=m1, m2=m2, h=h, N=N)
