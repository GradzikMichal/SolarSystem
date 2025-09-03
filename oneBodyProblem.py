import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

def f1(n, q, p, h, K, i):
    term1 = n - q[i]
    term2 = h * (p[i] - h * K ** 2 * 2 / (np.dot((n + q[i]).T, (n + q[i]))) ** (3 / 2) * (n + q[i]))
    result = term1 - term2
    return result


def kepler_one_body_midpoint(q0, p0, h, N):
    K = 2  # 2
    eps = 10 ** (-3)

    q = q0
    p = p0
    for i in range(N):
        n0 = q[i]
        f_n1 = f1(n0, q, p, h, K, i)

        err = np.sqrt(np.dot(f_n1.T, f_n1))
        while err > eps:
            J = np.eye(3) + 2 * h ** 2 * K ** 2 * (
                    1 / (np.dot((n0 + q[i]).T, (n0 + q[i]))) ** (3 / 2) * np.eye(3) - 3 * 1 / (
                        np.dot((n0 + q[i]).T, (n0 + q[i]))) ** (5 / 2) * ((n0 + q[i]) * (n0 + q[i]).T))
            f_n1 = f1(n0, q, p, h, K, i)
            x, resid, rank, s = np.linalg.lstsq(J, -f_n1)
            n0 = x + n0
            f_n1 = f1(n0, q, p, h, K, i)
            err = np.sqrt(np.dot(f_n1.T, f_n1))
        q = np.append(q, np.array([n0]), axis=0)
        p_p = p[i] - h * K ** 2 * 4 / np.dot((n0 + q[i]).T, (n0 + q[i])) ** (3 / 2) * (q[i + 1] + q[i])
        p = np.append(p, np.array([p_p]), axis=0)
    return q


if __name__ == "__main__":
    q0 = np.array([[1, 0.5, 0]])
    p0 = np.array([[0, 1, 0.5]])
    q = kepler_one_body_midpoint(q0, p0, 0.005, 20000)
    x = []
    y = []
    z = []
    for in_q in q:
        x.append(in_q[0])
        y.append(in_q[1])
        z.append(in_q[2])

    figure = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines',
    )])
    figure.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
        )
    )
    figure.show()


