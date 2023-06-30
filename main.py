import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


def EqOfMovement(y, t, m, l1, l2, k, g):
    dy = np.zeros_like(y)
    dy[0] = y[2]
    dy[1] = y[3]
    a11 = 2 * l1
    a12 = l2 * np.cos(y[1] - y[0])
    b1 = -np.sin(y[0]) * (2 * g + k * l1 * np.cos(y[0]) / m) + l2 * y[3] ** 2 * np.sin(y[1] - y[0])
    a21 = l1 * np.cos(y[1] - y[0])
    a22 = l2
    b2 = -g * np.sin(y[1]) - l1 * y[2] ** 2 * np.sin(y[1] - y[0])
    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a21 * a12)
    dy[3] = (a11 * b2 - a21 * b1) / (a11 * a22 - a21 * a12)
    return dy


m = 1
l1 = 0.3
l2 = 0.1
k = 10
g = 10
phi0 = 30
psi0 = 0
dphi0 = 0
dpsi0 = 0
y0 = [phi0, psi0, dphi0, dpsi0]
l0 = 0.5
n = 10
t = np.linspace(0, 7, 1000)
y = odeint(EqOfMovement, y0, t, (m, l1, l2, k, g))
phi = y[:, 0]
psi = y[:, 1]
dphi = y[:, 2]
dpsi = y[:, 3]
h = 0.05

Xm1 = l0 + l1 * np.sin(phi)
Ym1 = l2 + l1 * (1 - np.cos(phi))
Xm2 = Xm1 + l2 * np.sin(psi)
Ym2 = Ym1 - l2 * np.cos(psi)
Xo = l0
Yo = l2 + l1
Xp = np.linspace(0, 1, 2 * n + 1)
Yp = np.zeros(2 * n + 1)
ss = 0
for i in range(len(Yp)):
    Yp[i] = np.sin(ss) / 25
    ss += np.pi / 2
Xd = 0
Yd = Ym1
fig0 = plt.figure(figsize=[13, 9])
ax1 = fig0.add_subplot(2, 2, 1)
ax1.plot(t, phi, color=[1, 0, 0])
ax1.set_title('Phi(t)')

ax2 = fig0.add_subplot(2, 2, 2)
ax2.plot(t, psi, color=[0, 1, 0])
ax2.set_title('Psi(t)')

ax3 = fig0.add_subplot(2, 2, 3)
ax3.plot(t, dphi, color=[0, 0, 1])
ax3.set_title("Phi'(t)")

ax4 = fig0.add_subplot(2, 2, 4)
ax4.plot(t, dpsi, color=[0, 0, 0])
ax4.set_title("Psi'(t)")

fig = plt.figure(figsize=[60, 20])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[- l0, l1 + l2 + l0], ylim=[0, Yo + l1 + l2])
ax.arrow(Xo, Yo, 0, -l1, width=0.01)
ax.arrow(Xo, Yo, l1, 0, width=0.01)
M1O = ax.plot([Xm1[0], Xo], [Ym1[0], Yo], color=[0, 0, 0])[0]
M1M2 = ax.plot([Xm1[0], Xm2[0]], [Ym1[0], Ym2[0]], color=[0, 0, 0])[0]
triangle = ax.plot([Xo - 0.05, Xo + 0.05, Xo, Xo - 0.05], [Yo + 0.05, Yo + 0.05, Yo, Yo + 0.05],
                   color=[0, 0, 0])[0]
Pruzzhina, = ax.plot(Xp * Xm1[0], (Yp + Ym1[0]), color=[0, 0, 0])
M1 = ax.plot(Xm1[0], Ym1[0], 'o', color=[1, 0, 0])[0]
M2 = ax.plot(Xm2[0], Ym2[0], 'o', color=[1, 0, 0])[0]
O = ax.plot(Xo, Yo, 'o', color=[1, 0, 0])[0]
D = ax.plot(Xd, Yd[0], 's', color=[0, 0, 0])[0]

ax.plot([-0.045, -0.045], [0, Yo + l1 + l2], color=[0, 0, 0])
ax.plot([0.05, 0.05], [0, Yo + l1 + l2], color=[0, 0, 0])


def kadr(i):
    M1.set_data(Xm1[i], Ym1[i])
    M2.set_data(Xm2[i], Ym2[i])
    M1M2.set_data([Xm1[i], Xm2[i]], [Ym1[i], Ym2[i]])
    D.set_data(Xd, Yd[i])
    O.set_data(Xo, Yo)
    M1O.set_data([Xm1[i], Xo], [Ym1[i], Yo])
    Pruzzhina.set_data(Xp * Xm1[i], Yp + Ym1[i])
    return [M1, M2, M1M2, O, M1O, Pruzzhina, D]


kino = FuncAnimation(fig, kadr, interval=t[1] - t[0], frames=len(t))
plt.show()
