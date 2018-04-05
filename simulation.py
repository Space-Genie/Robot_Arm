"""
simulation.py

Created: December 2017
"""

import numpy as np

def runge_kutta(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

class Simulator:

    def __init__(self, rbdl, freq_sim=1000, freq_control=100, asynchronous=True):
        self.rbdl = rbdl
        self.dt = 1 / freq_sim
        self.num_integration_steps = int(freq_sim / freq_control)
        self.q   = None
        self.dq  = None
        self.tau = None
        self.asynchronous = asynchronous
        self.t = 0
        # TODO: asynchronous

    def set_configuration(self, q=None, dq=None):
        if q is None:
            q = np.zeros(self.rbdl.dof)
        if dq is None:
            dq = np.zeros(self.rbdl.dof)
        self.q  = q
        self.dq = dq

    def get_configuration(self):
        if self.q is None:
            self.set_configuration()
        # TODO: asynchronous
        return self.q, self.dq

    def set_tau(self, tau):
        self.tau = tau
        if not self.asynchronous:
            for _ in range(self.num_integration_steps):
                self.integrate()

    def integrate(self):
        if self.q is None:
            self.set_configuration()
        if self.tau is None:
            self.tau = np.zeros(self.rbdl.dof)

        # Forward dynamics
        ddq = self.rbdl.forward_dynamics(self.q, self.dq, self.tau)

        # ODE setup
        dof = self.rbdl.dof
        f = lambda t, x: np.concatenate([x[dof:], ddq])
        # f = lambda t, y: np.concatenate([y[dof:], self.rbdl.forward_dynamics(y[:dof], y[dof:], self.tau)])
        y = np.concatenate([self.q, self.dq])

        # Runge Kutta integration
        y += runge_kutta(f, 0, y, self.dt)
        self.t += self.dt
        self.q  = y[:dof]
        self.dq = y[dof:]
