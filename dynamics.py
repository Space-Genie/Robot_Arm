#!/usr/bin/env python
"""
dynamics.py


Created: December 2017
"""

import numpy as np
from links import *
from jacobian import *


def mass_matrix(links, q):
    """
    Computes the mass matrix for the current configuration q.

    Args:
        links [Link, ...]: list of Link objects
        q         [N x 1]: Numpy array of joint values

    Returns:
        [N x N]: Numpy array
    """
    # Check parameters
    if len(links) != q.shape[0]:
        raise ValueError("Number of links (%d) and joints q (%d) must be equal." % (len(links), q.shape[0]))

    dof = q.shape[0]
    M = np.zeros((dof, dof))
    Ts_i_to_0 = T_all_to_0(links, q)
    for i in range(dof):
        J_vi = linear_jacobian(links, q, pos_in_link=links[i].com, link_frame=i)
        R_0_to_i = Ts_i_to_0[i][:3,:3].T
        J_wi = R_0_to_i.dot(angular_jacobian(links, q, link_frame=i))
        M += links[i].mass * J_vi.T.dot(J_vi) + J_wi.T.dot(links[i].inertia).dot(J_wi)
    return M


def gravity_vector(links, q, g=np.array([0, 0, -9.81])):
    """
    Computes the mass matrix for the current configuration q.

    Args:
        links [Link, ...]: list of Link objects
        q         [N x 1]: Numpy array of joint values
        g         [1 x 3]: Numpy array for setting gravity.

    Returns:
        [N x 1]: Numpy array
    """
    # Check parameters
    if len(links) != q.shape[0]:
        raise ValueError("Number of links (%d) and joints q (%d) must be equal." % (len(links), q.shape[0]))

    dof = q.shape[0]
    G = np.zeros(dof)
    for i in range(dof):
        J_vi = linear_jacobian(links, q, pos_in_link=links[i].com, link_frame=i)
        G -= links[i].mass * J_vi.T.dot(g)
    return G


if __name__ == "__main__":

    # Mass, center of mass, and inertia of link 1
    m1 = 1
    C1 = np.array([0, 0, 0])
    I1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Mass, center of mass, and inertia of link 2
    m2 = 1
    C2 = np.array([0, 0, 0])
    I2 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Create likks array
    l1 = Link(a=0, alpha=0,        d=0,    theta=None, mass=m1, com=C1, inertia=I1)
    l2 = Link(a=0, alpha=-np.pi/2, d=None, theta=0,    mass=m2, com=C1, inertia=I2)
    links = [l1, l2]

    # Set robot configuration and direction of gravity
    q = np.array([0, 0.1])
    g = np.array([-9.81, 0, 0])

    # Compute dynamic quantities
    M = mass_matrix(links, q)
    G = gravity_vector(links, q, g)

    print(M)
    print(G)