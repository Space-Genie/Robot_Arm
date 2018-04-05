#!/usr/bin/env python
"""
jacobian.py

Created: December 2017
"""

import numpy as np
from links import *

def linear_jacobian(links, q, pos_in_link=np.zeros(3), link_frame=-1):
    """
    Computes the linear Jacobian J_v for the current configuration q.

    This Jacobian relates joint velocities to linear velocities at a specified
    point on the robot. The location of this point is characterized by the robot
    link it is attached to (link_frame) and the position of the point in that
    link (pos_in_link).

    Args:
        links   [Link, ...]: list of Link objects (see links.py)
        q           [N x 1]: Numpy array of joint values (radians for revolute joints)
        pos_in_link [3 x 1]: Numpy array for the position of the Jacobian point in link_frame
        link_frame    (int): frame index of the Jacobian point (-1 for the last link)

    Returns:
        [3 x N] Numpy array
    """
    # Make negative link_frame indices positive
    if link_frame < 0:
        link_frame += len(links)
    # Check parameters
    if len(links) != q.shape[0]:
        raise ValueError("Number of links (%d) and joints q (%d) must be equal." % (len(links), q.shape[0]))
    if pos_in_link.shape[0] != 3:
        raise ValueError("Position pos_in_link must be of dimension 3.")
    if link_frame < 0 or link_frame >= len(links):
        raise ValueError("Link frame index link_frame must be in the range [0, N-1].")

    dof = q.shape[0]

    Ts   = T_all_to_0(links[:link_frame+1], q[:link_frame+1])
    Z    = np.vstack([T_i_to_0[:3,2] for T_i_to_0 in Ts])
    P_0i = np.vstack([T_i_to_0[:3,3] for T_i_to_0 in Ts])
    p_0n = (P_0i[-1] + Ts[-1][:3,:3].dot(pos_in_link))[np.newaxis,:]
    P_in = p_0n - P_0i
    Z_x_P = np.cross(Z, P_in, axisa=1, axisb=1, axisc=0)
    eps   = prismatic_joints(links)[np.newaxis,:link_frame+1]

    J_v   = np.zeros((3, dof))
    J_v[:,:link_frame+1] = eps * Z.T + (1 - eps) * Z_x_P

    return J_v


def angular_jacobian(links, q, link_frame=-1):
    """
    Computes the angular Jacobian J_w for the current configuration q.

    This Jacobian relates joint velocities to angular velocities at a specified
    point on the robot. The location of this point is characterized by the robot
    link it is attached to (link_frame). The position of the point in the link
    does not matter.

    Args:
        links [Link, ...]: list of Link objects (see links.py)
        q         [N x 1]: Numpy array of joint values (radians for revolute joints)
        link_frame  (int): frame index of Jacobian point (-1 for the last link)

    Returns:
        [3 x N] Numpy array
    """
    # Make negative link_frame indices positive
    if link_frame < 0:
        link_frame += len(links)
    # Check parameters
    if len(links) != q.shape[0]:
        raise ValueError("Number of links (%d) and joints q (%d) must be equal." % (len(links), q.shape[0]))
    if link_frame < 0 or link_frame >= len(links):
        raise ValueError("Link frame index link_frame must be in the range [0, N-1].")

    dof = q.shape[0]

    Ts  = T_all_to_0(links, q)[:link_frame+1]
    Z   = np.column_stack([T_i_to_0[:3,2] for T_i_to_0 in Ts])
    eps_bar = revolute_joints(links)[np.newaxis,:link_frame+1]

    J_w = np.zeros((3, dof))
    J_w[:,:link_frame+1] = eps_bar * Z

    return J_w


def basic_jacobian(links, q, pos_in_link=np.zeros(3), link_frame=-1):
    """
    Computes the basic Jacobian J_0 = [J_v; J_w] for the current configuration q.

    Args:
        links   [Link, ...]: list of Link objects (see links.py)
        q           [N x 1]: Numpy array of joint values (radians for revolute joints)
        pos_in_link [3 x 1]: Numpy array for the position of the Jacobian point in link_frame
        link_frame    (int): frame index of the Jacobian point (-1 for the last link)

    Returns:
        [6 x N] Numpy array
    """
    return np.vstack([linear_jacobian (links, q, pos_in_link, link_frame),
                      angular_jacobian(links, q, link_frame)])


if __name__ == "__main__":

    # RPR Manipulator from Q1
    l1 = Link(a=0, alpha=0,        d=0,    theta=None)
    l2 = Link(a=0, alpha=-np.pi/2, d=None, theta=0)
    l3 = Link(a=0, alpha= np.pi/2, d=1,    theta=None)

    links       = [l1, l2, l3]
    q           = np.array([np.pi/2, 0.1, np.pi/2])
    pos_in_link = np.zeros((3,))
    link_frame  = len(links) - 1

    J = basic_jacobian(links, q, pos_in_link, link_frame)
    print(J)
