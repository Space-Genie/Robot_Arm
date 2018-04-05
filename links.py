"""
links.py

Created: December 2017
"""

import numpy as np
import dh
from util import *


def prismatic_joints(links):
    """
    Returns vectorized version of joint type indicator epsilon from the course reader.

    Args:
        links [Link, ...]: list of Link objects

    Returns:
        epsilon [n x 1]: Numpy array with element i = 1 if prismatic, 0 if revolute
    """
    return np.array([link.joint_type == Joint.PRISMATIC for link in links], dtype=float)


def revolute_joints(links):
    """
    Return vectorized version of joint type indicator epsilon_bar from the course reader.

    Args:
        links [Link, ...]: list of Link objects

    Returns:
        epsilon_bar [n x 1]: Numpy array with element i = 0 if prismatic, 1 if revolute
    """
    return 1 - prismatic_joints(links)

def T_all_to_prev(links, q):
    """
    List all transformation matrices between the given links.

    Args:
        links [Link, ...]: list of Link objects
        q         [N x 1]: Numpy array of joint values

    Returns:
        [T_1_to_0, T_2_to_1, T_3_to_2, ...]: list of [4 x 4] Numpy arrays
    """
    if len(links) != q.shape[0]:
        raise RuntimeError("links.T_all_to_prev(): Number of links and joints must be the same.")

    return [link.T_to_parent(q_i) for link, q_i in zip(links, q)]

def T_all_to_0(links, q):
    """
    List all transformation matrices to the ground frame.

    Args:
        links [Link, ...]: list of Link objects
        q         [N x 1]: Numpy array of joint values

    Returns:
        [T_1_to_0, T_2_to_0, T_3_to_0, ...]: list of [4 x 4] Numpy arrays
    """
    if len(links) != q.shape[0]:
        raise RuntimeError("links.T_all_to_0(): Number of links and joints must be the same.")

    Ts = T_all_to_prev(links, q)
    for i in range(1, len(Ts)):
        Ts[i] = Ts[i-1].dot(Ts[i])
    return Ts

def T_i_to_j(links, q, i, j):
    """
    Compute the transformation matrix from frame i to j.

    Args:
        links [Link, ...]: list of Link objects
        q         [N x 1]: Numpy array of joint values
        i           (int): index of "from" frame
        j           (int): index of "to" frame

    Returns:
        [4 x 4]: Numpy array
    """
    if len(links) != q.shape[0]:
        raise RuntimeError("links.T_i_to_j(): Number of links and joints must be the same.")
    if i < 0 or i > len(links) or j < 0 or j > len(links):
        raise RuntimeError("links.T_i_to_j(): Frame indices i and j must be between 0 and N.")

    def T_inv(T):
        R = T[:3,:3]
        p = T[:3,3]
        return np.block([
            [R.T, -R.T.dot(p)[:,np.newaxis]],
            [0, 0, 0, 1]
        ])

    if i > j:
        return product(T_all_to_prev(links[j:i], q[j:i]))
    elif i < j:
        return T_inv(product(T_all_to_prev(links[i:j], q[i:j])))
    else:
        return np.eye(4)

class Joint:
    """
    Enum for Joint type.
    """
    REVOLUTE  = 0
    PRISMATIC = 1

class Link:
    """
    Link of an articulated rigid body.

    Args:
        a           (float): DH parameter a_{i-1}
        alpha       (float): DH parameter alpha_{i-1}
        d      (float/None): DH parameter d_i (None for prismatic joints)
        theta  (float/None): DH parameter theta_i (None for revolute joints)
        mass        (float): Link mass
        com         [3 x 1]: Center of mass
        inertia     [3 x 3]: Link inertia at com in the same orientation as frame i

    Example: Revolute joint
        Link(
            a       = 0.1,
            alpha   = np.pi,
            d       = 0.05,
            theta   = None,
            mass    = 1,
            com     = np.array([0,0.05,0])
            inertia = np.eye(3)
        )
    """

    def __init__(self, a=0, alpha=0, d=None, theta=None, mass=0, com=np.zeros(3), inertia=np.zeros((3,3))):
        if d is None and theta is None:
            raise RuntimeError("links.Link(): At least one of d or theta must not be None.")
        if a is None or alpha is None:
            raise RuntimeError("links.Link(): a and alpha cannot be None.")

        self.a     = a
        self.alpha = alpha
        self.d     = d
        self.theta = theta

        if theta is None:
            self.joint_type = Joint.REVOLUTE
        elif d is None:
            self.joint_type = Joint.PRISMATIC
        else:
            self.joint_type = None

        self.mass    = mass
        self.com     = com
        self.inertia = inertia

    def dh(self, q):
        """
        Get a DH entry with the given joint value.

        Args:
            q - joint value (float)

        Returns:
            DH object
        """
        if self.joint_type == Joint.REVOLUTE:
            return dh.DH(self.a, self.alpha, self.d, q)
        elif self.joint_type == Joint.PRISMATIC:
            return dh.DH(self.a, self.alpha, q, self.theta)
        else:
            return dh.DH(self.a, self.alpha, self.d, self.theta)

    def T_to_parent(self, q):
        """
        Get the transformation matrix to the parent link.

        Args:
            q - joint value (float)

        Returns:
            T_i_to_prev - [4 x 4] Numpy array
        """
        return dh.dh_to_T(self.dh(q))

    def __repr__(self):
        joint = "None"
        if self.joint_type == Joint.REVOLUTE:
            joint = "Revolute"
        elif self.joint_type == Joint.PRISMATIC:
            joint = "Prismatic"
        return "Link(joint_type={})".format(joint)

