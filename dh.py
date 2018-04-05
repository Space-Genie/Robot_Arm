#!/usr/bin/env python
"""
dh.py


Created: December 2017
"""

import numpy as np

class DH:
    """
    Entry in a DH table.
    """

    def __init__(self, a, alpha, d, theta):
        self.a     = float(a)
        self.alpha = float(alpha)
        self.d     = float(d)
        self.theta = float(theta)

    def __repr__(self):
        return "DH(a={}, alpha={}, d={}, theta={})".format(self.a, self.alpha, self.d, self.theta)



############################################

def dh_to_T(dh):
    """
    Computes the transformation matrix for the given dh row.

    Args:
        dh (DH object): ith row of the DH table

    Returns:
        T_i_to_prev [4 x 4]: Numpy array
    """
    c_theta = np.cos(dh.theta)
    s_theta = np.sin(dh.theta)
    c_alpha = np.cos(dh.alpha)
    s_alpha = np.sin(dh.alpha)

    # Less readable but faster
    T = np.zeros((4,4))
    T[0,0] = c_theta
    T[0,1] = -s_theta
    T[0,3] = dh.a
    T[1,0] = s_theta * c_alpha
    T[1,1] = c_theta * c_alpha
    T[1,2] = -s_alpha
    T[1,3] = -s_alpha * dh.d
    T[2,0] = s_theta * s_alpha
    T[2,1] = c_theta * s_alpha
    T[2,2] = c_alpha
    T[2,3] = c_alpha * dh.d
    T[3,3] = 1
    return T

    # return np.array((
    #     (          c_theta,          -s_theta,        0,            dh.a),
    #     (s_theta * c_alpha, c_theta * c_alpha, -s_alpha, -s_alpha * dh.d),
    #     (s_theta * s_alpha, c_theta * s_alpha,  c_alpha,  c_alpha * dh.d),
    #     (                0,                 0,        0,               1)
    # ))



if __name__ == "__main__":
    """
    Sanity checks for dh.py

    Add your own sanity checks here. You may change this section however you like.
    The autograder will not run this code.
    """

    # 2c
    assert dh_to_T(DH(0.1,0.1,0.1,0.1)).shape == (4, 4)
    assert abs(np.linalg.det(dh_to_T(DH(0.1,-0.1,0.1,0.1))[0:3,0:3]) - 1) < 1e-3

