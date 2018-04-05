#!/usr/bin/env python
"""
rotations.py

Created: December 2017
"""

import numpy as np



#####################
# Example functions #
#####################

def scale(scalar):
    """
    Example function to compute a scaling matrix for the given scalar.

    Args:
        scalar (float): amount to scale by
    Returns:
        3x3 numpy array that can be used to scale points accordingly
    """
    return np.array([
        [scalar,      0,      0],
        [0,      scalar,      0],
        [0,           0, scalar]
    ])


def np_scale(scalar):
    """
    Alternative example function to compute a scaling matrix for the given scalar.
    This is just to show it is fine to use numpy methods when implementing below
    functions.

    Args:
        scalar (float): amount to scale by
    Returns:
        3x3 numpy array that can be used to scale points accordingly
    """
    return scalar * np.eye(3)



def rot_x(theta):
    """
    Returns a matrix that rotates vectors by theta radians about the x axis.

    Args:
        theta (float): angle to rotate by in radians
    Returns:
        3x3 numpy array that is a rotation matrix
    """
    return np.array([
        [1,             0,              0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])


def rot_y(theta):
    """
    Returns a matrix that rotates vectors by theta radians about the y axis.

    Args:
        theta (float): angle to rotate by in radians
    Returns:
        3x3 numpy array that is a rotation matrix
    """
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [             0, 1,             0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def rot_z(theta):
    """
    Returns a matrix that rotates vectors by theta radians about the z axis.

    Args:
        theta (float): angle to rotate by in radians
    Returns:
        3x3 numpy array that is a rotation matrix
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [            0,              0, 1]
    ])




def zyx_euler_angles_to_mat(alpha, beta, gamma):
    """
    Converts ZYX Euler angles (rotation about z, then resulting y, then resulting x)
    into a rotation matrix.

    Args:
        alpha (float): angle in radians to rotate about z
        beta  (float): angle in radians to rotate about y
        gamma (float): angle in radians to rotate about x
    Returns:
        3x3 numpy array that is a rotation matrix
    """
    return rot_z(alpha).dot(rot_y(beta)).dot(rot_x(gamma))


def xyz_fixed_angles_to_mat(alpha, beta, gamma):
    """
    Converts XYZ fixed angles (rotation about x, then fixed y, then fixed z)
    into a rotation matrix.

    Args:
        alpha (float): angle in radians to rotate about x
        beta  (float): angle in radians to rotate about y
        gamma (float): angle in radians to rotate about z
    Returns:
        3x3 numpy array that is a rotation matrix
    """
    return rot_z(gamma).dot(rot_y(beta)).dot(rot_x(alpha))


def mat_to_zyx_euler_angles(R):
    """
    Converts rotation matrix into ZYX Euler angles.

    Args:
        R (3x3 numpy array): the rotation matrix
    Returns:
        (alpha, beta, gamma): tuple of floats corresponding to the 3 ZYX Euler angles
    """
    beta   = -np.arctan2(R[2,0], np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0]))
    c_beta = np.cos(beta)
    if abs(c_beta) < 1e-5: # Singularity
        alpha = 0
        R[1,1] = max(-1, min(1, R[1,1]))
        gamma = -sign(np.sin(beta)) * np.arccos(R[1,1])
    else:
        alpha  = np.arctan2(R[1,0]/c_beta, R[0,0]/c_beta)
        gamma  = np.arctan2(R[2,1]/c_beta, R[2,2]/c_beta)
    return alpha, beta, gamma


def sign(x):
    """
    Helper function to determine the sign of x.

    Args:
        x (float)
    Returns:
        sign of x (float)
    """
    return 1 if x >= 0 else -1



def mat_to_quat(R):
    """
    Function to convert a rotation matrix to the corresponding quaternion.

    Args:
        R (3x3 numpy array): the rotation matrix
    Returns:
        An instance of the Quaternion class
    """
    x = 1 + R[0,0] - R[1,1] - R[2,2]
    y = 1 - R[0,0] + R[1,1] - R[2,2]
    z = 1 - R[0,0] - R[1,1] + R[2,2]
    w = 1 + R[0,0] + R[1,1] + R[2,2]
    x = 0.5 * np.sqrt(np.maximum(x, 0))
    y = 0.5 * np.sqrt(np.maximum(y, 0))
    z = 0.5 * np.sqrt(np.maximum(z, 0))
    w = 0.5 * np.sqrt(np.maximum(w, 0))
    x *= sign(R[2,1] - R[1,2])
    y *= sign(R[0,2] - R[2,0])
    z *= sign(R[1,0] - R[0,1])

    return Quaternion(x, y, z, w)


def quat_to_mat(q):
    """
    Function to convert a quaternion to the corresponding rotation matrix

    Args:
        q (Quaternion): an instance of the Quaternion class
    Returns:
        3x3 numpy array that is a rotation matrix
    """
    return np.array([
        [1 - 2*(q.y*q.y + q.z*q.z), 2*(q.x*q.y - q.z*q.w), 2*(q.x*q.z + q.y*q.w)],
        [2*(q.x*q.y + q.z*q.w), 1 - 2*(q.x*q.x + q.z*q.z), 2*(q.y*q.z - q.x*q.w)],
        [2*(q.x*q.z - q.y*q.w), 2*(q.y*q.z + q.x*q.w), 1 - 2*(q.x*q.x + q.y*q.y)]
    ])



##############
# Angle axis #
##############

def angle_axis_to_quat(aa):
    """
    Convert axis angle representation into a quaternion

    Args:
        aa [3 x 1]: Numpy array with angle axis values
                    [x * theta, y * theta, z * theta]
    Returns:
        Quaternion
    """
    theta = np.linalg.norm(aa)
    axis = aa / theta
    return axis_rotation_to_quat(axis, theta)

def quat_to_angle_axis(q):
    """
    Convert quaternion into axis angle representation

    Args:
        q: Quaternion
    Returns:
        [3 x 1]: Numpy array of angle axis values
    """
    theta, axis = quat_to_axis_rotation(q)
    return theta * axis

def axis_rotation_to_quat(axis, theta):
    """
    Construct a quaternion representing a rotation about the given axis.

    Args:
        axis  [3 x 1]: Numpy array for the axis of rotation
        theta (float): angle in radians
    Returns:
        Quaternion
    """
    xyz = axis * np.sin(theta / 2)
    w = np.cos(theta / 2)
    return Quaternion(xyz[0], xyz[1], xyz[2], w)

def quat_to_axis_rotation(q):
    """
    Compute the axis of rotation from the given quaternion. When the angle of
    rotation is 0, the axis defaults to (1, 0, 0).

    Args:
        q: Quaternion
    Returns:
        axis  [3 x 1]: Numpy array for the axis of rotation
        theta (float): angle in radians
    """
    theta = 2 * np.arccos(q.w)
    if theta < 1e-5:
        axis = np.array([1, 0, 0])
    else:
        axis = np.array([q.x, q.y, q.z]) / np.sin(theta / 2)
    return axis, theta


####################
# Quaternion class #
####################

class Quaternion:

    EPSILON = 1e-3

    def __init__(self, x, y, z, w):
        """
        Quaternion class with basic operators

        Args:
            x (float): x * sin(theta/2)
            y (float): y * sin(theta/2)
            z (float): z * sin(theta/2)
            w (float): cos(theta/2)
        """

        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

        # Check quaternion norm
        norm_squared = w*w + x*x + y*y + z*z
        if abs(norm_squared - 1) > self.EPSILON:
            raise ValueError("Norm of quaternion must equal 1. " + \
                             "|| {} ||^2 = {}.".format(self, norm_squared))

    def inv(self):
        """
        Quaternion conjugate.
        """
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def rotate(self, p):
        """
        Rotate point p by quaternion q: q * p * q^(-1).
        """
        if isinstance(p, Quaternion):
            # Point p represented as quaternion
            if abs(p.w) > self.EPSILON:
                raise ValueError("Quaternion to rotate must be a point (w = 0).")
            return self * p * self.inv()

        elif isinstance(p, np.ndarray):
            # Point p represented as numpy array
            p_quat = Quaternion(p[0], p[1], p[2], 0)
            qpq = self * p_quat * self.inv()
            return np.array([qpq.x, qpq.y, qpq.z])

        else:
            # Invalid type
            raise ValueError("Quaternion cannot rotate type: {}".format(type(p)))

    def array(self):
        """
        Return quaternion as numpy array [x, y, z, w]
        """
        return np.array([self.x, self.y, self.z, self.w])

    def __mul__(self, q):
        """
        Left multiplication: self * q
        """
        x = self.w*q.x + self.x*q.w + self.y*q.z - self.z*q.y
        y = self.w*q.y - self.x*q.z + self.y*q.w + self.z*q.x
        z = self.w*q.z + self.x*q.y - self.y*q.x + self.z*q.w
        w = self.w*q.w - self.x*q.x - self.y*q.y - self.z*q.z
        return Quaternion(x, y, z, w)

    def __rmul__(self, p):
        """
        Right multiplication: p * self
        """
        x = p.w*self.x + p.x*self.w + p.y*self.z - p.z*self.y
        y = p.w*self.y - p.x*self.z + p.y*self.w + p.z*self.x
        z = p.w*self.z + p.x*self.y - p.y*self.x + p.z*self.w
        w = p.w*self.w - p.x*self.x - p.y*self.y - p.z*self.z
        return Quaternion(x, y, z, w)

    def __repr__(self):
        return "Quaternion({:.4}, {:.4}, {:.4}; {:.4})".format(self.x, self.y, self.z, self.w)



if __name__ == "__main__":
    """
    Sanity checks for rotations.py

    Add your own sanity checks here. You may change this section however you like.
    The autograder will not run this code.
    """

    # 1b
    assert rot_x(0).shape == (3, 3)
    assert rot_y(0).shape == (3, 3)
    assert rot_z(0).shape == (3, 3)

    # 2b
    assert zyx_euler_angles_to_mat(0, 0, 0).shape == (3, 3)
    assert xyz_fixed_angles_to_mat(0, 0, 0).shape == (3, 3)
    assert len(mat_to_zyx_euler_angles(np.eye(3))) == 3

    # 4b
    assert isinstance(mat_to_quat(np.eye(3)), Quaternion)
