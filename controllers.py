#!/usr/bin/env python
"""
main.py

Created: March 2018
"""

import numpy as np
from dynamics import *
from jacobian import *
from links import *


def inverse_dynamics(links, q, dq, gains, q_des, dq_des=None, ddq_des=None):
    """
    PD inverse dynamics (joint space dynamics) control.

        tau = M(q) ddq + G(q)

    Args:
        links [Link, ...]: List of Link objects (see links.py)
        q             [N]: Numpy array of joint values (radians for revolute joints)
        dq            [N]: Numpy array of joint velocities (rad/s for revolute joints)
        gains      (Dict): PD gains { "kp_inv_dyn": Kp, "kv_inv_dyn": Kv }
        q_des         [N]: Numpy array of desired joint values
        dq_des        [N]: Numpy array of desired joint velocities (default 0)
        ddq_des       [N]: Numpy array of desired joint accelerations (default 0)

    Returns:
        tau [N]: Numpy array of control torques
    """
    dof = len(links)
    if dq_des is None:
        dq_des = np.zeros(dof)
    if ddq_des is None:
        ddq_des = np.zeros(dof)

    # Required gains
    Kp = gains["kp_inv_dyn"]
    Kv = gains["kv_inv_dyn"]

    M = mass_matrix(links, q)
    G = gravity_vector(links, q, g=np.array([0, 0, -9.81])).reshape(dof,1)
    ddq = (ddq_des - Kp*(q-q_des) - Kv*(dq-dq_des)).reshape(dof,1)
    tau = M.dot(ddq)+G
    return tau.reshape(dof)


def nonredundant_operational_space(links, q, dq, gains, x_des, dx_des=np.zeros(3), ddx_des=np.zeros(3), ee_offset=np.zeros(3)):
    """
    Operational Space Control (xyz position) for non-redundant manipulators.

        F = M_x(q) ddx + G_x(q)
        tau = J_v^T F

    Args:
        links [Link, ...]: List of Link objects (see links.py)
        q             [N]: Numpy array of joint values (radians for revolute joints)
        dq            [N]: Numpy array of joint velocities (rad/s for revolute joints)
        gains      (Dict): PD gains { "kp_op_space": Kp, "kv_op_space": Kv }
        x_des         [3]: Numpy array of desired end-effector position
        xq_des        [3]: Numpy array of desired end-effector velocity (default 0)
        xdq_des       [3]: Numpy array of desired end-effector acceleration (default 0)
        ee_offset     [3]: Position of end-effector in the last link frame (default 0)

    Returns:
        tau [N]: Numpy array of control torques
    """
    dof = len(links)
    if dof > 3:
        raise ValueError("nonredundant_operational_space(): len(links) cannot be greater than 3.")

    # Required gains
    Kp = gains["kp_op_space"]
    Kv = gains["kv_op_space"]

    M = mass_matrix(links, q)
    G = gravity_vector(links, q, g=np.array([0, 0, -9.81])) #.reshape(dof,1)
    Jv = linear_jacobian(links, q, ee_offset, link_frame=-1)
    # Mx = ((np.linalg.inv(Jv.T)).dot(M)).dot(np.linalg.inv(Jv))
    # Gx = (np.linalg.inv(Jv.T)).dot(G)
    dx = Jv.dot(dq)
    Ts   = T_all_to_0(links, q)
    EE = Ts[-1]
    x = EE[0:3,3]
    Roe = EE[0:3,0:3]
    xee = x+Roe.dot(ee_offset)
    ddx = (ddx_des-Kp*(xee-x_des)-Kv*(dx-dx_des)) #.reshape(dof,1)
    ddq = (np.linalg.pinv(Jv).dot(ddx)) #.reshape(dof,1)
    tau = (M.dot(ddq)+G) #.reshape(3)
    # F = Mx.dot(ddx)+Gx
    # tau = ((Jv.T).dot(F)).reshape(dof)
    return tau
    


def solve_qp(H, f=None, A=None, b=None, A_eq=None, b_eq=None):
    """
    Solve the QP:

        min 1/2 x^T H x + f^T x
        s.t.
            A    x  <= b
            A_eq x   = b_eq

    Args:
        H        [n x n]: Symmetric positive definite matrix
        f            [n]: Numpy array of size n
        A        [m x n]: Inequality constraint matrix
        b            [m]: Inequality constraint bias
        A_eq  [m_eq x n]: Equality constraint matrix
        b_eq      [m_eq]: Equality constraint bias
    """
    import quadprog

    n = H.shape[0]
    if f is None:
        f = np.zeros(n)
    if A is None and A_eq is None:
        return np.linalg.solve(H, -f)
    if A is None:
        A = np.zeros((0, n))
        b = np.zeros(0)
    if A_eq is None:
        A_eq = np.zeros((0, n))
        b_eq = np.zeros(0)
    m_eq = b_eq.shape[0]

    C = -np.hstack((A_eq.T, A.T))
    d = -np.hstack((b_eq, b))
    return quadprog.solve_qp(H, -f, C, d, m_eq)[0]



def inverse_kinematics(links, q, dq, gains, x_des, ee_offset=np.zeros(3), q_lim=(None, None)):
    """
    Velocity-based inverse kinematics control.

        Desired position will be converted into a desired velocity:

            dx_des = Kp * (x_des - x) / dt

        With joint limits, we solve the quadratic program:

            min || J_v dq_des - dx_des ||^2 + alpha || dq ||^2

            s.t. dq_des >= K_lim * (q_lim_lower - q) / dt
                 dq_des <= K_lim * (q_lim_upper - q) / dt

        Without joint limits, the problem gets reduced to:

            dq_des = J_v^{+} dx_des

        For a torque-controlled robot, the velocity will be converted into a
        torque using inverse dynamics:

            q_des  = q + dq_des * dt
            tau = inverse_dynamics(q_des, dq_des)

    Args:
        links        [Link, ...]: List of Link objects (see links.py)
        q                    [N]: Numpy array of joint values (radians for revolute joints)
        dq                   [N]: Numpy array of joint velocities (rad/s for revolute joints)
        gains             (Dict): IK gains { "kp_ik": Kp, k_joint_lim": K_lim, "ik_regularizer": alpha, "dt": dt }
        x_des                [3]: Numpy array of desired end-effector position
        ee_offset            [3]: Position of end-effector in the last link frame (default 0)
        q_lim (q_lower, q_upper): Tuple of lower and upper joint limits (default (None, None))

    Returns:
        tau [N]: Numpy array of control torques
    """
    dof = len(links)

    # Required gains
    Kp    = gains["kp_inv_kin"]
    K_lim = gains["k_joint_lim"]
    alpha = gains["ik_regularizer"]
    dt    = gains["dt"]

    Jv = linear_jacobian(links, q, ee_offset, link_frame=-1)
    JvT = Jv.T
    I = np.eye(dof)
    H = JvT.dot(Jv)+I*alpha

    Ts = T_all_to_0(links, q)
    EE = Ts[-1]
    x = EE[0:3,3]
    Roe = EE[0:3,0:3]
    xee = x+Roe.dot(ee_offset)
    dx_des = Kp*(x_des-xee)/dt
    f = -(dx_des.T).dot(Jv)

    q_lim_lower = q_lim[0]
    q_lim_upper = q_lim[1]
    if (q_lim_lower==None and q_lim_upper==None):
        dq_des = np.linalg.pinv(Jv).dot(dx_des)
    elif (q_lim_lower==None and q_lim_upper!=None):
        A = -np.eye(dof)
        b = np.array([-K_lim*(q_lim_lower-q)/dt])
        dq_des = solve_qp(H, f, A, b, A_eq=None, b_eq=None)
    elif (q_lim_lower!=None and q_lim_upper==None):
        A = np.eye(dof)
        b = np.array([K_lim*(q_lim_upper-q)/dt])
        dq_des = solve_qp(H, f, A, b, A_eq=None, b_eq=None)
    elif (q_lim_lower!=None and q_lim_upper!=None):
        A = np.vstack((-np.eye(dof),np.eye(dof)))
        b = np.hstack((-K_lim*(q_lim_lower-q)/dt , K_lim*(q_lim_upper-q)/dt))
        dq_des = solve_qp(H, f, A, b, A_eq=None, b_eq=None)

    q_des = q+dq_des*dt
    tau = inverse_dynamics(links, q, dq, gains, q_des, dq_des, ddq_des=None) #keep dq_des as none when we input?
    return tau
    




