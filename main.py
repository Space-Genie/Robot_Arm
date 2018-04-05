#!/usr/bin/env python
"""
main.py

Created: December 2017
"""

import numpy as np
import time
from robot import Robot
from controllers import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# Frequency constants
FREQ_SIMULATION = 1000 # Simulation frequency [Hz]
FREQ_CONTROL    = 200  # Controller frequency [Hz] (should be lower than sim)


def q1_q2_robot():
    robot = Robot(freq_sim=FREQ_SIMULATION, freq_control=FREQ_CONTROL)

    # Add links
    robot.add_revolute(a       = 0,
                        alpha   = 0,
                        d       = 0.0,
                        mass    = 0,
                        com     = np.array([0,0,0]),
                        inertia = np.zeros((3,3)))
    robot.add_revolute(a       = 0,
                        alpha   = np.pi/2.0,
                        d       = 0.0,
                        mass    = 1,
                        com     = np.array([0,-0.1,0]),
                        inertia = np.eye(3))
    robot.add_prismatic(a       = 0,
                        alpha   = np.pi/2.0,
                        theta   = 0,
                        mass    = 1,
                        com     = np.array([0,0,0.1]),
                        inertia = np.eye(3))

    # Add end-effector
    robot.set_ee_offset(np.array([0, 0, 0.2]))  # end-effector offset

    # Set (lower, upper) joint limits
    robot.set_joint_limits(np.array([-np.pi, -np.pi, 0]), np.array([np.pi, np.pi, 2]))

    # Set initial configuration
    robot.simulator.set_configuration(np.zeros(3))

    return robot


def q3_robot():
    robot = Robot(freq_sim=FREQ_SIMULATION, freq_control=FREQ_CONTROL)

    # Add links
    robot.add_revolute (a       = 0,
                        alpha   = 0,
                        d       = 0.0,
                        mass    = 0,
                        com     = np.array([0,0,0]),
                        inertia = np.zeros((3,3)))
    robot.add_revolute (a       = 0.2,
                        alpha   = np.pi/2.0,
                        d       = 0.0,
                        mass    = 1,
                        com     = np.array([0,0,-0.1]),
                        inertia = np.eye(3))
    robot.add_prismatic(a       = 0,
                        alpha   = np.pi/2.0,
                        theta   = 0,
                        mass    = 1,
                        com     = np.array([0,0,0]),
                        inertia = np.zeros((3,3)))

    # Add end-effector
    robot.set_ee_offset(np.array([0, 0, 0.1]))

    # Set (lower, upper) joint limits
    robot.set_joint_limits(np.array([-np.pi, -np.pi, 0]), np.array([np.pi, np.pi, 2]))

    # Set initial configuration
    robot.simulator.set_configuration(np.array([0, np.pi/2, 0]))

    return robot


def q1_trajectory(t):
    return np.array([0, t, 0.05*t])

def q2_trajectory(t):
    return np.array([(0.05*t+0.2)*np.sin(t), 0, -(0.05*t+0.2)*np.cos(t)])

def q3_trajectory(t):
    return np.array([0.4, 0.2*np.sin(np.pi/2*t), 0.03*t])


if __name__ == "__main__":
    # Time constants
    T_SIMULATION = 5  # Simulation duration [s]
    NUM_ITERS = int(T_SIMULATION * FREQ_CONTROL)  # Number of controller iterations

    #################
    ### Set gains ###
    #################

    # TODO: Tune gains
    gains = {
        # Inverse dynamics
        "kp_inv_dyn": 36,
        "kv_inv_dyn": 12,

        # Operational Space
        "kp_op_space": 36,
        "kv_op_space": 12,

        # Inverse kinematics
        "kp_inv_kin": 1,
        "k_joint_lim": 1,
        "ik_regularizer": 0.1,
        "dt": 1. / FREQ_CONTROL,
    }



    ####################
    ### Create robot ###
    ####################

    # TODO: Select robot

    robot = q1_q2_robot()
    # robot = q3_robot()



    # Record data for plotting
    FREQ_PLOTTING = 10
    plotting_data = {
        "q":   [],
        "dq":  [],
        "x":   [],
        "dx":  [],
        "tau": []
    }

    # Run for T_SIMULATION seconds
    t_start = time.time()
    for i in range(NUM_ITERS):
        # Retrieve robot state from simulation
        q, dq = robot.read_sensor_values()
        print(q)
        t_sim = float(i) / FREQ_CONTROL

        ############################
        ### Set desired position ###
        ############################

        # TODO: Select trajectory

        # q_des  = np.array([0., np.pi/2.0, 0])
        # x_des = np.array([0.15, 0.25, 0.35])
        q_des = q1_trajectory(t_sim)
        # x_des = q2_trajectory(t_sim)
        # x_des = q3_trajectory(t_sim)



        ###############################
        ### Compute control torques ###
        ###############################

        # TODO: Select controller

        tau = inverse_dynamics(robot.links, q, dq, gains, q_des)
        print(tau)
        # tau = nonredundant_operational_space(robot.links, q, dq, gains, x_des, ee_offset=robot.ee_offset)
        # tau = inverse_kinematics(robot.links, q, dq, gains, x_des, ee_offset=robot.ee_offset, q_lim=robot.joint_limits)



        # Send command torques and advance one simulation step
        robot.send_command_torques(tau)

        # Any values passed into this function will be displayed in the web interface
        x  = robot.ee_pos(q)
        dx = robot.ee_vel(q, dq)
        robot.publish_values({
            "q":   q,
            "dq":  dq,
            "x":   x,
            "dx":  dx,
            "tau": tau
        })
        # print("q",q)
        # Record data for plotting
        if i % (FREQ_CONTROL // FREQ_PLOTTING) == 0:
            plotting_data["q"].append(q)
            plotting_data["dq"].append(dq)
            plotting_data["x"].append(x)
            plotting_data["dx"].append(dx)
            plotting_data["tau"].append(tau)

        # Slow down simulation to real time
        t_next = t_start + (i + 1.) / FREQ_CONTROL
        t_real = time.time()
        if t_real < t_next:
            time.sleep(t_next - t_real)

    print("Simulated {0:f}s in {1}s.".format(T_SIMULATION, time.time() - t_start))



    #########################
    ### Plot trajectories ###
    #########################

    for key in plotting_data:
        plotting_data[key] = np.column_stack(plotting_data[key])

    # TODO: Generate plots. An example plot is given to you below.

    plt.figure()

    plt.title("Joint Position Trajectory") #switch q to x for position
    plt.ylabel("q(t)")
    plt.xlabel("time [s]")
    for i in range(robot.dof()):
        plt.plot(plotting_data["q"][i,:])
    plt.legend(["q_" + str(i+1) for i in range(robot.dof())])
    idx_sec = range(0, T_SIMULATION * FREQ_PLOTTING + 1, FREQ_PLOTTING)
    plt.xticks(idx_sec, [i for i in range(len(idx_sec))])

    plt.show()
