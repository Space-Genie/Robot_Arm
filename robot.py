"""
robot.py

Created: December 2017
"""

import redis
import json
import numpy as np
import threading

import links
import jacobian
import dynamics
from simulation import Simulator

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "rbdl_python")))
from rbdl import Rbdl

class Robot:

    def __init__(self, host="localhost", port=6379, db=0, freq_sim=1000, freq_control=100):
        self.links = ()
        self.rbdl  = Rbdl()
        self.simulator = Simulator(self.rbdl, freq_sim=freq_sim, freq_control=freq_control, asynchronous=False)
        self.redis_db = redis.Redis(host=host, port=port, db=db, decode_responses=True)

        self.publish_cv = threading.Condition()
        self.publish_queue = None
        self.publisher = threading.Thread(target=self.publisher_thread)
        self.publisher.daemon = True
        self.publisher.start()

        self.set_gravity()
        self.set_ee_offset()
        self.set_joint_limits()

    def dof(self):
        return self.rbdl.dof

    def add_link(self, a=0, alpha=0, d=None, theta=None, mass=0, com=np.zeros(3), inertia=np.zeros((3,3))):
        self.links = self.links + (links.Link(a, alpha, d, theta, mass, com, inertia),)

        T_to_parent = self.links[-1].T_to_parent(0)
        rotation = T_to_parent[:3,:3]
        translation = T_to_parent[:3,3]

        if theta is None:
            joint_type = Rbdl.JointTypeRevolute
        elif d is None:
            joint_type = Rbdl.JointTypePrismatic
        joint_axis = np.array([0, 0, 1])

        com     = self.links[-1].com
        inertia = self.links[-1].inertia

        self.rbdl.add_link(rotation, translation, joint_type, joint_axis, mass, com, inertia)

    def add_revolute(self, a=0, alpha=0, d=0, mass=0, com=np.zeros(3), inertia=np.zeros((3,3))):
        self.add_link(a, alpha, d, None, mass, com, inertia)

    def add_prismatic(self, a=0, alpha=0, theta=0, mass=0, com=np.zeros(3), inertia=np.zeros((3,3))):
        self.add_link(a, alpha, None, theta, mass, com, inertia)

    def set_ee_offset(self, offset=np.zeros(3)):
        self.ee_offset = offset

    def set_joint_limits(self, q_lim_lower=None, q_lim_upper=None):
        self.joint_limits = (q_lim_lower, q_lim_upper)

    def set_gravity(self, g=np.array([0, 0, -9.81])):
        self.rbdl.set_gravity(g[0], g[1], g[2])
        self.gravity = g

    def ee_pos(self, q):
        T_ee_to_0 = links.T_i_to_j(self.links, q, len(self.links), 0)
        return T_ee_to_0[:3,3] + T_ee_to_0[:3,:3].dot(self.ee_offset)

    def ee_vel(self, q, dq):
        return jacobian.linear_jacobian(self.links, q, pos_in_link=self.ee_offset).dot(dq)

    # def linear_jacobian(self, q, pos_in_link=np.zeros(3), link=-1):
    #     return jacobian.linear_jacobian(self.links, q, pos_in_link, link)

    # def angular_jacobian(self, q):
    #     return jacobian.angular_jacobian(self.links, q)

    # def basic_jacobian(self, q, pos_in_link=np.zeros(3), link=-1):
    #     return jacobian.basic_jacobian(self.links, q, pos_in_link, link)

    # def mass_matrix(self, q):
    #     return dynamics.mass_matrix(self.links, q)

    # def gravity_vector(self, q):
    #     return dynamics.gravity_vector(self.links, q, self.gravity)

    def read_sensor_values(self):
        return self.simulator.get_configuration()

    def send_command_torques(self, tau):
        self.simulator.set_tau(tau)

    def publisher_thread(self):
        import rotations

        KEY_PREFIX = "cs223a::simulator::"
        def stringify(arr):
            try:
                if len(arr.shape) == 1:
                    return " ".join(map(str, arr.tolist()))
                else:
                    return "; ".join([" ".join(map(str, row.tolist())) for row in arr])
            except:
                return str(arr)

        while True:
            # Wait until a value is ready
            with self.publish_cv:
                while self.publish_queue is None:
                    self.publish_cv.wait()
                values = self.publish_queue
                self.publish_queue = None

            # Publish value
            values_to_send = {}
            for key, val in values.items():
                if key == "robot":
                    raise ValueError("Cannot publish value with key: robot.")
                values_to_send[KEY_PREFIX + key] = stringify(val)

            Ts = links.T_all_to_0(self.links, values["q"])
            quaternions = [rotations.mat_to_quat(T[:3,:3]).array() for T in Ts]
            positions = [T[:3,3] for T in Ts]
            joint_types = links.prismatic_joints(self.links)
            json_robot = {
                "links": [{"quat": q.tolist(),
                           "pos" : p.tolist(),
                           "type": t} \
                          for q, p, t in zip(quaternions, positions, joint_types)],
                "ee": {
                    "pos": self.ee_offset.tolist()
                }
            }
            values_to_send[KEY_PREFIX + "robot"] = json.dumps(json_robot)

            self.redis_db.mset(values_to_send)

    def publish_values(self, values):
        with self.publish_cv:
            self.publish_queue = values
            self.publish_cv.notify()
