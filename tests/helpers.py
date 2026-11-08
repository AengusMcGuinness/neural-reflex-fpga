"""Shared test helpers: rotations, quaternions and a simulated arm bus (no hardware)."""

import math

import numpy as np

from neural_reflex.leader_arm import JOINTS


def rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = math.radians(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def matrix_to_quat(m):
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return [w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)]


class SimBus:
    def __init__(self, positions, limits):
        self.values = {n: {"Present_Position": positions[n], "Min_Position_Limit": limits[n][0],
                           "Max_Position_Limit": limits[n][1], "Torque_Enable": 0, "Operating_Mode": 0,
                           "Present_Voltage": 120, "Present_Temperature": 30, "Torque_Limit": 1000,
                           "Max_Torque_Limit": 1000} for n in JOINTS}
        self.goals = []

    def connect(self):
        pass

    def disconnect(self, *, disable_torque):
        pass

    def read(self, register, name, *, normalize):
        return self.values[name][register]

    def write(self, register, name, value, *, normalize):
        if register == "Goal_Position":
            self.goals.append((name, value))
            if self.values[name]["Torque_Enable"]:
                self.values[name]["Present_Position"] = value

    def enable_torque(self, name):
        self.values[name]["Torque_Enable"] = 1

    def disable_torque(self, name):
        self.values[name]["Torque_Enable"] = 0
