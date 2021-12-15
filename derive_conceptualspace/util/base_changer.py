from dataclasses import dataclass

import numpy as np

#all thanks to https://stackoverflow.com/a/69407977
#see also https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d

class NDPlane:
    def __init__(self, coef, intercept):
        assert coef.ndim == 1
        assert isinstance(intercept, (int, float))
        self.coef = np.array(coef)
        self.intercept = intercept

    @property
    def normal(self):
        return self.coef

    def __contains__(self, point: np.array):
        assert point.shape == self.coef.shape
        return np.isclose(np.sum(self.coef*point)+self.intercept, 0)

    def project(self, point: np.array):
        k = (np.sum(self.coef*point)+self.intercept) / (self.coef**2).sum()
        return point-(k*self.coef)

    def z(self, point: np.array):
        assert point.shape[0] == self.coef.shape[0]-1
        return (- self.intercept -np.sum(self.coef[:-1]*point)) / self.coef[-1]

class ThreeDPlane(NDPlane):
    a = property(lambda self: self.coef[0])
    b = property(lambda self: self.coef[1])
    c = property(lambda self: self.coef[2])
    d = property(lambda self: self.intercept)


def make_base_changer(plane: ThreeDPlane):
    uvec1 = plane.normal
    uvec2 = [0, -plane.d / plane.b, plane.d / plane.c]  # NOT [1, 0, plane.z(model, 1, 0)] !!
    uvec3 = np.cross(uvec1, uvec2)
    transition_matrix = np.linalg.inv(np.array([uvec1, uvec2, uvec3]).T)

    origin = np.array([0, 0, 0])
    new_origin = plane.project(origin)
    forward = lambda point: transition_matrix.dot(point - new_origin)
    backward = lambda point: np.linalg.inv(transition_matrix).dot(point) + new_origin
    return forward, backward


#VARIANT FOR EXACTLY 3 DIMENSIONS:
# @dataclass
# class Plane:
#     a: float
#     b: float
#     c: float
#     d: float
#
#     @property
#     def normal(self):
#         return np.array([self.a, self.b, self.c])
#
#     def __contains__(self, point: np.array):
#         return np.isclose(self.a * point[0] + self.b * point[1] + self.c * point[2] + self.d, 0)
#
#     def project(self, point):
#         x, y, z = point
#         k = (self.a * x + self.b * y + self.c * z + self.d) / (self.a ** 2 + self.b ** 2 + self.c ** 2)
#         return np.array([x - k * self.a, y - k * self.b, z - k * self.c])
#
#     def z(self, x, y):
#         return (- self.d - self.b * y - self.a * x) / self.c
#
# def make_base_changer_3d(plane):
#     uvec1 = plane.normal
#     uvec2 = [0, -plane.d / plane.b, plane.d / plane.c]  # NOT [1, 0, plane.z(model, 1, 0)] !!
#     uvec3 = np.cross(uvec1, uvec2)
#     transition_matrix = np.linalg.inv(np.array([uvec1, uvec2, uvec3]).T)
#
#     origin = np.array([0, 0, 0])
#     new_origin = plane.project(origin)
#     forward = lambda point: transition_matrix.dot(point - new_origin)
#     backward = lambda point: np.linalg.inv(transition_matrix).dot(point) + new_origin
#     return forward, backward
