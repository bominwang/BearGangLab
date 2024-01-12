"""
------------------------------------------------------------------------------------------------------------------------
BearGangLab
------------------------------------------------------------------------------------------------------------------------
CST method
------------------------------------------------------------------------------------------------------------------------
The source code was written by BoMin Wang
Beijing institute of technology, Beijing, Republic People of CHINA
------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np


class ClassFunction(object):
    def __init__(self, N1, N2):
        self.N1 = N1
        self.N2 = N2

    def thickness(self, chord):
        """
        :param chord: [0, 1] & [number of discrete point * batch size]
        :return: thickness [number of discrete point * batch size]
        """
        return np.power(chord, self.N1) * np.power(1 - chord, self.N2)


def customized_factorial(x):
    unit = x
    while unit > 1:
        unit = unit - 1
        x = unit * x
    return 1 if x == 0 else x


class BernsteinPoly(object):
    def __init__(self, order, index):
        """
        :param order:
        :param index:
        """
        self.order = order
        self.index = index

    def values(self, chords):
        """
        :param chords: [num_points, 1]
        :return:
        """
        constant = customized_factorial(self.order) / (
                customized_factorial(self.index) * customized_factorial(self.order - self.index))
        return constant * np.power(chords, self.index) * np.power(1 - chords, self.order - self.index)


class ShapeFunction(object):
    def __init__(self, points):
        """
        :param points: points
        The number of control points is determined by the order of the Bernstein polynomial.
        The order of the Bernstein curve is represented by the variable order,
        and the index range for each control point typically spans from 0 to order.
        """
        self.points = points
        self.num_points = self.points.shape[0]
        self.order = self.num_points - 1
        self.bernstein_polynomials = None

    def values(self, chords):
        self.bernstein_polynomials = [BernsteinPoly(self.order, i) for i in range(self.num_points)]
        curve_points = np.zeros_like(chords)
        for i, bernstein_poly in enumerate(self.bernstein_polynomials):
            curve_points += bernstein_poly.values(chords) * self.points[i]
        return curve_points


class ClassShapeTransformation(object):
    def __init__(self, N1, N2, points):
        """
        :param N1:
        :param N2:
        :param points:
        """
        self.N1 = N1
        self.N2 = N2
        self.points = points

    def thickness(self, chord):
        class_thickness = ClassFunction(N1=self.N1, N2=self.N2).thickness(chord=chord)
        shape_thickness = ShapeFunction(points=self.points).values(chords=chord)
        thickness = class_thickness * shape_thickness
        return thickness.reshape(-1, 1)


class ShapeFuncFit(object):
    def __init__(self, doctrine=None):
        """
        :param doctrine: if doctrine is True, the start and end of shape function can be setup
        """
        self.doctrine = doctrine
        self.order = None
        self.num_points = None
        self.control_points = None

    def fit(self, N1, N2, order, chords, thickness):
        # 计算class function定义的基准曲线厚度
        class_func = ClassFunction(N1=N1, N2=N2).thickness(chord=chords).reshape(-1, 1)
        # 计算未加权Bernstein多项式
        self.order = order
        self.num_points = self.order + 1
        bernstein_polynomials = [BernsteinPoly(self.order, i) for i in range(self.num_points)]
        shape_func = np.empty(shape=[len(chords), self.num_points])
        for i, bernstein_poly in enumerate(bernstein_polynomials):
            shape_func[:, i] = bernstein_poly.values(chords)

        # delta = thickness[-1]

        # 最小二乘
        matrix = class_func * shape_func
        self.control_points, _, _, _ = np.linalg.lstsq(matrix, thickness.reshape(-1, 1), rcond=None)
        return self.control_points






