import numpy as np
import matplotlib.pyplot as plt
from ClassShapeLibrary import *


def demonstrate_class_function():
    colors = ['b', 'orange', 'g', 'red', 'purple', 'brown', 'm']
    N1 = [0.5, 0.5, 1.0, 0.75, 0.75, 1.0, 0.001]
    N2 = [1.0, 0.5, 1.0, 0.75, 0.25, 0.001, 0.001]
    chord = np.linspace(0, 1, 100)
    plt.figure()
    for i in range(len(N1)):
        class_func = ClassFunction(N1[i], N2[i])
        thickness = class_func.thickness(chord=chord)
        plt.plot(thickness, '--', label=f'N1:{N1[i]}, N2:{N2[i]}', color=colors[i])
        plt.plot(-thickness, '--', color=colors[i])
    plt.legend()
    plt.show()

    N1 = 0.5
    N2 = 1.0
    plt.figure()
    class_func = ClassFunction(N1, N2)
    thickness = class_func.thickness(chord=chord)
    plt.plot(thickness, label='Upper', color='red')
    plt.plot(-thickness, label=f'Lower', color='blue')
    plt.xlabel('chord')
    plt.ylabel('thickness')
    plt.show()


def demonstrate_BernsteinPoly():
    order = 5
    polyset = [BernsteinPoly(order=order, index=i) for i in range(6)]
    chord = np.linspace(0, 1, 100).reshape(-1, 1)
    plt.figure()
    for i in range(6):
        label = f'$B_({i, 5})$'
        values = polyset[i].values(chords=chord)
        plt.plot(values, label=label)
    plt.legend()
    plt.tight_layout()
    plt.show()


def demonstrate_cs_function():
    base_airfoil = np.array([1, 1, 1, 1, 1, 1]).reshape(-1, 1)
    transformation = np.array([1, 2, 0.5, 2, 2, 0.5, 1]).reshape(-1, 1)
    N1 = 0.5
    N2 = 1
    chord = np.linspace(0, 1, 100)
    thickness = ClassShapeTransformation(N1=N1, N2=N2, points=base_airfoil).thickness(chord=chord)
    t_thickness = ClassShapeTransformation(N1=N1, N2=N2, points=transformation).thickness(chord=chord)
    plt.figure()
    plt.plot(thickness, '--', color='black', label='baseline')
    plt.plot(-thickness, '--', color='black')
    plt.plot(t_thickness, '--', color='red', label='deformed airfoil')
    plt.plot(-t_thickness, '--', color='red')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demonstrate_class_function()
    demonstrate_BernsteinPoly()
    demonstrate_cs_function()
