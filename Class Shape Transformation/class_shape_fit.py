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
import matplotlib.pyplot as plt
from ClassShapeLibrary import ShapeFuncFit, ClassShapeTransformation
from base_airfoil import load_airfoil


def main(filename):
    lower, upper = load_airfoil(filename=filename)
    N1 = 0.5
    N2 = 1.0
    control_points1 = ShapeFuncFit().fit(N1=N1, N2=N2, order=4, chords=lower[:, 0], thickness=lower[:, 1])
    control_points2 = ShapeFuncFit().fit(N1=N1, N2=N2, order=4, chords=upper[:, 0], thickness=upper[:, 1])

    new_lower = ClassShapeTransformation(N1=N1, N2=N2, points=control_points1).thickness(chord=lower[:, 0])
    new_upper = ClassShapeTransformation(N1=N1, N2=N2, points=control_points2).thickness(chord=upper[:, 0])

    plt.figure()
    plt.plot(lower[:, 0], lower[:, 1], '--r', label='Original Low')

    plt.plot(upper[:, 0], upper[:, 1], '--b', label='Original Up')

    plt.plot(lower[:, 0], new_lower, '-r', label='CST Low')
    plt.plot(upper[:, 0], new_upper, '-b', label='CST Up')

    plt.legend(fontsize=16)
    plt.xlabel('chord', fontsize=16)
    plt.ylabel('thickness', fontsize=16)
    plt.ylim(-0.5, 0.5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(filename, fontsize=16)
    plt.tight_layout()
    plt.show()

    # error
    plt.figure()
    plt.plot(lower[:, 0], np.abs(lower[:, 1] - new_lower.squeeze()), '-r', label='Error Low')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('abs error of low', fontsize=16)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(upper[:, 0], np.abs(upper[:, 1] - new_upper.squeeze()), '-b', label='Error Up')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('abs error of up', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(filename='NACA0012')
    main(filename='RAE2822')
    main(filename='NACA6412')
