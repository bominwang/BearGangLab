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


def calculate_beta(chord, thickness):
    """
    :param chord: positive direction 0->1
    :param thickness: direction with chord
    :return:
    """
    num = len(chord)
    return None


def load_airfoil(filename, flag=False):
    if filename == 'NACA0012':
        lip_radii = 2.32e-3  # leading edge radii
        lower = np.loadtxt('NACA0012LOWER.txt')
        upper = np.loadtxt('NACA0012UPPER.txt')
        if flag:
            plt.figure()
            plt.plot(lower[:, 0], lower[:, 1], '-r', label='LOW')
            indices = np.arange(0, lower.shape[0], 5)
            plt.plot(lower[indices, 0], lower[indices, 1], 'r^', markersize=4)
            plt.plot(upper[:, 0], upper[:, 1], '-b', label='UP')
            indices = np.arange(0, upper.shape[0], 5)
            plt.plot(upper[indices, 0], upper[indices, 1], 'b^', markersize=4)
            plt.legend(fontsize=16)
            plt.xlabel('chord', fontsize=16)
            plt.ylim(-0.5, 0.5)
            plt.ylabel('thickness', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title('NACA0012', fontsize=16)
            plt.tight_layout()
            plt.show()
        return lower, upper

    elif filename == 'NACA6412':
        lip_radii = 2.49e-3
        lower = np.loadtxt('NACA6412LOWER.txt')
        upper = np.loadtxt('NACA6412UPPER.txt')
        if flag:
            plt.figure()
            plt.plot(lower[:, 0], lower[:, 1], '-r', label='LOW')
            indices = np.arange(0, lower.shape[0], 5)
            plt.plot(lower[indices, 0], lower[indices, 1], 'r^', markersize=4)
            plt.plot(upper[:, 0], upper[:, 1], '-b', label='UP')
            indices = np.arange(0, upper.shape[0], 5)
            plt.plot(upper[indices, 0], upper[indices, 1], 'b^', markersize=4)
            plt.legend(fontsize=16)
            plt.xlabel('chord', fontsize=16)
            plt.ylabel('thickness', fontsize=16)
            plt.ylim(-0.5, 0.5)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title('NACA6412', fontsize=16)
            plt.tight_layout()
            plt.show()
        return lower, upper

    elif filename == 'RAE2822':
        lip_radii = 1.33e-3
        lower = np.loadtxt('RAE2822LOWER.txt')
        upper = np.loadtxt('RAE2822UPPER.txt')
        if flag:
            plt.figure()
            plt.plot(lower[:, 0], lower[:, 1], '-r', label='LOW')
            indices = np.arange(0, lower.shape[0], 5)
            plt.plot(lower[indices, 0], lower[indices, 1], 'r^', markersize=4)
            plt.plot(upper[:, 0], upper[:, 1], '-b', label='UP')
            indices = np.arange(0, upper.shape[0], 5)
            plt.plot(upper[indices, 0], upper[indices, 1], 'b^', markersize=4)
            plt.legend(fontsize=16)
            plt.xlabel('chord', fontsize=16)
            plt.ylabel('thickness', fontsize=16)
            plt.ylim(-0.5, 0.5)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title('RAE2822', fontsize=16)
            plt.tight_layout()
            plt.show()
        return lower, upper


if __name__ == '__main__':
    load_airfoil(filename='NACA6412')
    load_airfoil(filename='NACA0012')
    load_airfoil(filename='RAE2822')
