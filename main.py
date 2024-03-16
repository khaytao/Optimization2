import scipy
import numpy as np

X1 = scipy.io.loadmat("X1.mat")
Y = scipy.io.loadmat("Y.mat")


def get_dx(m, n):
    """
    this function returns the dx matrix such that Dx x is the column stack representation of the spacial derivative in the x direction
    :param m:
    :param n:
    :return:
    """
    pass


def get_dy(m, n):
    """
    this function returns the dy matrix such that Dy x is the column stack representation of the spacial derivative in the y direction
    :param m:
    :param n:
    :return:
    """
    dy = np.zeros([m * n, m * n])
    for j in range(n):
        for i in range(1, m): #iterate over the y direction
            dy[j*m + i-1, j*m + i-1] = -1
            dy[j*m +i-1, j*m + i] = 1
    return dy

if __name__ == '__main__':
    dy = get_dy(3,3)