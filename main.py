import scipy
import numpy as np

X1 = scipy.io.loadmat("X1.mat")
Y = scipy.io.loadmat("Y.mat")


# def get_dx(m, n):
#     """
#     this function returns the dx matrix such that Dx x is the column stack representation of the spacial derivative in the x direction
#     :param m:
#     :param n:
#     :return:
#     """
#     dx = np.zeros([m * n, m * n])
#     for i in range(m):
#         for j in range(n):
#
#             dx[i*n+i, j] = -1
#             dx[j*(n-1), j + m] = 1
#     return dx

def get_dx(m, n):
    dx = np.zeros([m * n, m * n])

    j = 0

    while j + m < m * n:
        dx[j, j] = -1
        dx[j, j + m] = 1
        j+=1

    return dx


def get_dy(m, n):
    """
    this function returns the dy matrix such that Dy x is the column stack representation of the spacial derivative in the y direction
    :param m:
    :param n:
    :return:
    """
    # dy = np.zeros([m * n, m * n])
    # for j in range(n):
    #     for i in range(1, m):  # iterate over the y direction
    #         dy[j * m + i - 1, j * m + i - 1] = -1
    #         dy[j * m + i - 1, j * m + i] = 1
    # return dy

    dy = np.zeros([m * n, m * n])

    j = 0
    while j //(m-1) + j < m*n:
    #for j in range(m*n):
        index = j //(m-1) + j
        dy[index, index] = -1
        dy[index, index + 1] = 1
        j += 1
    return dy

if __name__ == '__main__':
    m = 3
    n = 2
    Dy = get_dy(m, n)
    Dx = get_dx(m, n)

    x = np.arange(m * n)
    dx = np.dot(Dx, x)
    dy = np.dot(Dy, x)
