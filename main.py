import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array


def show_image(x):
    plt.figure()
    plt.imshow(x, cmap="Grays")
    plt.show()


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

def get_dx(m, n, dtype='<f8'):
    """
    This function returns the Dx matrix such that Dx x is the column stack representation of the spacial derivative in the x direction.
    :param m:
    :param n:
    :param dtype:
    :return:
    """
    # dx = np.zeros([m * n, m * n])

    j = 0
    data = []
    rows = []
    cols = []
    while j + m < m * n:
        # The j'th row of the dx is x_{j+m}-x{j} for j + m < m * n, and zeros otherwise
        data.append(-1)
        rows.append(j)
        cols.append(j)
        data.append(1)
        rows.append(j)
        cols.append(j + m)
        # dx[j, j] = -1
        # dx[j, j + m] = 1
        j += 1
    dx = csr_array((data, (rows, cols)), shape=(m * n, m * n),
                   dtype=dtype)  # using a sparse array as that matrix can be quite large
    return dx


def get_dy(m, n, dtype='<f8'):
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

    # dy = np.zeros([m * n, m * n])

    data = []
    rows = []
    cols = []

    j = 0
    while j // (m - 1) + j < m * n:
        # The j'th row of the dy is X_{index+1} - X{index}. the index is j shifted up by j // (m - 1),
        # as this creates zero rows between changes in columns
        index = j // (m - 1) + j
        data.append(-1)
        rows.append(index)
        cols.append(index)

        data.append(1)
        rows.append(index)
        cols.append(index + 1)
        # index = j //(m-1) + j
        # dy[index, index] = -1
        # dy[index, index + 1] = 1
        j += 1
    dy = csr_array((data, (rows, cols)), shape=(m * n, m * n), dtype=dtype)
    return dy

def question_13():
    """
    This function calculate the norm1 and norm2 of the data given in Question 13.
    :return:
    """
    X = [_ for _ in range(-4, 5, 1)]
    f_1_X = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]
    f_2_X = [0, 0.0025, 0.018, 0.1192, 0.5, 0.8808, 0.9820, 0.9975, 1]

    # calculation of derivative using existing method
    # getting dimensions
    # m = n = int(np.sqrt(len(X)))
    #
    # Dx = get_dx(m, n)
    # Dy = get_dy(m, n)
    #
    # f_1_DX = Dx @ f_1_X
    # f_2_DX = Dx @ f_2_X
    # f_1_Dy = Dy @ f_1_X
    # f_2_Dy = Dy @ f_2_X

    # calculate derivative using list
    f_1_DX = [x2 - x1 for x1, x2 in zip(f_1_X[:-1], f_1_X[1:])]
    f_2_DX = [x2 - x1 for x1, x2 in zip(f_2_X[:-1], f_2_X[1:])]

    # compute norm 1
    f_1_DX_norm1 = np.linalg.norm(f_1_DX, ord=1)
    f_2_DX_norm1 = np.linalg.norm(f_2_DX, ord=1)

    # compute norm 2
    f_1_DX_norm2 = np.linalg.norm(f_1_DX, ord=2)
    f_2_DX_norm2 = np.linalg.norm(f_2_DX, ord=2)

    print("f_1_DX:")
    print(f_1_DX)
    print("f_2_DX:")
    print(f_2_DX)

    print(f" The norm 1 of f_1 is: {f_1_DX_norm1}")
    print(f" The norm 1 of f_2 is: {f_2_DX_norm1}")
    print(f" The norm 2 of f_1 is: {f_1_DX_norm2}")
    print(f" The norm 2 of f_2 is: {f_2_DX_norm2}")




if __name__ == '__main__':
    X1 = scipy.io.loadmat("X1.mat")["X1"]
    X2 = scipy.io.loadmat("X2.mat")["X2"]
    X3 = scipy.io.loadmat("X3.mat")["X3"]
    Y = scipy.io.loadmat("Y.mat")

    X = X3  # Choose signal to analyze
    m, n = X.shape
    Dy = get_dy(m, n)
    Dx = get_dx(m, n)

    # x = np.arange(m * n)
    # dx = np.dot(Dx, x)
    # dy = np.dot(Dy, x)

    dx = Dx @ X.reshape([-1, 1])
    dy = Dy @ X.reshape([-1, 1])

    gradient_amplitude = np.sqrt(dx ** 2 + dy ** 2)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(X, cmap="Grays")
    axs[0, 0].set_title('Source Signal')
    axs[0, 0].axis('off')  # Remove axis ticks and labels

    axs[0, 1].imshow(dx.reshape([m, n]), cmap="Grays")
    axs[0, 1].set_title('x-direction partial derivative')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(dy.reshape([m, n]), cmap="Grays")
    axs[1, 0].set_title('y-direction partial derivative')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(gradient_amplitude.reshape([m, n]), cmap="Grays")
    axs[1, 1].set_title('gradient_amplitude')
    axs[1, 1].axis('off')

    plt.show()
