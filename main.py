import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, csc_array, csr_matrix, vstack
import sympy as sp
import math
# todo these are python implementations of the ex, for debugging
from scipy.sparse.linalg import cg
from scipy.optimize import line_search
from tqdm import tqdm


def show_image(x):
    plt.figure()
    plt.imshow(x, cmap="Grays")
    plt.show()


def get_dx(m, n, l=1, dtype='<f8'):
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
        j += 1
    dx = csr_array((data, (rows, cols)), shape=(m * n, m * n),
                   dtype=dtype)  # using a sparse array as that matrix can be quite large
    return dx


def get_dx2(m, n, l=1, dtype='<f8'):
    """
    This function returns the Dx matrix such that Dx x is the column stack representation of the spacial derivative in the x direction.
    :param m:
    :param n:
    :param dtype:
    :return:
    """

    data = []
    rows = []
    cols = []
    i = 0
    while i < l:
        j = 0
        while j + m < m * n:
            # The j'th row of the dx is x_{j+m}-x{j} for j + m < m * n, and zeros otherwise
            data.append(-1)
            rows.append(j + (i * m * n))
            cols.append(j + (i * m * n))
            data.append(1)
            rows.append(j+ (i * m * n))
            cols.append(j + m + (i * m * n))
            j += 1
        i += 1
    # dx = csr_array((data, (rows, cols)), shape=(m * n * l, m * n * l),
    #                dtype=dtype)  # using a sparse array as that matrix can be quite large
    dx = csr_matrix((data, (rows, cols)), shape=(m * n * l, m * n * l),
                   dtype=dtype)  # using a sparse array as that matrix can be quite large
    return dx

def get_dy(m, n, l=1, dtype='<f8'):
    """
    this function returns the dy matrix such that Dy x is the column stack representation of the spacial derivative in the y direction
    :param m:
    :param n:
    :return:
    """

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
        j += 1
    dy = csr_array((data, (rows, cols)), shape=(m * n, m * n), dtype=dtype)
    return dy

def get_dy2(m, n, l=1, dtype='<f8'):
    """
    this function returns the dy matrix such that Dy x is the column stack representation of the spacial derivative in the y direction
    :param m:
    :param n:
    :return:
    """

    data = []
    rows = []
    cols = []

    i = 0
    while i < l:
        j = 0
        while j // (m - 1) + j < m * n:
            # The j'th row of the dy is X_{index+1} - X{index}. the index is j shifted up by j // (m - 1),
            # as this creates zero rows between changes in columns
            index = j // (m - 1) + j
            data.append(-1)
            rows.append(index + (i * m * n))
            cols.append(index + (i * m * n))

            data.append(1)
            rows.append(index + (i * m * n))
            cols.append(index + 1 + (i * m * n))
            j += 1
        i += 1
    # dy = csr_array((data, (rows, cols)), shape=(m * n * l, m * n * l), dtype=dtype)
    dy = csr_matrix((data, (rows, cols)), shape=(m * n * l, m * n * l), dtype=dtype)
    return dy

def get_dz(m, n, l=1, dtype='<f8'):
    return get_dx2(m*n,l, 1)


def cgls(A, L, y, lamda=1e-5, k_max=300, tolerance=1e-6, m=1, n=1, l=1):
    err_values = []

    block_size = m * n * l
    x = np.zeros(A.shape[1])
    y_padded = np.concatenate([y, np.zeros(L.shape[0])])
    B = scipy.sparse.vstack([A, np.sqrt(lamda) * L]) # Our matrix Q = B^T B
    sk = B @ x - y_padded
    grad = B.T @ sk
    g2_new = grad @ grad  # g2 is the power of the norm of the gradient
    d = -grad

    for k in range(k_max):

        Bd = B @ d
        g2_old = g2_new
        # calculate step size
        ak = g2_old / (np.linalg.norm(Bd, ord=2)**2)
        # calculate new x
        x = x + ak * d
        # calculate decent direction - new point
        sk = sk + ak * Bd
        grad = B.T @ sk

        diff = np.linalg.norm(grad, ord=2)
        err_values.append(diff)
        if diff < tolerance:
            return x, k, err_values

        g2_new = grad @ grad
        beta = g2_new / g2_old
        d = -grad + beta * d
    return x, k_max, err_values


def question_3():
    X1 = scipy.io.loadmat("X1.mat")["X1"]
    X2 = scipy.io.loadmat("X2.mat")["X2"]
    X3 = scipy.io.loadmat("X3.mat")["X3"]

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

def question_11():
    y = scipy.io.loadmat("Small/y.mat")["y"]
    A = scipy.io.loadmat("Small/A.mat")["A"]

    Y = np.squeeze(y)
    x_dim = A.shape[1]
    m = n = l = round(x_dim ** (1/3))

    Dx = get_dx2(m, n, l)
    Dy = get_dy2(m, n, l)
    Dz = get_dz(m, n, l)

    # L = np.vstack((Dx, Dy, Dz))
    L = scipy.sparse.vstack([Dx, Dy, Dz])

    tol = 1e-6
    lam = 1e-5
    I_max = 10000

    x_opt, num_iter, err = cgls(A.tocsr(), L, Y, lam, I_max, tol, m, n, l)
    x_opt_org = x_opt.reshape([m, n, l])
    for _ in range(m):
        show_image(x_opt_org[:, :, _])

    return


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
    Dy = get_dy(len(X), 1)
    #
    # f_1_Dx = Dx @ f_1_X
    # f_2_Dx = Dx @ f_2_X
    df_1_Dy = Dy @ f_1_X
    df_2_Dy = Dy @ f_2_X

    # calculate derivative using list
    # df_1_Dy = [x2 - x1 for x1, x2 in zip(f_1_X[:-1], f_1_X[1:])]
    # df_2_Dy = [x2 - x1 for x1, x2 in zip(f_2_X[:-1], f_2_X[1:])]

    # compute norm 1
    f_1_Dy_norm1 = np.linalg.norm(df_1_Dy, ord=1)
    f_2_Dy_norm1 = np.linalg.norm(df_2_Dy, ord=1)

    # compute norm 2
    f_1_Dy_norm2 = np.linalg.norm(df_1_Dy, ord=2)
    f_2_Dy_norm2 = np.linalg.norm(df_2_Dy, ord=2)

    # # printing df_Dx
    # print("df_1_Dx:")
    # print(df_1_Dx)
    # print("f_2_Dx:")
    # print(df_2_Dx)

    # printing df_dy
    print("df_1_DX:")
    print(df_1_Dy)
    print("f_2_DX:")
    print(df_2_Dy)

    print(f" The norm 1 of f_1 is: {f_1_Dy_norm1}")
    print(f" The norm 1 of f_2 is: {f_2_Dy_norm1}")
    print(f" The norm 2 of f_1 is: {f_1_Dy_norm2}")
    print(f" The norm 2 of f_2 is: {f_2_Dy_norm2}")


def question_15(alpha: float = 1/2, num_iter: int = 1000, eps=1e-10, tolerance=1e-7, dtype='<f8'):

    A = get_A_toyExample()
    Y = get_y_toy_problem()

    X = np.zeros(5 * 5)

    m = n = 5
    L = get_dy(m, n)

    idx = [_ for _ in range(m*n)]

    k = num_iter
    err = []

    X = np.squeeze(X.reshape([-1, 1]))
    while k:
        W_vals = 1 / (np.absolute(L.T @ X) + eps)
        W = csr_array((W_vals, (idx, idx)), shape=(m * n, m * n), dtype=dtype)

        y_padded = np.concatenate([Y, np.zeros(m * n)])
        B = np.vstack((A.toarray(), np.sqrt(alpha) * (W @ L).toarray()))
        grad = B.T @ (B @ X - y_padded)
        g2_new = grad @ grad
        d = -grad

        g2_old = g2_new
        # get step size
        ak = g2_old / (
                2 * d @ B.T @ B @ d)  # todo closed form step size, doesn't fit into the 1 matrix multiplication restriction
        # evaluate x
        X = X + ak * d
        # evaluate distance from solution
        error = np.linalg.norm(A @ X - Y)
        err.append(error)
        if len(err) > 2:
            if (error - err[-2]) < tolerance:  # todo use formula for better evaluation
                return X, num_iter-k, err

        grad = B.T @ (B @ X - y_padded)
        g2_new = grad @ grad
        beta = g2_new / g2_old
        d = -grad + beta * d
        k = k-1

    return X, num_iter, err

def get_A_toyExample():
    values = []
    rows = []
    cols = []
    delta_x = [(1, 1), (1, 6), (1, 11), (1, 16), (1, 21), (3, 2), (3, 7), (3, 12), (3, 17), (3, 22), (4, 4), (4, 9),
               (4, 14), (4, 19), (4, 24)]
    diag = [
        (2, 2), (2, 8), (2, 14), (2, 20),
        (5, 5), (5, 9), (5, 13), (5, 17), (5, 21),
        (6, 4), (6, 10),
        (8, 2), (8, 8), (8, 14), (8, 20)
    ]
    delta_y = [
        (7, 16), (7, 17), (7, 18), (7, 19), (7, 20)
    ]

    for pair in delta_x:
        values.append(1)
        rows.append(pair[0] - 1)
        cols.append(pair[1] - 1)
    for pair in delta_y:
        values.append(1)
        rows.append(pair[0] - 1)
        cols.append(pair[1] - 1)
    for pair in diag:
        values.append(math.sqrt(2))
        rows.append(pair[0] - 1)
        cols.append(pair[1] - 1)
    return csr_array((values, (rows, cols)), shape=(8, 25), dtype=np.float64)


def get_y_toy_problem():
    Y = scipy.io.loadmat("Y.mat")["Y"]
    y_flat = np.array(Y.todense().flatten())
    return y_flat[
        y_flat != 0].flatten()  # we have defined A in a way that y is ordered correctly. All that's needed is to flatten and take the nonzero elements


if __name__ == '__main__':
    #question_11()
    A = get_A_toyExample().todense()
    L = np.concatenate([get_dx(5, 5).todense(), get_dy(5, 5).todense()])
    y = get_y_toy_problem()
    l = 10 ** -5
    I_max = 10000
    tol = 1e-9
    # # x, k = cgls2(A, L, y, l, I_max, tol, 5, 5)
    x1, k, q10_err = cgls(A, L, y, l, I_max, tol, 5, 5)
    y_padded = np.concatenate([y, np.zeros(50)])
    B = np.vstack((A, np.sqrt(l) * L))
    Q = 2 * B.T @ B
    x_hat, exit_code, err = cg(B.T @ B, B.T @ y_padded, atol=1e-5)  # scipy implementation for reference
    show_image(x1.reshape([5, 5]))
    show_image(x_hat.reshape([5, 5]))
    #
    # print(f"number of iter: {k}.")
    # print("Optimal X:")
    # print(x1)
    # print("Last 10 errors:")
    # print(q10_err[-10:])
    #
    #
    # plt.figure()
    # plt.plot(q10_err[-15:])
    # plt.show()

    # #15
    # X_15 = np.zeros((5, 5))
    # q15_opt_x, q15_iter, q15_err = question_15()
    # print(f"number of iter: {q15_iter}.")
    # print("Optimal X:")
    # print(q15_opt_x)
    # print("Last 10 errors:")
    # print(q15_err[-10:])
