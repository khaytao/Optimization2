import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, csc_array, csr_matrix, vstack
import sympy as sp
import math


from scipy.sparse.linalg import cg
from PIL import Image



def show_image(x, title="", save=False):
    if save:
        im = Image.fromarray(x)
        im.save(title+".jpeg")
    plt.figure()
    plt.imshow(x, cmap="Grays")
    if title:
        plt.title(title)
    plt.show()


def get_dx(m, n, l=1, dtype="<f8"):
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
    dx = csr_array(
        (data, (rows, cols)), shape=(m * n, m * n), dtype=dtype
    )  # using a sparse array as that matrix can be quite large
    return dx


def get_dx2(m, n, l=1, dtype="<f8"):
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
            rows.append(j + (i * m * n))
            cols.append(j + m + (i * m * n))
            j += 1
        i += 1
    dx = csr_matrix(
        (data, (rows, cols)), shape=(m * n * l, m * n * l), dtype=dtype
    )  # using a sparse array as that matrix can be quite large
    return dx


def get_dy(m, n, l=1, dtype="<f8"):
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


def get_dy2(m, n, l=1, dtype="<f8"):
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
    dy = csr_matrix((data, (rows, cols)), shape=(m * n * l, m * n * l), dtype=dtype)
    return dy


def get_dz(m, n, l=1, dtype="<f8"):
    return get_dx2(m * n, l, 1)


def cgls(A, L, y, lamda=1e-5, k_max=300, tolerance=1e-6):
    err_values = []
    x = np.zeros(A.shape[1])
    y_padded = np.concatenate([y, np.zeros(L.shape[0])])
    B = scipy.sparse.vstack([A, np.sqrt(lamda) * L])  # Our matrix Q = B^T B
    sk = B @ x - y_padded
    grad_new = B.T @ sk
    g2_new = grad_new @ grad_new  # g2 is the power of the norm of the gradient
    d = -grad_new

    for k in range(k_max):

        grad_old = grad_new
        g2_old = g2_new
        Bd = B @ d

        # calculate step size
        ak = g2_old / (np.linalg.norm(Bd, ord=2) ** 2)
        # calculate new x
        x = x + ak * d
        # calculate decent direction - new point
        sk = sk + ak * Bd
        grad_new = B.T @ sk

        diff = np.linalg.norm(grad_new, ord=2)
        err_values.append(diff)
        if diff < tolerance:
            return x, k, err_values

        g2_new = grad_new @ grad_new
        beta = (g2_new - grad_new.T @ grad_old)/ g2_old
        d = -grad_new + beta * d
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

    gradient_amplitude = np.sqrt(dx**2 + dy**2)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(X, cmap="Grays")
    axs[0, 0].set_title("Source Signal")
    axs[0, 0].axis("off")  # Remove axis ticks and labels

    axs[0, 1].imshow(dx.reshape([m, n]), cmap="Grays")
    axs[0, 1].set_title("x-direction partial derivative")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(dy.reshape([m, n]), cmap="Grays")
    axs[1, 0].set_title("y-direction partial derivative")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(gradient_amplitude.reshape([m, n]), cmap="Grays")
    axs[1, 1].set_title("gradient_amplitude")
    axs[1, 1].axis("off")

    plt.show()


def eq_4(A, lamda, L, x, y):
    """Implementation of equation 4 from the project document."""
    
    first_term = scipy.sparse.vstack([A, np.sqrt(lamda) * L]) @ x
    
    zeros_dim = first_term.shape[0] - y.shape[0]

    second_term = np.hstack((y, np.zeros(zeros_dim)))
    
    vec = first_term - second_term
    
    return vec.T @ vec


def question_11(lam=1e-5, show_plots=True, large_data=False):
    debug = False

    if debug:
        A = get_A_toyExample()
        Y = get_y_toy_problem()
        x_dim = A.shape[1]
        m = n = round(x_dim ** (1 / 2))
        l = 1
        Dx = get_dx(
            m,
            n,
        )
        Dy = get_dy(m, n)
        L = scipy.sparse.vstack([Dx, Dy])

    else:
        if large_data:
            y = scipy.io.loadmat("Large/y.mat")["y"]
            A = scipy.io.loadmat("Large/A.mat")["A"]
        else:
            y = scipy.io.loadmat("Small/y.mat")["y"]
            A = scipy.io.loadmat("Small/A.mat")["A"]

        Y = np.squeeze(y)
        x_dim = A.shape[1]
        m = n = l = round(x_dim ** (1 / 3))

        Dx = get_dx2(m, n, l)
        Dy = get_dy2(m, n, l)
        Dz = get_dz(m, n, l)

        L = scipy.sparse.vstack([Dx, Dy, Dz])

    tol = 1e-6
    I_max = 10000

    x_opt, num_iter, err = cgls(A.tocsr(), L, Y, lam, I_max, tol)
    x_opt_org = x_opt.reshape([m, n, l])

    if show_plots:
        for _ in range(m):
            if not debug:
                show_image(x_opt_org[:, :, _])

    return {
        "x_opt": x_opt,
        "x_opt_org": x_opt_org,
        "num_iter": num_iter,
        "err": err,
        "L": L,
        "A": A,
        "y": Y,
        "lamda": lam
    }


def question_13():
    """
    This function calculate the norm1 and norm2 of the data given in Question 13.
    :return:
    """
    X = [_ for _ in range(-4, 5, 1)]
    f_1_X = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]
    f_2_X = [0, 0.0025, 0.018, 0.1192, 0.5, 0.8808, 0.9820, 0.9975, 1]


    Dy = get_dy(len(X), 1)

    df_1_Dy = Dy @ f_1_X
    df_2_Dy = Dy @ f_2_X

    f_1_Dy_norm1 = np.linalg.norm(df_1_Dy, ord=1)
    f_2_Dy_norm1 = np.linalg.norm(df_2_Dy, ord=1)

    # compute norm 2
    f_1_Dy_norm2 = np.linalg.norm(df_1_Dy, ord=2)
    f_2_Dy_norm2 = np.linalg.norm(df_2_Dy, ord=2)

    print("df_1_DX:")
    print(df_1_Dy)
    print("f_2_DX:")
    print(df_2_Dy)

    print(f" The norm 1 of f_1 is: {f_1_Dy_norm1}")
    print(f" The norm 1 of f_2 is: {f_2_Dy_norm1}")
    print(f" The norm 2 of f_1 is: {f_1_Dy_norm2}")
    print(f" The norm 2 of f_2 is: {f_2_Dy_norm2}")


def question_toy(
    alpha: float = 1 / 2, num_iter: int = 10000, eps=1e-12, tolerance=1e-10, dtype="<f8"
):

    A = get_A_toyExample()
    Y = get_y_toy_problem()

    x_dim = A.shape[1]
    X = np.random.rand(x_dim)
    # X = np.zeros(x_dim)
    m = n = round(x_dim ** (1 / 2))

    Dx = get_dx(m, n)
    Dy = get_dy(m, n)
    D = scipy.sparse.vstack([Dx, Dy])

    idx = [_ for _ in range(D.shape[0])]  # number of rows of L
    k = num_iter
    err_list = []

    X = np.squeeze(X.reshape([-1, 1]))
    IRLS_iter = 10000
    IRLS_tol = 1e-6
    while k:
        W_vals = 1 / (np.absolute(D @ X) + eps)
        W = csr_array((W_vals, (idx, idx)), shape=(D.shape[0], D.shape[0]), dtype=dtype)
        IRLS_x_opt, IRLS_num_iter, IRLS_err = cgls(
            A.tocsr(), (W @ D), Y, np.sqrt(alpha), IRLS_iter, IRLS_tol
        )
        err_list.append(IRLS_err[-1])
        X = IRLS_x_opt
        if err_list[-1] < tolerance:
            return X, num_iter - k, err_list
        k = k - 1

    return X, num_iter, err_list


def question_15(
    alpha: float = 1 / 2, num_iter: int = 10000, eps=1e-12, tolerance=1e-6, dtype="<f8"
):
    debug = False
    y = scipy.io.loadmat("Small/y.mat")["y"]
    A = scipy.io.loadmat("Small/A.mat")["A"]
    Y = np.squeeze(y)
    x_dim = A.shape[1]
    m = n = l = round(x_dim ** (1 / 3))

    Dx = get_dx2(m, n, l)
    Dy = get_dy2(m, n, l)
    Dz = get_dz(m, n, l)
    D = scipy.sparse.vstack([Dx, Dy, Dz])

    ret_q11 = question_11(show_plots=False)
    X = ret_q11["x_opt"]

    if debug:
        y_padded = np.concatenate([Y, np.zeros(D.shape[0])])
        debug_diff = []


    idx = [_ for _ in range(D.shape[0])]  # number of rows of L
    k = num_iter
    err_list = []
    IRLS_num_iter_list = []
    X = np.squeeze(X.reshape([-1, 1]))
    IRLS_iter = 10000
    IRLS_tol = 1e-6
    for k in range(num_iter):
        if debug:
            print(f"iteration num {k}")
        W_vals = 1 / (np.sqrt(np.absolute(D @ X)) + np.sqrt(eps))
        sqrt_W = csr_array((W_vals, (idx, idx)), shape=(D.shape[0], D.shape[0]), dtype=dtype)
        IRLS_x_opt, IRLS_num_iter, IRLS_err = cgls(
            A=A.tocsr(),
            L=(sqrt_W @ D),
            y=Y,
            lamda=np.sqrt(alpha),
            k_max=IRLS_iter,
            tolerance=IRLS_tol,
        )
        obj_func_val = (1 / 2) * (IRLS_x_opt.T @ A.T @ A @ IRLS_x_opt +
                                  alpha * IRLS_x_opt.T @ D.T @ sqrt_W @ sqrt_W @ D @ IRLS_x_opt -
                                  IRLS_x_opt.T @ A.T @ Y -
                                  Y.T @ A @ IRLS_x_opt.T +
                                  Y.T @ Y)
        print(f" Objective value in iteration {k} is: {obj_func_val}")

        if debug:
            # B = scipy.sparse.vstack([A, np.sqrt(alpha) *  sqrt_W @ D])
            Q = (A.T @ A + alpha * D.T @ sqrt_W @ sqrt_W @ D)
            x_hat, exit_code = cg(A.T @ A + alpha * D.T @ sqrt_W @ sqrt_W @ D, A.T @ Y, maxiter=IRLS_iter, atol=IRLS_tol, rtol=0)
            debug_diff.append(np.linalg.norm(x_hat-IRLS_x_opt)/np.linalg.norm(x_hat))

        err_list.append(IRLS_err[-1])
        IRLS_num_iter_list.append(IRLS_num_iter)
        if debug:
            X = x_hat
        else:
            X = IRLS_x_opt
        if err_list[-1] < tolerance:
            break

    x_opt_org = X.reshape([m, n, l])
    if debug:
        plt.figure()
        plt.plot(debug_diff)
    for _ in range(m):
        show_image(x_opt_org[:, :, _])

    return X, k, err_list

def question_16(
    alpha: float = 1 / 2, num_iter: int = 10000, eps=1e-12, tolerance=1e-7, dtype="<f8"
):
    debug = False
    y = scipy.io.loadmat("Large/y.mat")["y"]
    A = scipy.io.loadmat("Large/A.mat")["A"]
    Y = np.squeeze(y)
    x_dim = A.shape[1]
    m = n = l = round(x_dim ** (1 / 3))

    Dx = get_dx2(m, n, l)
    Dy = get_dy2(m, n, l)
    Dz = get_dz(m, n, l)
    D = scipy.sparse.vstack([Dx, Dy, Dz])

    # ret_q11 = question_11(show_plots=False, large_data=True)
    # X = ret_q11["x_opt"]
    X = np.zeros(A.shape[1])

    idx = [_ for _ in range(D.shape[0])]  # number of rows of L
    k = num_iter
    err_list = []
    IRLS_num_iter_list = []
    X = np.squeeze(X.reshape([-1, 1]))
    IRLS_iter = 10000
    IRLS_tol = 1e-6
    for k in range(num_iter):
        if debug:
            print(f"iteration num {k}")
        W_vals = 1 / (np.sqrt(np.absolute(D @ X)) + np.sqrt(eps))
        sqrt_W = csr_array((W_vals, (idx, idx)), shape=(D.shape[0], D.shape[0]), dtype=dtype)
        IRLS_x_opt, IRLS_num_iter, IRLS_err = cgls(
            A=A.tocsr(),
            L=(sqrt_W @ D),
            y=Y,
            lamda=np.sqrt(alpha),
            k_max=IRLS_iter,
            tolerance=IRLS_tol,
        )

        obj_func_val = (1/2) * (IRLS_x_opt.T @ A.T @ A @ IRLS_x_opt +
                                alpha * IRLS_x_opt.T @ D.T @ sqrt_W @ sqrt_W @ D @ IRLS_x_opt -
                                IRLS_x_opt.T @ A.T @ Y -
                                Y.T @ A @ IRLS_x_opt.T +
                                Y.T @ Y)
        print(f" Objective value in iteration {k} is: {obj_func_val}")


        err_list.append(IRLS_err[-1])
        IRLS_num_iter_list.append(IRLS_num_iter)

        X = IRLS_x_opt
        if err_list[-1] < tolerance:
            break

    x_opt_org = X.reshape([m, n, l])
    for _ in range(m):
        show_image(x_opt_org[:, :, _], title=f"q16_image_{_+1}", save=True)

    return X, k, err_list



def get_A_toyExample():
    values = []
    rows = []
    cols = []
    delta_x = [
        (1, 1),
        (1, 6),
        (1, 11),
        (1, 16),
        (1, 21),
        (3, 2),
        (3, 7),
        (3, 12),
        (3, 17),
        (3, 22),
        (4, 4),
        (4, 9),
        (4, 14),
        (4, 19),
        (4, 24),
    ]
    diag = [
        (2, 2),
        (2, 8),
        (2, 14),
        (2, 20),
        (5, 5),
        (5, 9),
        (5, 13),
        (5, 17),
        (5, 21),
        (6, 4),
        (6, 10),
        (8, 2),
        (8, 8),
        (8, 14),
        (8, 20),
    ]
    delta_y = [(7, 16), (7, 17), (7, 18), (7, 19), (7, 20)]

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
        y_flat != 0
    ].flatten()  # we have defined A in a way that y is ordered correctly. All that's needed is to flatten and take the nonzero elements
