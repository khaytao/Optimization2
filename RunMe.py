import sys

import numpy as np
import sympy as sp
from scipy.io import loadmat

from main import (
    question_13,
    question_15,
    get_dx,
    get_dy
)


def q3():
    print("===================")
    print("Question 3:")
    print()

    print("Dx =", get_dx(5, 5))
    print()
    print("Dy =", get_dy(5, 5))

    print()
    print("===================")    


def q4():
    print("Q4 TODO COMPLETE")


def q9():
    delta_x, delta_y = sp.symbols("\Delta\\tilde{x} \Delta\\tilde{y}")
    A = sp.zeros(8, 25)

    non_zeros = {
        delta_x: [
            (1, 1), (1, 6), (1, 11), (1, 16), (1, 21),
            (3, 2), (3, 7), (3, 12), (3, 17), (3, 22),
            (4, 4), (4, 9), (4, 14), (4, 19), (4, 24)
        ],
        sp.sqrt(delta_x**2 + delta_y**2): [
            (2, 2), (2, 8), (2, 14), (2, 20),
            (5, 5), (5, 9), (5, 13), (5, 17), (5, 21),
            (6, 4), (6, 10),
            (8, 2), (8, 8), (8, 14), (8, 20)
        ],
        delta_y: [
            (7, 16), (7, 17), (7, 18), (7, 19), (7, 20)
        ]
    }

    for value, element_locs in non_zeros.items():
        for element_loc in element_locs:
            A[element_loc[0] - 1, element_loc[1] - 1] = value

    A_q9 = A.subs({delta_x: 1, delta_y: 1})
    np_A_q9 = np.array(A_q9).astype(np.float64)

    m = 5
    n = 5
    L = np.vstack(
        (
            get_dx(m, n).todense(),
            get_dy(m, n).todense()
        )
    )

    lamda = 1e-5
    B = np.vstack((np_A_q9, np.sqrt(lamda) * L))

    Q = B.T @ B
    # Q = loadmat("Q_question_9.mat")["Q"]

    eigvals = np.linalg.eigvals(Q)

    print("===================")
    print("Question 9:")
    print()

    print("Q =", Q)
    print()
    print("Condition number K(Q) =", max(eigvals) / min(eigvals))

    print()
    print("===================")


def q10():
    print("Q10 TODO COMPLETE")
    

def q11():
    print("Q11 TODO COMPLETE")


def q13():
    print("===================")
    print("Question 13:")
    print()

    question_13()

    print()
    print("===================")


def q15():
    print("===================")
    print("Question 15:")
    print()

    X, num_iters, error = question_15()
    print("X =", X)
    print("num_iters =", num_iters)
    print("error =", error)

    print()
    print("===================")


def q16():
    print("Q16 TODO COMPLETE")


def main():
    """All answers to code questions."""
    
    q3()
    q4()
    q9()
    q10()
    q11()
    q13()
    q15()
    q16()


# FLags definition
if len(sys.argv) > 1:
    flag = sys.argv[1]
    
    flags = {
        "-q3": q3,
        "-q4": q4, # TODO COMPLETE
        "-q9": q9,
        "-q10": q10, # TODO COMPLETE
        "-q11": q11, # TODO COMPLETE
        "-q13": q13,
        "-q15": q15,
        "-q16": q16, # TODO COMPLETE
        "-main": main
    }

    flags[flag]()