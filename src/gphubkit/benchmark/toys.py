"""Toy functions database for benchmarking."""

import numpy as np


def fn_BM_01(X: np.ndarray) -> np.ndarray:
    """SANTNER ET AL. (2003) DAMPED COSINE FUNCTION.

    1D - x ∈ [0, 1]
    ref: https://www.sfu.ca/~ssurjano/santetal03dc.html
    train_size: (10xd-20xd) -> 10-20.
    """
    assert X.shape[1] == 1
    return np.exp(-1.4 * X) * np.cos(3.5 * np.pi * X)


def fn_BM_02(X: np.ndarray) -> np.ndarray:
    """BRANIN (BRANIN-HOO) FUNCTION.

    2D - x ∈ [-5, 10] x [0, 15]
    ref: https://www.sfu.ca/~ssurjano/branin.html
    train_size: (10xd-20xd) -> 20-40.
    """
    assert X.shape[1] == 2
    a, r, s = 1, 6, 10
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    t = 1 / (8 * np.pi)
    term1 = a * (X[:, 1] - b * X[:, 0] ** 2 + c * X[:, 0] - r) ** 2
    term2 = s * (1 - t) * np.cos(X[:, 0])
    return term1 + term2 + s


def fn_BM_03(X: np.ndarray) -> np.ndarray:
    """GOLDSTEIN-PRICE FUNCTION.

    2D - x ∈ [-2, 2] x [-2, 2]
    ref: https://www.sfu.ca/~ssurjano/goldpr.html
    train_size: (10xd-20xd) -> 20-40
    """
    assert X.shape[1] == 2
    fact1a = (X[:, 0] + X[:, 1] + 1) ** 2
    fact1b = 19 - 14 * X[:, 0] + 3 * X[:, 0] ** 2 - 14 * X[:, 1] + 6 * X[:, 0] * X[:, 1] + 3 * X[:, 1] ** 2
    fact2a = (2 * X[:, 0] - 3 * X[:, 1]) ** 2
    fact2b = 18 - 32 * X[:, 0] + 12 * X[:, 0] ** 2 + 48 * X[:, 1] - 36 * X[:, 0] * X[:, 1] + 27 * X[:, 1] ** 2
    return (1 + fact1a * fact1b) * (30 + fact2a * fact2b)


def fn_BM_04(X: np.ndarray) -> np.ndarray:
    """DETTE & PEPELYSHEV (2010) CURVED FUNCTION.

    3D - x ∈ [0, 1] x [0, 1] x [0, 1]
    ref: https://www.sfu.ca/~ssurjano/detpep10curv.html
    train_size: (10xd-20xd) -> 30-60
    """
    assert X.shape[1] == 3
    term1 = 4 * (X[:, 0] - 2 + 8 * X[:, 1] - 8 * X[:, 1] ** 2) ** 2
    term2 = (3 - 4 * X[:, 1]) ** 2
    term3 = 16 * np.sqrt(X[:, 2] + 1) * (2 * X[:, 2] - 1) ** 2
    return term1 + term2 + term3


def fn_BM_05(X: np.ndarray) -> np.ndarray:
    """OTL CIRCUIT FUNCTION.

    6D - Rb1, Rb2, Rf, Rc1, Rc2, Beta - x ∈ [50, 150] x [25, 70] x [0.5, 3] x [1.2, 2.5] x [0.25, 1.2] x [50, 300]
    ref: https://www.sfu.ca/~ssurjano/otlcircuit.html
    train_size: (10xd-20xd) -> 60-120
    """
    eps = 1e-10
    assert X.shape[1] == 6
    Rb1, Rb2, Rf, Rc1, Rc2, Beta = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    Vb1 = 12 * Rb2 / (Rb1 + Rb2 + eps)
    return (
        ((Vb1 + 0.74) * Beta * (Rc2 + 9)) / (Beta * (Rc2 + 9) + Rf + eps)
        + 11.35 * Rf / (Beta * (Rc2 + 9) + Rf + eps)
        + 0.74 * Rf * Beta * (Rc2 + 9) / ((Beta * (Rc2 + 9) + Rf) * Rc1 + eps)
    )


def fn_BM_06(X: np.ndarray) -> np.ndarray:
    """BOREHOLE FUNCTION.

    8D - rw, r, Tu, Hu, Tl, Hl, L, Kw - x ∈ [0.05, 0.15] x [100, 50000] x [63070, 115600] x [990, 1110] x [63.1, 116] x [700, 820] x [1120, 1680] x [9855, 12045]
    ref: https://www.sfu.ca/~ssurjano/borehole.html
    train_size: (10xd-20xd) -> 80-160
    """
    # Ux = torch.distributions.Uniform(
    #     low=torch.tensor([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]),
    #     high=torch.tensor([0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]),
    # )
    assert X.shape[1] == 8
    eps = 1e-10
    rw, r, Tu, Hu, Tl, Hl, L, Kw = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    frac1 = 2 * np.pi * Tu * (Hu - Hl)
    frac2a = 2 * L * Tu / ((np.log((r / (rw + eps)) + eps) * rw**2 * Kw) + eps)
    frac2b = Tu / (Tl + eps)
    frac2 = np.log((r / (rw + eps)) + eps) * (1 + frac2a + frac2b)
    return frac1 / (frac2 + eps)


def fn_BM_07(X: np.ndarray) -> np.ndarray:
    """MORRIS FUNCTION.

    20D - x ∈ [0, 1]**20 (see, e.g., Le Gratiet, Marelli, Sudret (2016))
    ref: https://arxiv.org/pdf/1606.04273
    train_size: (10xd-20xd) -> 200-400
    """
    assert X.shape[1] == 20
    N, M = X.shape
    W = 2 * (X - 0.5)
    W[:, [2, 4, 6]] = 2 * (1.1 * X[:, [2, 4, 6]] / (X[:, [2, 4, 6]] + 0.1) - 0.5)
    Y = 0
    for i in range(20):
        bi = 20 if i < 10 else (-1) ** (i + 1)
        Y += bi * W[:, i]
    for i in range(19):
        for j in range(i + 1, 20):
            bij = -15 if i < 6 and j < 6 else (-1) ** (i + j + 1)
            Y += bij * W[:, i] * W[:, j]
    for i in range(18):
        for j in range(i + 1, 19):
            for k in range(j + 1, 20):
                bijl = -10 if i < 5 and j < 5 and k < 5 else 0
                Y += bijl * W[:, i] * W[:, j] * W[:, k]
    bijls = 5
    Y += bijls * W[:, 0] * W[:, 1] * W[:, 2] * W[:, 3]
    return Y
