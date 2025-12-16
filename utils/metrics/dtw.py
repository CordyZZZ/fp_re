import numpy as np

def dtw_distance(a: np.ndarray, b: np.ndarray, *, normalize: bool = True) -> float:
    """
    Classic DTW with L2 pointwise cost.
    a: (Ta, D), b: (Tb, D)
    Returns accumulated distance (optionally normalized by Ta+Tb).
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    Ta, D = a.shape
    Tb, _ = b.shape

    # DP matrix with inf padding
    dp = np.full((Ta + 1, Tb + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0

    # compute costs row by row
    for i in range(1, Ta + 1):
        ai = a[i-1]
        # vectorized local costs against all b
        costs = np.linalg.norm(b - ai[None, :], axis=1)
        for j in range(1, Tb + 1):
            c = costs[j-1]
            dp[i, j] = c + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

    dist = float(dp[Ta, Tb])
    if normalize:
        dist = dist / float(Ta + Tb)
    return dist
