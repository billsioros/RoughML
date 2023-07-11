import numpy as np


def correlation(z_ngs):
    N = z_ngs.shape[0]

    rdif, hhcf1d = np.arange(N // 2), np.zeros(N // 2)

    for ndif in range(N // 2):
        surf1 = z_ngs[:N, : (N - ndif)]
        surf2 = z_ngs[:N, ndif:N]
        difsur2 = (surf1 - surf2) ** 2
        hhcf1d[ndif] = np.sqrt(np.mean(np.mean(difsur2)))

    return rdif, hhcf1d
