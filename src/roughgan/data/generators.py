from abc import ABC, abstractmethod

import numpy as np
import sympy
from scipy import stats

from roughgan.plot import as_grayscale_image


class SurfaceGenerator(ABC):
    def __init__(self, n_points, rms, skewness, kurtosis, corlength_x, corlength_y, alpha) -> None:
        self.n_points = n_points
        self.rms = rms
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.corlength_x = corlength_x
        self.corlength_y = corlength_y
        self.alpha = alpha

        self._mean = 0
        self._length = 0

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.n_points}, {self.rms}, {self.skewness}, {self.kurtosis}, {self.corlength_x}, {self.corlength_y}, {self.alpha})"

    def __repr__(self) -> str:
        return f"<{self}>"

    def __call__(self, length):
        self._length = length

        return self

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for _ in range(self._length):
            yield self.generate_surface()

    def sort(self, elements):
        indices = np.argsort(elements, axis=0)

        return elements[indices], indices

    @abstractmethod
    def autocorrelation(self, tx, ty):
        raise NotImplementedError

    def generate_surface(self):
        # 1st step: Generation of a Gaussian surface

        # Determine the autocorrelation function R(tx,ty)
        R = np.zeros((self.n_points, self.n_points))

        txmin = -self.n_points // 2
        txmax = self.n_points // 2

        tymin = -self.n_points // 2
        tymax = self.n_points // 2

        dtx = (txmax - txmin) // self.n_points
        dty = (tymax - tymin) // self.n_points

        for tx in range(txmin, txmax, dtx):
            for ty in range(tymin, tymax, dty):
                R[tx + txmax, ty + tymax] = self.autocorrelation(tx, ty)

        # According to the Wiener-Khinchine theorem FR is the power spectrum of the desired profile
        FR = np.fft.fft2(R, s=(self.n_points, self.n_points))
        AMPR = np.sqrt(dtx**2 + dty**2) * abs(FR)

        # 2nd step: Generate a white noise, normalize it and take its Fourier transform
        X = np.random.rand(self.n_points, self.n_points)
        aveX = np.mean(np.mean(X))

        dif2X = (X - aveX) ** 2
        stdX = np.sqrt(np.mean(np.mean(dif2X)))
        X = X / stdX

        XF = np.fft.fft2(X, s=(self.n_points, self.n_points))

        # 3rd step: Multiply the two Fourier transforms
        YF = XF * np.sqrt(AMPR)

        # 4th step: Perform the inverse Fourier transform of YF and get the desired surface
        zaf = np.fft.ifft2(YF, s=(self.n_points, self.n_points))
        z = np.real(zaf)

        avez = np.mean(np.mean(z))
        dif2z = (z - avez) ** 2
        stdz = np.sqrt(np.mean(np.mean(dif2z)))
        z = ((z - avez) * self.rms) / stdz

        # Define the fraction of the surface to be analysed
        xmin = 0
        xmax = self.n_points
        ymin = 0
        ymax = self.n_points
        z_gs = z[xmin:xmax, ymin:ymax]

        # 2nd step: Generation of a non-Gaussian noise NxN
        z_ngn = stats.pearson3.rvs(
            self.skewness,
            loc=self._mean,
            scale=self.rms,
            size=(self.n_points, self.n_points),
        )

        # 3rd step: Combination of z_gs with z_ngn to output a z_ms
        v_gs = z_gs.flatten(order="F")
        v_ngn = z_ngn.flatten(order="F")

        Igs = np.argsort(v_gs)

        vs_ngn = np.sort(v_ngn)

        v_ngs = np.zeros_like(vs_ngn)
        v_ngs[Igs] = vs_ngn

        z_ngs = np.asmatrix(v_ngs.reshape(self.n_points, self.n_points, order="F")).H

        return z_ngs


class NonGaussianSurfaceGenerator(SurfaceGenerator):
    def __init__(
        self,
        n_points=128,
        rms=1,
        skewness=0,
        kurtosis=3,
        corlength_x=4,
        corlength_y=4,
        alpha=1,
    ) -> None:
        super().__init__(n_points, rms, skewness, kurtosis, corlength_x, corlength_y, alpha)

    def autocorrelation(self, tx, ty):
        return (self.rms**2) * np.exp(
            -(
                (abs(np.sqrt((tx / self.corlength_x) ** 2 + (ty / self.corlength_y) ** 2)))
                ** (2 * self.alpha)
            ),
        )


class BesselNonGaussianSurfaceGenerator(NonGaussianSurfaceGenerator):
    def __init__(
        self,
        n_points=128,
        rms=1,
        skewness=0,
        kurtosis=3,
        corlength_x=4,
        corlength_y=4,
        alpha=1,
        beta_x=1,
        beta_y=1,
    ) -> None:
        super().__init__(n_points, rms, skewness, kurtosis, corlength_x, corlength_y, alpha)

        self.beta_x, self.beta_y = beta_x, beta_y

    def autocorrelation(self, tx, ty):
        return super().autocorrelation(tx, ty) * sympy.besselj(
            0,
            (2 * np.pi * np.sqrt((tx / self.beta_x) ** 2 + (ty / self.beta_y) ** 2)),
        )


if __name__ == "__main__":
    generate = NonGaussianSurfaceGenerator()

    for surface in generate(1):
        as_grayscale_image(surface)

    besel_generate = BesselNonGaussianSurfaceGenerator(128, 1, 0, 3, 16, 16, 0.5, 4000, 4000)

    for surface in besel_generate(1):
        as_grayscale_image(surface)
