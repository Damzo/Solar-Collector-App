import numpy as np
from scipy.optimize import minimize


class optimizeRingArray:
    """
    This class optimized the parameters for designing a ring array solar reflector
    with a fresnel lens in the internal ray of the first ring
    Inputs: internal ray of the first ring in the top (Rin_0), Total area of the reflector surface (A_target),
            Number of rings (N), maximum height of the array (h_max), material width (w)
    Outputs: Optimization rms, Focal distances array of each ring (f_2D), heights array (internal and external) of
                each ring (h)
    """

    def __init__(self, Rin_0: float, A_target: float, N: int, w: float, h_max: float):
        # Inputs parameters
        self.Rin_0 = Rin_0  # first top ring internal ray, equal to the fresnel lens diameter (m)
        self.A_target = A_target  # targeted reflector area (m²)
        self.N = N  # number of rings
        self.w = w  # reflector materiel width (m)
        self.h_max = h_max  # max height of the array, corresponding to the external height of the 1st ring (m)

        # Some other initial parameters needed
        self.f0 = 1e-2
        self.G = -0.5
        self.b_lim = (-self.G * Rin_0 + 1e-3, h_max - self.G * Rin_0)  # limit conditions on b_in
        self.heights = np.ones((N, 2))
        self.b_in = 0
        self.b_ex = 0
        self.R = np.ones((N, 2))
        self.f_2D = np.ones((N, 2))

    def update_init(self, Rin_0: float, A_target: float, N: int, w: float, h_max: float):
        inputdatas = {'Rin_0': Rin_0, 'A_target': A_target, 'N': N, 'w': w, 'h_max': h_max}
        default_inputdatas = {'Rin_0': self.Rin_0, 'A_target': self.A_target, 'N': self.N, 'w': self.w,
                              'h_max': self.h_max}
        for name, val in inputdatas.items():
            # print('name is', name, 'and value is', val)
            if val is None:
                inputdatas[name] = default_inputdatas[name]

        self.__init__(**inputdatas)

    def evaluate_rms(self, X):

        self.G = X[0]
        self.f0 = X[1]
        self.N = round(X[2])

        self.R = np.ones((self.N, 2))
        self.f_2D = np.ones((self.N, 2))
        self.heights = np.ones((self.N, 2))
        n_inc = np.array([0, -1, 0])
        theta = np.zeros(2 * self.N)

        for i in np.arange(0, self.N):

            if i == 0:
                # R[0, :] = [Rin_0, Rin_0 + dR]
                self.R[0, 0] = self.Rin_0  # R_in of ring 0
                self.R[0, 1] = np.sqrt(4 * self.f0 * self.h_max)  # R_ex of ring 0

                self.f_2D[0, :] = self.f0

                self.heights[i, 0] = self.Rin_0 ** 2 / (4 * self.f_2D[0, 0])
                self.heights[i, 1] = self.h_max

                self.b_ex = self.h_max - self.G * self.R[0, 1]
                self.b_in = self.Rin_0 ** 2 / (4 * self.f0) - self.G * self.Rin_0

            else:
                self.R[i, 0] = self.R[i - 1, 1] + self.w
                self.heights[i, 0] = self.G * self.R[i, 0] + self.b_in
                # Computation of focal distance of ring i
                a = 4
                b = 4 * (self.G * self.R[i, 0] + self.b_in - self.f_2D[0, 0])
                c = -self.R[i, 0] ** 2
                coeff = [a, b, c]
                roots = np.roots(coeff)
                if np.iscomplex(roots).any():
                    return 1e3
                else:
                    self.f_2D[i, :] = np.max(roots)
                # Computation of external ray of ring i
                coeff = [1 / (4 * self.f_2D[i, 0]), -self.G, -(self.f_2D[i, 0] - self.f_2D[0, 0] + self.b_ex)]
                roots = np.roots(coeff)
                if np.iscomplex(roots).any():
                    return 1e3
                else:
                    self.R[i, 1] = np.max(roots)

                self.heights[i, 1] = self.G * self.R[i, 1] + self.b_ex

                if self.heights[i, 1] < 0:
                    return 1e3

                n_in = np.array([self.R[i, 0] / (2 * self.f_2D[i, 0]), 1, 0])
                n_ex = np.array([self.R[i, 1] / (2 * self.f_2D[i, 1]), 1, 0])
                ur_in = n_inc - 2 * np.dot(n_inc, n_in) * n_in
                ur_ex = n_inc - 2 * np.dot(n_inc, n_ex) * n_ex
                ur_in = ur_in / np.linalg.norm(ur_in)
                ur_ex = ur_ex / np.linalg.norm(ur_ex)

                theta[2 * i: 2 * i + 2] = np.array([np.arccos(np.dot(ur_in, -n_inc)),
                                                    np.arccos(np.dot(ur_ex, -n_inc))])
        dtheta = 1
        for i in np.arange(0, self.N - 1):
            dtheta = dtheta * (theta[2 * i + 3] - theta[2 * i])
        if dtheta < 0:
            return 1e3

        # A = np.pi * self.R / (6 * self.heights ** 2) * ((self.R ** 2 + 4 * self.heights ** 2) ** (3 / 2) - self.R ** 3)
        A = np.pi * (self.R[:, 1]**2 - self.R[:, 0]**2)
        # dA = A[:, 1] - A[:, 0]
        A_total = np.sum(A)
        rms = (A_total - self.A_target) ** 2  # + int(angles > 0)* 100

        return rms

    def optimize(self):
        bnds = ((None, 0), (1e-6, 5e-2), (1, None))
        X0 = np.array([self.G, self.f0, self.N])  # X=[G, b_in]

        cons = ({'type': 'ineq', 'fun': lambda x: self.Rin_0 ** 2 / (4 * x[1]) - x[0] * self.Rin_0})
        res = minimize(self.evaluate_rms, X0, method='SLSQP', options={'disp': False}, bounds=bnds, constraints=cons)

        outputdatas = {'RMS': res.fun, 'G': res.x[0], 'f0': res.x[1], 'N': res.x[2],
                       'focal': self.f_2D[:, 0], 'Rays': self.R, 'Heights': self.heights}

        return outputdatas
