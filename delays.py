import numpy as np


class BaseDelay:
    def __init__(self, n) -> None:
        self.n = n
        self.base = 0

    def __call__(self, n1, n2) -> int:
        if not isinstance(n1, int) or not 0 <= n1 < self.n:
            raise ValueError(f'n1 {n1} is not an integer in [0, {self.n})')
        if not isinstance(n2, int) or not 0 <= n2 < self.n:
            raise ValueError(f'n1 {n2} is not an integer in [0, {self.n})')

        return -1

    def rebase(self, b) -> None:
        self.base = b


class ConstantDelay(BaseDelay):
    def __init__(self, n, d):
        BaseDelay.__init__(self, n)
        self.delay = d

    def __call__(self, n1, n2) -> int:
        super().__call__(n1, n2)
        return self.delay


class UniformDelay(BaseDelay):
    def __init__(self, n, dmin, dmax):
        BaseDelay.__init__(self, n)
        self.dmin = dmin
        self.dmax = dmax
        self.dist = np.tril(np.random.randint(dmin, dmax + 1, size=(n, n)))
        self.dist = self.dist + self.dist.T

    def __call__(self, n1, n2) -> int:
        super().__call__(n1, n2)

        if self.dist[n1, n2] > self.dmin:
            return np.random.randint(self.dmin, self.dist[n1, n2] + 1)
        else:
            return self.dist[n1, n2]


class ModuloDelay(BaseDelay):
    def __init__(self, n, start, step) -> None:
        super().__init__(n)
        self.start = start
        self.step = step

    def __call__(self, n1, n2) -> int:
        super().__call__(n1, n2)
        n1, n2 = (n1-self.base) % self.n, (n2-self.base) % self.n
        return (abs(n2 - n1) % self.n) * self.step + self.start
