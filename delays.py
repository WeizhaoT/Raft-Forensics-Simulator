import numpy as np


class BaseDelay:
    def __init__(self, n, adv=-1) -> None:
        self.n = n
        self.base = 0
        self.adv = adv

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


# Special delay for testing "bad vote"
class ModuloSplitDelay(BaseDelay):
    def __init__(self, n, start, step, adv) -> None:
        super().__init__(n, adv)
        self.start = start
        self.step = step
        self.dist_rank = {i: i for i in range(n)}
        self.rebase(0)

    def __call__(self, n1, n2) -> int:
        super().__call__(n1, n2)
        if self.base == n1 or self.base == n2:
            other = n1 if n2 == self.base else n2
            return self.start * self.dist_rank[other] if self.dist_rank[other] * 2 < self.n else \
                self.start * (self.dist_rank[other] - self.n // 2) + self.step
        else:
            d = abs(self.dist_rank[n1] - self.dist_rank[n2])
            return self.start * d if d * 2 < self.n else self.start * (d - self.n // 2) + self.step

        # n1, n2 = (n1-self.base) % self.n, (n2-self.base) % self.n
        # return self.start * abs(n2 - n1) if abs(n2 - n1) * 2 < self.n else \
        #     self.start * (abs(n2 - n1) - self.n // 2) + self.step

    def rebase(self, b) -> None:
        super().rebase(b)
        self.dist_rank[b] = 0
        rank = 1
        if self.adv != self.base:
            self.dist_rank[self.adv] = 1
            rank = 2

        for i in range(self.n-1):
            j = (self.base + 1 + i) % self.n
            if j == self.adv:
                continue
            self.dist_rank[j] = rank
            rank += 1


class ModuloRandomDelay(BaseDelay):
    def __init__(self, n: int, start: int, step: int, rand_mult: float) -> None:
        super().__init__(n)
        self.start = start
        self.step = step
        self.mult = max(1., rand_mult)

    def __call__(self, n1: int, n2: int) -> int:
        super().__call__(n1, n2)
        n1, n2 = (n1-self.base) % self.n, (n2-self.base) % self.n
        dist = (abs(n2 - n1) % self.n) * self.step + self.start
        return np.random.randint(dist, int(dist * self.mult) + 1)
