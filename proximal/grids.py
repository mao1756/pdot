from backend_extension import get_backend_ext
import numpy


"""
todo:
"roll" and "diff" does not exist on backends so we need a way to replace it.
Done: Added roll and diff to the backend module.
"""


class Var:
    """Base class for variables.

    Attributes:
        cs (tuple) : The shape of the centered grid. cs[0] is the shape of time dimension\
        and cs[1:] is the size of the space dimension.
        ll (tuple) : Size of time x space domain i.e. the domain is [0, ll[0]] x ... x \
            [0, ll[N-1]].
        D (list) : The list containing the density and the momentum. D[0] is the density \
            and D[1], ...., D[N] is the momentum.
        Z (array) : The source term of the variable.
        nx (module) : The backend module used for computation such as numpy or torch.
    """

    def __init__(self, N: int, cs: tuple, ll: tuple, D: list, Z):
        self.N = N
        self.cs = cs
        self.ll = ll
        self.D = D
        self.Z = Z
        self.nx = get_backend_ext(D + [Z])

    def proj_positive(self):
        """Project the density to be positive."""
        self.D[0] = numpy.maximum(self.D[0], 0)

    def dilate_grid(self, s):
        """Apply pushforward by T:(t,x) -> (t,sx) to the variable."""
        self.ll = tuple([self.ll[0]] + [self.ll[k] * s for k in range(1, self.N)])
        self.D[0] /= s ** (self.N - 1)
        for k in range(1, self.N):
            self.D[k] /= s ** (self.N - 2)
        self.Z /= s ** (self.N - 1)

    def __add__(self, other):
        """Add two variables."""
        assert self.cs == other.cs
        assert self.ll == other.ll
        return Var(
            self.N,
            self.cs,
            self.ll,
            [self.D[k] + other.D[k] for k in range(self.N)],
            self.Z + other.Z,
        )

    def __iadd__(self, other):
        """In-place addition of two variables."""
        assert self.cs == other.cs
        assert self.ll == other.ll
        for k in range(self.N):
            self.D[k] += other.D[k]
        self.Z += other.Z
        return self

    def __sub__(self, other):
        """Subtract two variables."""
        assert self.cs == other.cs
        assert self.ll == other.ll
        return Var(
            self.N,
            self.cs,
            self.ll,
            [self.D[k] - other.D[k] for k in range(self.N)],
            self.Z - other.Z,
        )


class Cvar(Var):
    """A class for centered variales.

    Attributes:
        cs (tuple) : The size of the centered grid. cs[0] is the size of time dimension\
        and cs[1:] is the size of the space dimension.
        ll (tuple) : Size of time x space domain i.e. the domain is [0, ll[0]] x ... x \
            [0, ll[N-1]].
        D (list) : The list containing the density and the momentum. D[0] is the density \
            and D[1], ...., D[N] is the momentum.
        Z (array) : The source term of the variable.
        nx (module) : The backend module used for computation such as numpy or torch.
    """

    def __init__(self, cs, ll, D, Z):
        assert len(ll) == len(cs)
        assert all(Dk.shape == cs for Dk in D)
        assert Z.shape == cs
        super().__init__(len(cs), cs, ll, D, Z)

    def energy(self, delta: float, p: float, q: float):
        """Compute the energy of the variable
        ∫∫ (1/p) |ω|^p/rho^(p-1) + s^p (1/q) |ζ|^q/rho^(q-1).
        """
        fp = self.nx.zeros(self.cs)
        fq = self.nx.zeros(self.cs)
        ind = self.D[0] > 0
        if q >= 1:
            fp[ind] = (
                (delta**p / q)
                * (self.nx.abs(self.Z[ind]) ** q)
                / (self.D[0][ind] ** (q - 1))
            )
        if p >= 1:
            fq[ind] = (
                (1.0 / p)
                * self.nx.sum([self.D[k][ind] ** p for k in range(1, self.N)], axis=0)
            ) / (self.D[0][ind] ** (p - 1))
        return self.nx.sum(fp + fq) * self.nx.prod(self.ll) / self.nx.prod(self.cs)


class Svar(Var):
    """A class for staggered variables.

    Attributes:
        cs (tuple) : The size of the centered grid. cs[0] is the size of time dimension\
        and cs[1:] is the size of the space dimension.
        ll (tuple) : size of time x space domain.
        D (list) : The list containing the density and the momentum. D[0] is the density \
            and D[1], ...., D[N] is the momentum.
        Z (array) : The source term of the variable.
        rho0 (array) : The initial density.
        rho1 (array) : The final density.
    """

    def __init__(self, rho0, rho1, T: int, ll: tuple):
        N = len(rho0.shape) + 1
        assert rho0.shape == rho1.shape
        assert len(ll) == N
        self.rho0 = rho0
        self.rho1 = rho1
        nx = get_backend_ext([rho0, rho1])

        cs = (T,) + rho0.shape
        ll = ll
        shape_before_staggered = numpy.array(self.cs)
        shapes_staggered = [shape_before_staggered + numpy.eye(N)[k] for k in range(N)]
        D = [nx.zeros(shapes_staggered[k]) for k in range(N)]
        D[0] = linear_interpolation(
            rho0, rho1, T
        )  # Initialize density by linear interpolation
        Z = nx.zeros(self.cs)

        super().__init__(N, cs, ll, D, Z)

    def proj_BC(self):
        """Project the variable to satisfy the boundary conditions."""
        self.D[0][0] = self.rho0
        self.D[0][-1] = self.rho1
        for k in range(1, self.N):
            slices = [slice(None)] * self.N
            slices[k] = 0
            self.D[k][tuple(slices)] = 0  # Neumann BC
            slices[k] = -1
            self.D[k][tuple(slices)] = 0  # Neumann BC

    def remainder_CE(self):
        """Calculate div(D) - Z. If the continuity equation is satisfied, the result
        should be zero."""
        v = -self.Z
        for k in range(len(self.D)):
            v += self.nx.diff(self.D[k], axis=k) * self.cs[k] / self.ll[k]
        return v

    def dist_from_CE(self):
        """Calculate the L2 norm of div(D) - Z."""
        return (
            self.nx.sum(self.remainder_CE() ** 2)
            * self.nx.prod(self.ll)
            / self.nx.prod(self.cs)
        )


class CSvar:
    def __init__(self, rho0, rho1, T: int, ll: tuple, U: Svar = None, V: Cvar = None):
        self.cs = (T,) + rho0.shape
        self.ll = ll
        # Initialize U and V
        if U is None:
            self.U = Svar(rho0, rho1, T, ll)
        else:
            self.U = U
        if V is None:
            self.V = interp(self.U)
        else:
            self.V = V
        self.nx = get_backend_ext([rho0, rho1])

    def interp_(self):
        """Interpolate U to V in-place."""
        interp_(self.V, self.U)

    def proj_positive(self):
        """Project the density to be positive."""
        self.U.proj_positive()
        self.V.proj_positive()

    def proj_BC(self):
        """Project the variable to satisfy the boundary conditions."""
        self.U.proj_BC()

    def dist_from_CE(self):
        """Calculate the L2 norm of div(D) - Z."""
        return self.U.dist_from_CE()

    def dilate_grid(self, s):
        """Apply pushforward by T:(t,x) -> (t,sx) to the variable."""
        self.U.dilate_grid(s)
        self.V.dilate_grid(s)

    def dist_from_interp(self):
        """Calculate the L2 norm of interp(U) - V"""
        dist = 0
        intU = interp(self.U)
        for k in range(self.U.N):
            dist += self.U.nx.sum((intU.D[k] - self.V.D[k]) ** 2)
        dist += self.U.nx.sum((intU.Z - self.V.Z) ** 2)
        return dist * self.U.nx.prod(self.U.ll) / self.U.nx.prod(self.U.cs)

    def energy(self, delta: float, p: float, q: float):
        """Compute the energy of the variable
        ∫∫ (1/p) |ω|^p/rho^(p-1) + s^p (1/q) |ζ|^q/rho^(q-1).
        """
        return self.V.energy(delta, p, q)

    def __add__(self, other: "CSvar"):
        """Add two variables."""
        assert self.cs == other.cs
        assert self.ll == other.ll
        place_holder = CSvar(
            self.U.rho0, self.U.rho1, self.U.cs[0], self.U.ll, self.U, self.V
        )
        place_holder.U += other.U
        place_holder.V += other.V
        return place_holder

    def __iadd__(self, other: "CSvar"):
        """In-place addition of two variables."""
        assert self.cs == other.cs
        assert self.ll == other.ll
        self.U += other.U
        self.V += other.V
        return self

    def __sub__(self, other: "CSvar"):
        """Subtract two variables."""
        assert self.cs == other.cs
        assert self.ll == other.ll
        place_holder = CSvar(
            self.U.rho0, self.U.rho1, self.U.cs[0], self.U.ll, self.U, self.V
        )
        place_holder.U = self.U - other.U
        place_holder.V = self.V - other.V
        return place_holder

    def __mul__(self, other: float):
        """Multiply a variable by a scalar."""
        if isinstance(other, (int, float)):
            place_holder = CSvar(
                self.U.rho0, self.U.rho1, self.U.cs[0], self.U.ll, self.U, self.V
            )
            place_holder.U = self.U * other
            place_holder.V = self.V * other
            return place_holder
        else:
            raise ValueError("Multiplication not supported for the given type.")

    def __rmul__(self, other: float):
        """Multiply a variable by a scalar."""
        return self.__mul__(other)


def linear_interpolation(r0, r1, T: int):
    nx = get_backend_ext([r0, r1])
    t = nx.linspace(0, 1, T + 1)
    t = t.reshape(-1, *([1] * len(r0.shape)))
    return t * r1 + (1 - t) * r0


def interp(U: Svar):
    V = Cvar(U.cs, U.ll, [U.nx.zeros(U.cs) for _ in range(U.N)], U.nx.zeros(U.cs))
    interp_(V, U)
    return V


def interpT_(U: Svar, V: Cvar):
    """Apply the transpose of the interpolation operator in-place.
    We apply the transpose of the interpolation operator to the variable V and store the
    result in U.
    """
    for k in range(V.N):
        dk = V.cs
        dk[k] = 1
        cat = V.nx.concatenate([V.nx.zeros(dk), V.D[k], V.nx.zeros(dk)], axis=k)
        U.D[k] = (cat + V.nx.roll(cat, -1, axis=k)) / 2
    U.Z[...] = V.Z


def interp_(V: Cvar, U: Svar):  # in-place interpolation
    for k in range(U.N):
        V.D[k] = (U.D[k] + U.nx.roll(U.D[k], -1, axis=k)) / 2
    V.Z[...] = U.Z


def speed_and_growth(V: Cvar, max_ratio=100):
    """Compute momentum/density and source/density for the variable V.
    If the density is too small, the result is set to zero.
    """
    ind = V.D[0] > V.nx.max(V.D[0]) / max_ratio  # indices with large enough density
    g = V.nx.zeros(V.cs)
    g[ind] = V.Z[ind] / V.D[0][ind]
    v = [
        V.nx.zeros(V.cs) for _ in range(len(V.D) - 1)
    ]  # Not len(V.D) because V.D[0] is the density
    for k in range(len(V.D) - 1):
        v[k][ind] = V.D[k + 1][ind] / V.D[0][ind]
    return v, g


""" Unused codes from ChatGPT
def main_constructor(rho0, rho1, T, ll):
    U = Svar(
        (T,) + rho0.shape,
        ll,
        [
            np.zeros((T + 1,) + rho0.shape + (np.eye(len(rho0.shape) + 1)[k],))
            for k in range(len(rho0.shape) + 1)
        ],
        np.zeros((T,) + rho0.shape),
    )
    U.D[0] = linear_interpolation(rho0, rho1, T)
    V = interp(U)
    return CSvar(U.cs, U.ll, U, V)

def diff_inplace(dest, M, dim):
    dest[:] = np.diff(M, axis=dim)


def diff(M, dim):
    return np.diff(M, axis=dim)
"""
