from proximal.backend_extension import get_backend_ext
import numpy
import math


class Var:
    """Base class for variables.

    Attributes:
        cs (tuple) : The shape of the centered grid. cs[0] is the shape of time dimension\
        and cs[1:] is the size of the space dimension.
        ll (tuple) : Size of time x space domain i.e. the domain is [0, ll[0]] x ... x \
            [0, ll[N-1]].
        D (list) : The list containing the density and the momentum. D[0] is the density \
            and D[1], ...., D[N] is the momentum.
        Z (array-like of shape cs) : The source term of the variable.
        nx (module) : The backend module used for computation such as numpy or torch.
    """

    def __init__(self, cs: tuple, ll: tuple, D: list, Z):
        self.N = len(cs)
        assert len(ll) == self.N
        assert Z.shape == cs
        self.cs = cs
        self.ll = ll
        self.nx = get_backend_ext(*D, Z)
        self.D = [self.nx.copy(Dk) for Dk in D]
        self.Z = self.nx.copy(Z)

    def proj_positive(self):
        """Project the density to be positive."""
        self.D[0] = self.nx.maximum(self.D[0], 0)

    def dilate_grid(self, s: float):
        """Based on Proposition 3.1 in Chizat et al. 2021, apply rescaling of the space
        so that the new delta (the interpolating paramter) = s * delta.
        For T:(t,x) -> (t,sx), the new variable is
        new rho = T_* rho,
        new omega = s T_* omega,
        new Z = T_* Z. Here, T_* is the pushforward operator.

        Args:
            s (float) : The scaling factor.

        """
        self.ll = tuple([self.ll[0]] + [self.ll[k] * s for k in range(1, self.N)])
        self.D[0] /= s ** (self.N - 1)
        for k in range(1, self.N):
            self.D[k] /= s ** (self.N - 2)
        self.Z /= s ** (self.N - 1)

    def __add__(self, other: "Var"):
        """Add two variables."""
        assert self.cs == other.cs
        assert self.ll == other.ll
        return self.__class__(
            self.cs,
            self.ll,
            [self.D[k] + other.D[k] for k in range(self.N)],
            self.Z + other.Z,
        )

    def __iadd__(self, other: "Var"):
        """In-place addition of two variables."""
        assert self.cs == other.cs
        assert self.ll == other.ll
        for k in range(self.N):
            self.D[k] += other.D[k]
        self.Z += other.Z
        return self

    def __sub__(self, other: "Var"):
        """Subtract two variables."""
        assert self.cs == other.cs
        assert self.ll == other.ll
        return self.__class__(
            self.cs,
            self.ll,
            [self.D[k] - other.D[k] for k in range(self.N)],
            self.Z - other.Z,
        )

    def __mul__(self, other: float):
        """Multiply a variable by a scalar."""
        if isinstance(other, (int, float)):
            return self.__class__(
                self.cs,
                self.ll,
                [self.D[k] * other for k in range(self.N)],
                self.Z * other,
            )
        else:
            raise ValueError("Multiplication not supported for the given type.")

    def __rmul__(self, other: float):
        """Multiply a variable by a scalar."""
        return self.__mul__(other)


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

    def __init__(self, cs: tuple, ll: tuple, D, Z):
        assert len(ll) == len(cs)
        assert all(Dk.shape == cs for Dk in D)
        assert Z.shape == cs
        super().__init__(cs, ll, D, Z)

    def energy(self, delta: float, p: float, q: float):
        """Compute the energy of the variable
        ∫∫ (1/p) |ω|^p/rho^(p-1) + s^p (1/q) |ζ|^q/rho^(q-1).
        """
        fp = self.nx.zeros(self.cs)  # s^p (1/q) |ζ|^q/rho^(q-1)
        fq = self.nx.zeros(self.cs)  # (1/p) |ω|^p/rho^(p-1)
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
                * self.nx.sum(
                    self.nx.stack([self.D[k][ind] ** p for k in range(1, self.N)]),
                    axis=0,
                )
            ) / (self.D[0][ind] ** (p - 1))
        return self.nx.sum(fp + fq) * math.prod(self.ll) / math.prod(self.cs)


class Svar(Var):
    """A class for staggered variables.

    Attributes:
        cs (tuple) : The size of the centered grid. cs[0] is the size of time dimension\
        and cs[1:] is the size of the space dimension.
        ll (tuple) : size of time x space domain.
        D (list) : The list containing the density and the momentum. D[0] is the density \
            and D[1], ...., D[N] is the momentum.
        Z (array-like) : The source term of the variable.
    """

    def __init__(self, cs: tuple, ll: tuple, D: list, Z):
        """
         def __init__(self, rho0, rho1, T: int, ll: tuple):
        N = len(rho0.shape) + 1
        assert rho0.shape == rho1.shape
        assert len(ll) == N
        nx = get_backend_ext(rho0, rho1)
        self.rho0 = nx.copy(rho0)
        self.rho1 = nx.copy(rho1)

        cs = (T,) + rho0.shape
        ll = ll
        shapes_staggered = [
            tuple((numpy.array(cs) + numpy.eye(N, dtype=int)[k])) for k in range(N)
        ]
        D = [nx.zeros(shapes_staggered[k]) for k in range(N)]
        D[0] = linear_interpolation(
            rho0, rho1, T
        )  # Initialize density by linear interpolation
        Z = nx.zeros(cs)
        """
        super().__init__(cs, ll, D, Z)

    def proj_BC(self, rho0, rho1):
        """Project the variable to satisfy the boundary conditions."""
        self.D[0][0] = rho0
        self.D[0][-1] = rho1
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

    def copy(self):
        """Return a copy of the variable."""
        return Svar(self.cs, self.ll, self.D, self.Z)


class CSvar:
    """ A pair of staggered and centered variables.

    Attributes:
        nx (Backend) : The backend module used for computation such as numpy or torch.
        cs (tuple) : The size of the centered grid. cs[0] is the size of time dimension\
        and cs[1:] is the size of the space dimension.
        ll (tuple) : size of time x space domain.
        U (Svar) : The staggered variable.
        V (Cvar) : The centered variable.

    """

    def __init__(self, rho0, rho1, T: int, ll: tuple, U: Svar = None, V: Cvar = None):
        self.N = len(rho0.shape) + 1
        self.nx = get_backend_ext(rho0, rho1)
        self.cs = (T,) + rho0.shape
        self.ll = ll
        self.rho0 = self.nx.copy(rho0)
        self.rho1 = self.nx.copy(rho1)
        # Initialize U and V
        if U is None:
            N = len(rho0.shape) + 1
            cs = (T,) + rho0.shape
            shapes_staggered = get_staggered_shape(cs)
            D = [self.nx.zeros(shapes_staggered[k]) for k in range(N)]
            D[0] = linear_interpolation(rho0, rho1, T)
            Z = self.nx.zeros(cs)
            self.U = Svar(cs, ll, D, Z)
        else:
            self.U = Svar(
                U.cs,
                U.ll,
                [self.nx.copy(U.D[k]) for k in range(U.N)],
                self.nx.copy(U.Z),
            )
        if V is None:
            self.V = interp(self.U)
        else:
            self.V = Cvar(
                self.cs,
                ll,
                V.D,
                V.Z,
            )

    def interp_(self):
        """Interpolate U to V in-place."""
        interp_(self.V, self.U)

    def proj_positive(self):
        """Project the density to be positive."""
        self.U.proj_positive()
        self.V.proj_positive()

    def dist_from_CE(self):
        """Calculate the L2 norm of div(D) - Z."""
        return self.U.dist_from_CE()

    def dilate_grid(self, s: float):
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
        return dist * math.prod(self.U.ll) / math.prod(self.U.cs)

    def dist_from_constraint(self, H, F):
        """Calculate the L2 norm of |HU-F|."""
        HU = (
            self.nx.sum(interp(self.U).D[0] * H, axis=tuple(range(1, self.U.N)))
            * math.prod(self.U.ll[1:])
            / math.prod(self.U.cs[1:])
        )
        return self.nx.sum(self.nx.abs(HU - F))

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
            self.rho0, self.rho1, self.U.cs[0], self.U.ll, self.U, self.V
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
        assert isinstance(self.U, Svar)
        place_holder = CSvar(
            self.rho0, self.rho1, self.U.cs[0], self.ll, self.U, self.V
        )
        place_holder.U = self.U - other.U
        place_holder.V = self.V - other.V
        return place_holder

    def __mul__(self, other: float):
        """Multiply a variable by a scalar."""
        if isinstance(other, (int, float)):
            place_holder = CSvar(
                self.rho0, self.rho1, self.U.cs[0], self.ll, self.U, self.V
            )
            place_holder.U = self.U * other
            place_holder.V = self.V * other
            return place_holder
        else:
            raise ValueError("Multiplication not supported for the given type.")

    def __rmul__(self, other: float):
        """Multiply a variable by a scalar."""
        return self.__mul__(other)


def get_staggered_shape(cs: tuple):
    """Given the shape of a centered grid, return the shape of the staggered grids.

    Args:
        cs (tuple) : The shape of the centered grid.

    Returns:
        list : The shape of the staggered grids.

    """
    return [
        tuple((numpy.array(cs) + numpy.eye(len(cs), dtype=int)[k]))
        for k in range(len(cs))
    ]


def linear_interpolation(r0, r1, T: int):
    """Given two densities r0 and r1, return the linear interpolation between them on a
    time staggered grid.
    """
    nx = get_backend_ext(r0, r1)
    t = nx.linspace(0, 1, T + 1, type_as=r0)
    t = t.reshape(-1, *([1] * len(r0.shape)))
    return t * r1 + (1 - t) * r0


def interp_(V: Cvar, U: Svar):  # in-place interpolation
    for k in range(U.N):
        slices = [slice(None)] * U.N
        slices[k] = slice(0, U.cs[k])
        V.D[k][...] = ((U.D[k] + U.nx.roll(U.D[k], -1, axis=k)) / 2)[tuple(slices)]
    V.Z[...] = U.Z


def interpT_(U: Svar, V: Cvar):
    """Apply the transpose of the interpolation operator in-place.
    We apply the transpose of the interpolation operator to the variable V and store the
    result in U.
    """
    for k in range(V.N):
        dk = list(V.cs)
        dk[k] = 1
        dk = tuple(dk)
        slices = [slice(None)] * V.N
        slices[k] = slice(0, V.cs[k] + 1)
        cat = V.nx.concatenate([V.nx.zeros(dk), V.D[k], V.nx.zeros(dk)], axis=k)
        U.D[k][...] = ((cat + V.nx.roll(cat, -1, axis=k)) / 2)[tuple(slices)]
    U.Z[...] = V.Z


def interp(U: Svar):
    V = Cvar(U.cs, U.ll, [U.nx.zeros(U.cs) for _ in range(U.N)], U.nx.zeros(U.cs))
    interp_(V, U)
    return V


def interpT(V: Cvar):
    """Apply the transpose of the interpolation operator to the variable V."""
    staggered_shape = get_staggered_shape(V.cs)
    U = Svar(
        V.cs,
        V.ll,
        [V.nx.zeros(staggered_shape[k]) for k in range(V.N)],
        V.nx.zeros(V.cs),
    )
    interpT_(U, V)
    return U


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
