import proximal.grids as grids
from proximal.backend_extension import get_backend_ext
from ot.backend import Backend
import math


def root(a, b, c, d, nx: Backend):
    """Compute the root of a cubic polynomial ax^3 + bx^2 + cx + d.
    If a, b, c, and d are arrays, the function computes the roots elementwise.

    Args:
        a (array): The coefficient of x^3.
        b (array): The coefficient of x^2.
        c (array): The coefficient of x.
        d (array): The constant term.
        nx (module): The backend module used for computation such as numpy or torch.

    """
    assert (
        a.shape == b.shape == c.shape == d.shape
    ), "Coefficients must have the same shape"
    z = nx.zeros(a.shape)
    u = nx.zeros(a.shape)
    v = nx.zeros(a.shape)
    # Transform coefficients
    p = -((b / a) ** 2) / 3 + c / a
    q = 2 * (b / a) ** 3 / 27 - (b * c) / (a**2) / 3 + d / a
    delta = q**2 + 4 / 27 * p**3

    id = delta > 0
    u[id] = nx.cbrt((-q[id] + nx.sqrt(delta[id])) / 2)
    v[id] = nx.cbrt((-q[id] - nx.sqrt(delta[id])) / 2)
    z[id] = u[id] + v[id] - b[id] / a[id] / 3

    id = delta < 0
    u[id] = (((-q[id] + 1j * nx.sqrt(-delta[id])) / 2) ** (1 / 3)).real
    z[id] = 2 * u[id] - b[id] / a[id] / 3

    id = delta == 0
    z[id] = nx.where(
        q[id] == 0,
        -b[id] / a[id] / 3,
        3 * q[id] / p[id] - b[id] / a[id] / 3,
    )
    return z


def mdl(M, nx: Backend):
    """Given a list of arrays of the same shape, return the elementwise L2 norm."""
    return nx.sqrt(sum(m**2 for m in M))


def proxA_(dest: grids.Cvar, M, gamma):
    """In-place calculation of proximal operator for sum abs (M_i)

    I think this is actually calculating the proximal operator for the L2 norm
    i.e. sum abs(M_i)^2

    As this operator does not affect WFR, I keep it as it is for now.

    """
    softth = dest.nx.maximum(1 - gamma / mdl(M, dest.nx), 0.0)
    for k in range(len(M)):
        dest[k] = softth * M[k]


def proxB_(destR, destM: list, R, M: list, gamma: float, nx: Backend):
    """In-place calculation of proximal operator for sum |M_i|^2/R_i."""
    a = nx.ones(R.shape)
    b = 2 * gamma - R
    c = gamma**2 - 2 * gamma * R
    d = -(gamma / 2) * mdl(M, nx) ** 2 - gamma**2 * R
    destR[...] = nx.maximum(0.0, root(a, b, c, d, nx))
    DD = nx.zeros(R.shape)
    DD[destR > 0] = 1.0 - gamma / (gamma + destR[destR > 0])
    for k in range(len(M)):
        destM[k] = DD * M[k]


def proxF_(dest: grids.Cvar, V: grids.Cvar, gamma: float, p: float, q: float):
    "Return prox_F(V) where F is the energy functional and V on centered grid"
    if p == 1 and q < 1:  # W1
        dest.D[0][:] = V.D[0]
        proxA_(dest.D[1:], V.D[1:], gamma)
    elif p == 2 and q < 1:  # W2
        proxB_(dest.D[0], dest.D[1:], V.D[0], V.D[1:], gamma, dest.nx)
    elif p == 1 and q == 1:  # "Bounded Lipschitz"
        dest.D[0][:] = V.D[0]
        proxA_(dest.D[1:], V.D[1:], gamma)
        proxA_(dest.Z, V.Z, gamma)
    elif p == 2 and q == 2:  # WFR
        proxB_(
            dest.D[0],
            dest.D[1:] + [dest.Z],
            V.D[0],
            V.D[1:] + [V.Z],
            gamma,
            dest.nx,
        )
    elif p == 2 and q == 1:  # Partial W2
        proxB_(dest.D[0], dest.D[1:], V.D[0], V.D[1:], gamma, dest.nx)
        proxA_(dest.Z, V.Z, gamma)
    elif p == 1 and q == 2:  # W1-FR
        proxA_(dest.D[1:], V.D[1:], gamma)
        proxB_(dest.D[0], dest.Z, V.D[0], V.Z, gamma, dest.nx)
    else:
        raise ValueError("Functional not implemented")


def poisson_(f, ll, source, nx: Backend):
    """Solve Δu+f=0(source=False) or Δu-u+f=0 on the centered grid.
    The BC is Neumann BC defined on the staggered grid.

    f will be overwritten with the solution in-place.

    Args:
        f (array): The function f on the centered grid.
        ll (tuple): The length scales of the domain.
        source (bool): True if source problem.
    """
    d = f.ndim
    N = f.shape
    h = [length / n for length, n in zip(ll, N)]
    dims = [1] * d
    D = nx.zeros(f.shape)

    for k in range(d):
        dims = [1] * d
        dims[k] = N[k]
        dep = nx.zeros(tuple(dims))
        for i in range(N[k]):
            slices = [slice(None)] * d
            slices[k] = i
            slices = tuple(slices)
            dep[slices] = (2 * math.cos(math.pi * i / N[k]) - 2) / h[k] ** 2
        D += dep

    if source:
        D -= 1
    else:
        D[0] = 1

    # dctn
    """
    renorm =  math.prod(N) * (2**d)
    f[...] = nx.dctn(f, axes=range(d))
    f /= -D * renorm
    f[...] = nx.idctn(f, axes=range(d))
    """
    # axis wise DCT
    for axe in range(d):
        f[...] = nx.dct(f, axis=axe, norm="ortho")
    f /= -D
    for axe in range(d):
        f[...] = nx.idct(f, axis=axe, norm="ortho")


def minus_interior_(dest, M, dpk, cs, dim):
    """ Subtract dpk from the interior of a staggered variable M and replaces the \
        interior of dest in place by the result.

    Args:
        dest (array): The destination array. It should have the same shape as M.
        M (array): The source array. It should be defined on a staggered grid. \
        That is, the shape should be (N0, N1, ..., N_k +1,  Nd) where k is the staggered\
        dimension.
        dpk (array): The array to subtract. It should have the shape\
        (N0, N1,..., N_k-1,..., Nd).
        cs (tuple): The size of the centered grid i.e. (N0, N1, ..., Nd).
        dim (int): The dimension along which to subtract. k in the above description.
    """
    assert dest.shape == M.shape, "Destination and source shapes must match"
    slices = [slice(None)] * M.ndim
    slices[dim] = slice(1, cs[dim])

    interior_diff = M[tuple(slices)] - dpk
    dest[tuple(slices)] = interior_diff


def projCE_(dest: grids.Svar, U: grids.Svar, rho_0, rho_1, source: bool):
    """ Given a staggered variable U, project it so that it satisfies the \
        continuity equation. The result is stored in dest in place.

    Args:
        dest (Svar): The destination variable.
        U (Svar): The source variable.
        rho_0 (array): The source density.
        rho_1 (array): The destination density.
        source (bool): True if we are solving a OT with source problem.
    """

    assert dest.ll == U.ll, "Destination and source lengths must match"

    U.proj_BC(rho_0, rho_1)
    p = -U.remainder_CE()
    poisson_(p, U.ll, source, dest.nx)

    for k in range(len(U.cs)):
        dpk = dest.nx.diff(p, axis=k) * U.cs[k] / U.ll[k]
        minus_interior_(dest.D[k], U.D[k], dpk, U.cs, k)

    dest.proj_BC(rho_0, rho_1)
    if source:
        dest.Z[...] = U.Z - p


def invQ_mul_A_(dest, Q, src, dim: int, nx: Backend):
    """Apply the inverse of Q to the input src along the specified dimension.

    This function modifies the array dest in-place and calculates
    dest[i1, i2, ... , :, i_n] = Q^-1 * src[i1,i2,... :, i_n] for all i1,...,in and :\
    is at the specified dimension. That is, it applies the inverse of Q to each 1D slice.

    Args:
        dest (array): The destination array.
        Q (array): The matrix Q to apply the inverse.
        src (array): The source array.
        dim (int): The dimension along which to apply the inverse.
        nx (module): The backend module used for computation such as numpy or torch.

    """

    # Put the dimension to the first axis
    new_axes = (dim,) + tuple(i for i in range(src.ndim) if i != dim)
    one_d_slices = nx.transpose(src, axes=new_axes)
    dim_shape = one_d_slices.shape[0]
    remaining_shape = one_d_slices.shape[1:]

    # Reshape the result to 2D
    one_d_slices = one_d_slices.reshape(dim_shape, -1)

    # Put the batch dimension to the first axis
    # one_d_slices = nx.transpose(one_d_slices)

    # Apply the inverse of Q to each slice
    invQ_slices = nx.solve(Q, one_d_slices)

    # Reshape the result back to the original shape
    invQ_slices = invQ_slices.reshape(-1, *remaining_shape)

    # Put the dimension back to the original axis
    inverse_axes = tuple(0 if i == dim else i + 1 - (i >= dim) for i in range(src.ndim))
    invQ_slices = nx.transpose(invQ_slices, axes=inverse_axes)

    # Put the result back to the original array
    dest[...] = invQ_slices


def projinterp_(dest: grids.CSvar, x: grids.CSvar, Q):
    """Calculate the projection of the interpolation operator for x.

    Given the input x=(U,V), calculate the projection of the interpolation operator by
    U' = Q^-1(U+I*V) and V' = I(U) where I is the interpolation operator.
    Here, Q = Id+I*I. The result is stored in dest=(U',V').

    Args:
        dest (CSvar): The destination variable.
        x (CSvar): The input variable.
        Q (list): The tensor of Q matrices [Q1, Q2, ..., QN] where
        Qk = Id + I^T I such that I is the interpolation matrix
        (1/2, 1/2, .... 0)
        (0, 1/2, 1/2, ...,0)
        ...
        (0, ..., 1/2, 1/2)
        of size (x.cs[k]-1) x x.cs[k] for each dimension k.
        We will precompute these matrices for efficiency.
    """
    assert dest.ll == x.ll, "Length scales must match"

    x_U_copy = x.U.copy()  # Copy x.U since interpT_ will overwrite x.U if dest=x

    # Calculate I*V and store it in U'
    grids.interpT_(dest.U, x.V)
    # Add U to I*V and store it in U'... (*)
    dest.U += x_U_copy

    # Apply inverse Q matrix operation for each dimension
    for k in range(dest.U.N):
        invQ_mul_A_(dest.U.D[k], Q[k], dest.U.D[k], k, dest.nx)

    # Average source terms (* adds x.V.Z(moved to dest.U by interpT_) and x.U.Z, so we
    # only need to divide by 2 here)
    dest.U.Z *= 0.5

    # Calculate V' = I(U) and store it in V'
    dest.interp_()


def precomputeProjInterp(cs, nx: Backend):
    B = []
    for n in cs:
        # Create a tridiagonal matrix
        main_diag = nx.full((n + 1,), 6)
        main_diag[0], main_diag[-1] = 5, 5  # Adjust the first and last element
        off_diag = nx.ones(n)
        Q = nx.diag(off_diag, -1) + nx.diag(main_diag, 0) + nx.diag(off_diag, 1)
        Q /= 4
        # Store the result
        B.append(Q)
    return B


def stepDR(
    w: grids.CSvar, x: grids.CSvar, y: grids.CSvar, z: grids.CSvar, prox1, prox2, alpha
):
    """Apply one step of the Douglas-Rachford algorithm to the variables
    w, x, y, and z."""
    # Step 1: Update x based on z and w
    x = 2 * z - w

    # Step 2: Apply proximal operator 1 to x, updating y
    prox1(y, x)

    # Step 3: Update w with step size alpha
    w += alpha * (y - z)

    # Step 4: Apply proximal operator 2 to w, updating z
    prox2(z, w)

    return w, x, y, z


def computeGeodesic(rho0, rho1, T, ll, p=2.0, q=2.0, delta=1.0, niter=1000):
    assert delta > 0, "Delta must be positive"
    source = q >= 1.0  # Check if source problem

    nx = get_backend_ext(rho0, rho1)

    def prox1(y: grids.CSvar, x: grids.CSvar, source, gamma, p, q):
        projCE_(y.U, x.U, rho0 * delta**rho0.ndim, rho1 * delta**rho0.ndim, source)
        proxF_(y.V, x.V, gamma, p, q)

    def prox2(y, x, Q):
        projinterp_(y, x, Q)

    # Adjust mass to match if not a source problem
    if q < 1.0:
        print("Computing geodesic for standard optimal transport...")
        rho1 *= nx.sum(rho0) / nx.sum(rho1)
        delta = 1.0  # Ensure delta is set correctly for non-source problems
        alpha, gamma = 1.8, max(nx.max(rho0), nx.max(rho1)) / 2
    else:
        print("Computing a geodesic for optimal transport with source...")
        alpha, gamma = (
            1.8,
            delta ** (rho0.ndim - 1) * max(nx.max(rho0), nx.max(rho1)) / 15,
        )

    # Initialize using linear interpolation
    w, x, y, z = [grids.CSvar(rho0, rho1, T, ll) for _ in range(4)]

    # Change of variable for scale adjustment
    for var in [w, x, y, z]:
        var.dilate_grid(1 / delta)
        var.rho1 *= delta**rho0.ndim
        var.rho0 *= delta**rho0.ndim

    # Precompute projection interpolation operators if needed
    Q = precomputeProjInterp(x.cs, x.nx)

    Flist, Clist = nx.zeros(niter), nx.zeros(niter)

    for i in range(niter):
        if i % (niter // 100) == 0:
            print(f"\rProgress: {i // (niter // 100)}%", end="")

        w, x, y, z = stepDR(
            w,
            x,
            y,
            z,
            lambda y, x: prox1(y, x, source, gamma, p, q),
            lambda y, x: prox2(y, x, Q),
            alpha,
        )

        Flist[i] = z.energy(delta, p, q)
        Clist[i] = z.dist_from_CE()

    # Final projection and positive density adjustment
    projCE_(z.U, z.U, rho0 * delta**rho0.ndim, rho1 * delta**rho0.ndim, source)
    z.proj_positive()
    z.dilate_grid(delta)  # Adjust back to original scale
    z.interp_()  # Final interpolation adjustment

    print("\nDone.")
    return z, (Flist, Clist)
