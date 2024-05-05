import grids
from backend_extension import get_backend_ext
from ot.backend import Backend

# import numpy


def root(a, b, c, d, nx: Backend):
    """Compute the root of a cubic polynomial ax^3 + bx^2 + cx + d."""

    # Transform coefficients
    p = -((b / a) ** 2) / 3 + c / a
    q = 2 * (b / a) ** 3 / 27 - (b * c) / (a**2) / 3 + d / a
    delta = q**2 + 4 / 27 * p**3

    if delta > 0:
        u = nx.power((-q + nx.sqrt(delta)) / 2, 1.0 / 3)
        v = nx.power((-q - nx.sqrt(delta)) / 2, 1.0 / 3)
        z = u + v - b / a / 3
    elif delta < 0:
        u = ((-q + 1j * nx.sqrt(-delta)) / 2) ** (1 / 3)
        z = (u + u.conj() - b / a / 3).real
    else:
        z = (3 * q / p - b / a / 3).real()
    return z


def mdl(M, nx: Backend):
    """Given a list of arrays of the same shape, return the elementwise L2 norm."""
    return nx.sqrt(sum(m**2 for m in M))


def proxA_(dest: grids.Cvar, M, gamma):
    """In-place calculation of proximal operator for sum abs (M_i)"""
    softth = dest.nx.maximum(1 - gamma / mdl(M, dest.nx), 0.0)
    for k in range(len(M)):
        dest[k] = softth * M[k]


def proxB_(destR, destM, R, M, gamma, nx: Backend):
    """In-place calculation of proximal operator for sum |M_i|^2/R_i"""
    mdl = nx.sqrt(sum(m**2 for m in M))
    a = 1.0
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
    elif p == 2 and q == 2:  # WF
        proxB_(
            dest.D[0],
            dest.nx.concatenate((dest.D[1:], [dest.Z])),
            V.D[0],
            dest.nx.concatenate((V.D[1:], [V.Z])),
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
    """Solve Δu+f=0(source=False) or Δu-u+f=0 with Neumann BC on the staggered grid.

    f will be overwritten with the solution in-place.

    Args:
        f (array): The function f.
        ll (tuple): The length scales of the domain.
        source (bool): True if source problem.
    """
    d = f.ndim
    N = f.shape
    h = [length / n for length, n in zip(ll, N)]
    dims = nx.ones(d)
    D = nx.zeros(f.shape)

    for k in range(d):
        dims[:] = 1
        dims[k] = N[k]
        dep = nx.zeros(dims, float)
        for i in range(N[k]):
            slices = [slice(None)] * d
            slices[k] = i
            slices = tuple(slices)
            dep[slices] = (2 * nx.cos(nx.pi * i / N[k]) - 2) / h[k] ** 2
        D += dep

    if source:
        D -= 1
    else:
        D[0] = 1

    renorm = nx.prod(N) * (2**d)  # I am not sure why we need this
    for axe in range(d):
        f = nx.dct(f, axis=axe, norm="ortho")
    f /= -D / renorm
    for axe in range(d):
        f = nx.idct(f, axis=axe, norm="ortho")


def minus_interior_(dest, M, dpk, cs, dim):
    """ Subtract dpk from the interior of a staggered variable M. Replaces dest in place.

    Args:
        dest (array): The destination array. It shoul have the same shape as M.
        M (array): The source array. It should be defined on a staggered grid. \
        That is, the shape should be (N0, N1, ..., N_k +1,  Nd) where k is the staggered\
        dimension.
        dpk (array): The array to subtract. It should have the shape\
        (N0, N1,..., N_k-1,..., Nd).
        cs (tuple): The size of the centered grid i.e. (N0, N1, ..., Nd).
        dim (int): The dimension along which to subtract. k in the above description.
    """
    assert dest.shape == M.shape, "Destination and source shapes must match"
    interior_diff = M.take(indices=range(1, cs[dim]), axis=dim) - dpk
    slices = [slice(None)] * M.ndim
    slices[dim] = slice(1, cs[dim])
    dest[tuple(slices)] = interior_diff


def projCE(dest: grids.Svar, U: grids.Svar, source, plan="none", iplan="none"):
    assert dest.nx.all(dest.ll == U.ll), "Destination and source lengths must match"

    U.proj_BC()
    p = -U.remainder_CE()
    poisson_(p, U.ll, source, dest.nx)

    for k in range(len(U.cs)):
        dpk = dest.nx.diff(p, axis=k) * U.cs[k] / U.ll[k]
        minus_interior_(dest.D[k], U.D[k], dpk, U.cs, k)

    U.proj_BC()
    if source:
        dest.Z[...] = U.Z - p


def projinterp(dest: grids.CSvar, x: grids.CSvar, Q):
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

    # Calculate I*V and store it in U'
    grids.interpT_(dest.U, x.V)

    # Add U to I*V and store it in U'
    for k in range(dest.U.N):
        dest.U.D[k] += x.U.D[k]

    # Apply inverse Q matrix operation for each dimension
    for k in range(dest.U.N):
        invQ_mul_A(
            dest.U.D[k], Q[k], dest.U.D[k], k + 1
        )  # k + 1 because dimensions are 1-based in Julia

    # Average source terms and assign to dest.U.Z
    dest.U.Z = (x.U.Z + x.V.Z) / 2

    # Calculate V' = I(U) and store it in V'
    dest.interp()


def invQ_mul_A(dest, Q, src, dim: int, nx: Backend):
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
    one_d_slices = nx.transpose(one_d_slices)

    # Apply the inverse of Q to each slice
    invQ_slices = nx.solve(Q, one_d_slices)

    # Reshape the result back to the original shape
    invQ_slices = invQ_slices.reshape(-1, *remaining_shape)

    # Put the dimension back to the original axis.
    # Argsort on permutation gives the inverse permutation
    invQ_slices = nx.transpose(invQ_slices, axes=nx.argsort(new_axes))

    # Put the result back to the original array
    dest[...] = invQ_slices


def precomputeProjInterp(cs, nx: Backend):
    B = []
    for n in cs:
        # Create a tridiagonal matrix
        main_diag = nx.full(n, 6)
        main_diag[0], main_diag[-1] = 5, 5  # Adjust the first and last element
        off_diag = nx.ones(n - 1)
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


def computeGeodesic(rho0, rho1, T, ll, p=2.0, q=2.0, delta=1.0, niter=1000):
    assert delta > 0, "Delta must be positive"
    source = q >= 1.0  # Check if source problem

    nx = get_backend_ext([rho0, rho1])

    def prox1(y: grids.CSvar, x: grids.CSvar, source, gamma, p, q):
        # Apply proximal operators, assuming implementations for projCE and proxF
        projCE(y.U, x.U, source, plan="none", iplan="none")
        y.proxF(gamma, p, q)

    def prox2(y, x, Q):
        # Apply interpolation projection, assuming an implementation for projinterp
        projinterp(y, x, Q)

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
    w = grids.CSvar(rho0, rho1, T, ll)
    x, y, z = w, w, w  # Simplified; deep copy or equivalent may be needed

    # Change of variable for scale adjustment
    for var in (w, x, y, z):
        var.dilate_grid(1 / delta)
        var.rho1 *= delta ** (var.N)
        var.rho0 *= delta ** (var.N)

    # FFT plans for efficiency (not direct in NumPy, use rfft for real-input DCT)
    # Precompute projection interpolation operators if needed
    Q = precomputeProjInterp(x.cs, x.nx)  # Assuming a suitable definition exists

    Flist, Clist = nx.zeros(niter), nx.zeros(niter)

    for i in range(niter):
        if i % (niter // 100) == 0:
            print(f"\rProgress: {i // (niter // 100)}%", end="")

        stepDR(
            w,
            x,
            y,
            z,
            lambda y, x: prox1(y, x, rho0, rho1, source, alpha, gamma, p, q),
            lambda y, x: prox2(y, x, Q),
            alpha,
        )

        Flist[i] = z.energy(delta, p, q)
        Clist[i] = z.dist_from_CE()

    # Final projection and positive density adjustment
    projCE(z, z, source)
    z.proj_positive()
    z.dilate_grid(delta)  # Adjust back to original scale
    z.interp_()  # Final interpolation adjustment

    print("\nDone.")
    return z, (Flist, Clist)
