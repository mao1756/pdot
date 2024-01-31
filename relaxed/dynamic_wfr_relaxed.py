import torch
import torchdiffeq
import numpy as np
import scipy as sp
import functools
import math


def _WFR_energy(p: torch.Tensor, v: torch.Tensor, z: torch.Tensor, delta: float):
    """Calculates the Wasserstein-Fisher-Rao energy for the path (p, v, z).

    This function calculates the (scaled) WFR energy or D_delta(mu) in the
    original paper by Σ(p|v|²+δ²pz²)

    Args:
        p (torch.Tensor of size (T, N1, N2, ..., N_n)): The mass distribution over time.

        v (torch.Tensor of size (T, N1, N2, ..., N_n, n)): The velocity field over time.

        z (torch.Tensor of size (T, N1, N2, ..., N_n))): The source function over time.

        delta: the interpolation parameter for WFR.

    Returns:
        wfr: torch.FloatTensor of size (1,)
            The scaled Wasserstein-Fisher-Rao energy. To get the exact WFR, you need to \
            multiply this quantity by 1/2, the size of the space steps (dx) and the size \
            of the time steps (dt).
    """

    # ToDo: Comment out the input check once everything works
    if p.shape != v.shape[:-1] or p.shape != z.shape or v.shape[:-1] != z.shape:
        raise TypeError("p.shape, v.shape[:-1] and z.shape should all match")

    if len(v.shape) - 2 != v.shape[-1]:
        raise TypeError(
            "The dimension of the grid and the dimension of the vector does not match"
        )

    v_norm = torch.norm(v, dim=-1)
    wfr = torch.sum(p * v_norm**2 + delta**2 * p * z**2)

    return wfr


def _project_affine(
    p: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    H: torch.Tensor,
    dHdt: torch.Tensor,
    gradH: list,
    Fprime: torch.Tensor,
    dx: list,
):
    """Given an unconstrained v and z, returns v and z projected onto the constraint
    int H(t,x)dp_t = F(t).

    By taking the derivative of both sides of the constraint and applying the continuity \
    equation, we have

    int (grad H * v + Hz)pdx = F'-int (dH/dt)pdx

    for all t (We assume some regularity on p, v, z, h and f). This function projects v,z
    to the affine space defined by the equation above.

    We will see the constraint above as an affine equation

    <c, x> = b

    where c = (c1,c2) = (p gradH, pH), x = (v, z), b= F'-int (dH/dt)pdx at a given time \
    and projects x onto this set by

    proj(x) = x - (<c,x>-b)c/|c|^2

    Args:
        p (torch.Tensor of shape (N_1,...,N_n)) : The density at a given time.

        v (torch.Tensor of shape (N_1,...,N_n, n)) : The velocity field.

        z (torch.Tensor of shape (N_1,...,N_n)) : The source field.

        H (torch.Tensor of shape (N_1,...,N_n)) : The H function.

        gradH (list of Tensors of shape (N_1,...,N_n)) : The gradient of H. H[i]\
        is assumed to be the derivative of H function w.r.t. the ith space variable.

        Fprime (torch.Tensor of shape ()) : The derivative of the F function. \

        dx (list of floats) : The list of space steps.

    All of the functions are assumed to be evaluated at the same time.
    """

    c1 = torch.stack([p * gradH_component for gradH_component in gradH], dim=-1)
    c2 = p * H
    b = Fprime - (dHdt * p).sum() * math.prod(dx)

    cdotx = ((c1 * v).sum() + (c2 * z).sum()) * math.prod(dx)
    c_norm_sq = (torch.norm(c1) ** 2 + torch.norm(c2) ** 2) * math.prod(dx)

    if c_norm_sq == 0:
        v_new = v
        z_new = z
    else:
        v_new = v - (cdotx - b) * c1 / c_norm_sq
        z_new = z - (cdotx - b) * c2 / c_norm_sq

    return v_new, z_new


def _batch_project_affine(
    p: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    H: torch.Tensor,
    dHdt: torch.Tensor,
    gradH: list,
    Fprime: torch.Tensor,
    dx: list,
):
    """Given p, v, z at all time, returns the projected v, z at all time. In other words,\
    we repeatedly apply _proejct_affine for each time.

    Args:
        p: torch.Tensor of shape (T+1, N1, N2, ..., N_n)
            The mass distribution at each time.

        v: torch.Tensor of shape (T, N1, N2, ..., N_n, n)
            The vector field for all time.

        z: torch.Tensor of shape (T, N1, N2, ..., N_n)
            The source function for all time.

        H (torch.Tensor of shape (T+1, N_1,...,N_n)) : The H funtion in the constraint. \
        H[i] is H(i/T, x). if any of H,  dHdt, gradH and Fprime is `None`, we assume \
        there is no constraint.

        dH/dt (torch.Tensor of shape (T+1, N_1,...,N_n)) : The derivative of H function \
        in the constraint. dHdt[i] is dH/dt at t=i/T. if any of H, dHdt, gradH and Fprime\
        is `None`, we assume there is no constraint.

        gradH (list of n Tensors of shape (T+1, N_1,....,N_n)): The gradient of the H \
        function in the constraint. gradH[i][j] is the derivative of H by the ith space\
        variable evaluated at t=j/T. If any of H, dHdt, gradH and Fprime is\
        `None`, we assume there is no constraint.

        Fprime (torch.Tensor of shape (T+1)) : The F function in the constraint. F[i] is \
        F at t=i/T. Fprime[i] is F'(t) at t=i/T. if any of H,  dHdt, gradH and Fprime is\
        `None`, we assume there is no constraint.

        dx (list of floats) : The space steps in each dimension.
    """

    new_v = torch.zeros_like(v)
    new_z = torch.zeros_like(z)

    T = p.shape[0] - 1
    if all(var is not None for var in [H, dHdt, gradH, Fprime]):
        for time_step in range(T):
            new_v[time_step], new_z[time_step] = _project_affine(
                p[time_step],
                v[time_step],
                z[time_step],
                H[time_step],
                dHdt[time_step],
                [component[time_step] for component in gradH],
                Fprime[time_step],
                dx,
            )
    else:
        new_v = v
        new_z = z

    return new_v, new_z


def _div_plus_pz_grid(
    t: torch.Tensor,
    p: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    dx: list,
    T: int,
    H: torch.Tensor = None,
    dHdt: torch.Tensor = None,
    gradH: list = None,
    Fprime: torch.Tensor = None,
):
    """Calculates -div(pv)+pz given p, v and z where div is the Euclidean divergence.

    Args:
        t: torch.Tensor of shape (1,)
            The time to evaluate -div(pv)+pz.

        p: torch.Tensor of shape (N1, N2, ..., N_n)
            The mass distribution at time t.

        v: torch.Tensor of shape (T, N1, N2, ..., N_n, n)
            The vector field for all time.

        z: torch.Tensor of shape (T, N1, N2, ..., N_n)
            The source function for all time.

        dx: list of floats
                The step size in each spatial direction for the grid.
        T: int
            The grid size in time. The step size in time is defined by 1/T.

        H (torch.Tensor of shape (T+1, N_1,...,N_n)) : The H funtion in the constraint. \
        H[i] is H(i/T, x). if any of H,  dHdt, gradH and Fprime is `None`, we assume \
        there is no constraint.

        dH/dt (torch.Tensor of shape (T+1, N_1,...,N_n)) : The derivative of H function \
        in the constraint. dHdt[i] is dH/dt at t=i/T. if any of H, dHdt, gradH and Fprime\
        is `None`, we assume there is no constraint.

        gradH (list of n Tensors of shape (T+1, N_1,....,N_n)): The gradient of the H \
        function in the constraint. gradH[i][j] is the derivative of H by the ith space\
        variable evaluated at t=j/T. If any of H, dHdt, gradH and Fprime is\
        `None`, we assume there is no constraint.

        Fprime (torch.Tensor of shape (T+1)) : The F function in the constraint. F[i] is \
        F at t=i/T. Fprime[i] is F'(t) at t=i/T. if any of H,  dHdt, gradH and Fprime is\
        `None`, we assume there is no constraint.

        Returns:
            torch.Tensor of shape (N1, ..., Nn): -div(pv)+pz at time t.

    """
    # For a given t, the following lines finds the index of the closest discretization
    # points i/T where i = 0,...,T-1. We use the vector field/source term at the closest
    # point as the value at t. Basically, we treat the vector field/source term as a
    # histogram centered at each grid
    time_step_num = torch.round(t * T).int()
    time_step_num = min(time_step_num, T - 1)  # avoid rounding to t=T

    spatial_dim = len(dx)

    # Apply the constraint
    if all(var is not None for var in [H, dHdt, gradH, Fprime]):
        _v, _z = _project_affine(
            p,
            v[time_step_num],
            z[time_step_num],
            H[time_step_num],
            dHdt[time_step_num],
            [component[time_step_num] for component in gradH],
            Fprime[time_step_num],
            dx,
        )
    else:
        _v = v[time_step_num]
        _z = z[time_step_num]

    # pre_div represents the list of (pv_i(t, ...x_i+dx_i...)-pv_i(t, ...x_i-dx_i...))
    # /2dx_i
    # in each dimension i.e. a numerical approximation of
    # (∂pv_1/∂x1, ∂pv_2/∂x2, ..., ∂pv_n/∂xn)
    pre_div = [
        (torch.roll(p * _v[..., i], -1, i) - torch.roll(p * _v[..., i], 1, i))
        / (2 * dx[i])
        for i in range(spatial_dim)
    ]

    divpv = sum(pre_div)

    return -divpv + p * _z


def wfr_grid(
    p1: torch.Tensor,
    p2: torch.Tensor,
    delta: float,
    rel: float,
    T: float,
    dx: list = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    num_iter: int = 1000,
    solver: str = "euler",
    optim_class: torch.optim.Optimizer = torch.optim.SGD,
    **optim_params,
):
    """Calculates the Wasserstein-Fisher-Rao distance between two (discretized)
    measures defined on a rectangular periodic grid.

    Args:
        p1 (torch.Tensor): The discretized measure to calculate the distance between \
        p2. The size of p1 and p2 implicitly defines the shape of the grid.

        p1 (torch.Tensor): The discretized measure to calculate the distance between \
        p2. The size of p1 and p2 implicitly defines the shape of the grid.

        delta (float): The interpolation parameter for WFR.

        rel (float): The relaxation constant.

        T (int):  The grid size in time. The step size in time is defined by 1/T.

        dx (list of floats), default = None: The step size in each spatial direction \
            for the grid. The number of elements should match the number of \
            dimensions of p1, p2. if None, dx = [1/N_1, ..., 1/N_n] where \
            (N_1, ..., N_n) is the shape of p1,p2.

        atol (float), default = 1e-8: The absolute tolerance for convergence check.

        rtol (float), default = 1e-5: The relative tolerance for convergence check.

        num_iter (int), default = 1000: The maximal number of iterations.

        solver (str), default = 'euler': The ODE solver used for torchdiffeq.

        optim_class (torch.optim.Optimizer), default = 'torch.optim.SGD': The scipy \
              optimizer to use. Currently, only `lbfgs` is supported.

        **optim_params: The parameters to be passed to the optimizer.

    Returns:
        wfr (torch.FloatTensor): The Waserstein-Fisher-Rao distance between p1 and p2.

        p (torch.Tensor of shape (T+1, N_1,...., N_n)): The interpolated measures at \
            each time. p[i] is the measure at time t=i/T.

        v (torch.Tensor of shape (T, N_1,...., N_n, n)): The velocity vector field at\
            each time. v[i] is the vector field at time t=i/T.

        z (torch.Tensor of shape (T, N_1,...., N_n)): The source field at\
            each time. z[i] is the source field at time t=i/T.
    """

    if p1.shape != p2.shape:
        raise TypeError("The shape of p1 and p2 should match")

    if rel < 0:
        raise ValueError("The relaxation constant should be nonnegative")

    if dx is None:
        dx = [1.0 / n for n in p1.shape]

    if len(dx) != len(p1.shape):
        raise TypeError("The spatial dimension of dx and p1, p2 should match")

    if torch.cuda.is_available():
        torch_device = "cuda:0"
        print("Using GPU")
    else:
        torch_device = "cpu"

    if optim_class == torch.optim.LBFGS:
        raise NotImplementedError("We currently do not support LBFGS.")

    spatial_dim = len(p1.shape)
    v_shape = (T,) + p1.shape + (spatial_dim,)
    z_shape = (T,) + p1.shape

    # Initialization of v and z
    v = torch.zeros(v_shape, requires_grad=True).to(torch_device)
    z = torch.zeros(z_shape, requires_grad=True).to(torch_device)

    # Initialize the optimizer
    optimizer = optim_class([v, z], **optim_params)

    for _ in range(num_iter):
        optimizer.zero_grad()

        # Solve the continuity equation
        divpz = functools.partial(_div_plus_pz_grid, v=v, z=z, dx=dx, T=T)
        p = torchdiffeq.odeint(divpz, p1, torch.linspace(0, 1, T + 1), method=solver)

        # Find the loss
        loss = _WFR_energy(p[:-1], v, z, delta) + rel * torch.norm(p[-1] - p2)
        loss.backward()

        prev_v = v.clone().detach()
        prev_z = z.clone().detach()
        optimizer.step()

        # Check convergence
        if torch.allclose(v, prev_v, atol=atol, rtol=rtol) and torch.allclose(
            z, prev_z, atol=atol, rtol=rtol
        ):
            print("Early break at iteration", _)
            break

    p = torchdiffeq.odeint(
        divpz, p1, torch.linspace(0, 1.0, steps=T + 1), method=solver
    ).detach()
    prod_dx = math.prod(dx)
    dt = 1.0 / T
    wfr = torch.sqrt(0.5 * _WFR_energy(p[:-1], v, z, delta) * prod_dx * dt).detach()

    return wfr, p, v, z


def wfr_grid_scipy(
    p1: np.ndarray,
    p2: np.ndarray,
    delta: float,
    rel: float,
    T: float,
    v0: np.ndarray = None,
    z0: np.ndarray = None,
    dx: list[float] = None,
    H: np.ndarray = None,
    F: np.ndarray = None,
    num_iter: int = 1000,
    solver: str = "euler",
    optim: str = "lbfgs",
    **optim_params,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the Wasserstein-Fisher-Rao distance between two (discretized)
    measures defined on a rectangular periodic grid.
    The difference between the `dynamic_wfr_relaxed_grid` function is that we use
    the scipy optimizer for optimization. We still use PyTorch for gradient
    calculation.

    This implementation is heavily influenced by Emanuel Hartman's code.
    The reason why he didn't use PyTorch LBFGS was that he had issues with the line \
    search, saying that 'it only does only one line search'.

    Args:
        p1 (np.ndarray): The discretized density to calculate the distance between \
        p2. The size of p1 and p2 implicitly defines the size of the grid. \
        We note that p1, p2 are assumed to be a DENSITY, that is, \
        (p1.sum())*math.prod(dx)=1.

        p1 (np.ndarray): The discretized density to calculate the distance between \
        p2. The size of p1 and p2 implicitly defines the size of the grid.

        delta (float): The interpolation parameter for WFR.

        rel (float): The relaxation constant.

        T (int):  The grid size in time. The step size in time is defined by 1/T.

        v0 (np.ndarray of shape (T, N_1,..., N_n, n)): The initial guess for the vector \
        field. If None, it will default to the zero array. (N_1, ..., N_n) is the shape \
        of p1,p2.

        z0 (np.ndarray of shape (T, N_1, ..., N_n)) : The initial guess for the source \
        field. If None, it will default to the zero array. (N_1, ..., N_n) is the shape \
        of p1,p2.

        dx (list of floats), default = None: The step size in each spatial direction \
            for the grid. The number of elements should match the number of \
            dimensions of p1, p2. if None, dx = [1/N_1, ..., 1/N_n] where \
            (N_1, ..., N_n) is the shape of p1,p2.

        H (np.ndarray of shape (T+1, N_1,..,N_n)): The H function in the affine \
        constraint int H(t,x) dp_t = F(t). (N_1,...,N_n) is the shape of p1, p2. \
        H[i] corresponds to H(i/T, x).

        F (np.ndarray of shape (T+1,)): The H function in the affine constraint \
        int H(t,x) dp_t = F(t).  F[i] corresponds to F(i/T).

        num_iter (int), default = 1000: The maximal number of iterations.

        solver (str), default = 'euler': The ODE solver used for torchdiffeq.

        optim (str), default = 'lbfgs': The scipy optimizer to use. \
            Currently, only `lbfgs` is supported.

    Returns:
        wfr (float): The Waserstein-Fisher-Rao distance between p1 and p2.

        p (np.ndarray of shape (T+1, N_1,...., N_n)): The interpolated measures at \
            each time. p[i] is the measure at time t=i/T. (N_1, ..., N_n) is the shape
            of p1 or p2.

        v (np.ndarray of shape (T, N_1,...., N_n, n)): The velocity vector field at\
            each time. v[i] is the vector field at time t=i/T.

        z (np.ndarray of shape (T, N_1,...., N_n)): The source field at\
            each time. z[i] is the source field at time t=i/T.
    """

    if p1.shape != p2.shape:
        raise TypeError("The shape of p1 and p2 should match")

    if rel < 0:
        raise ValueError("The relaxation constant should be nonnegative")

    if T <= 0:
        raise ValueError("T should be positive")

    if dx is None:
        dx = [1.0 / n for n in p1.shape]

    if len(dx) != len(p1.shape):
        raise TypeError("The spatial dimension of dx and p1, p2 should match")

    if optim != "lbfgs":
        raise NotImplementedError("Currently, only lbfgs is supported.")

    if not isinstance(p1, np.ndarray) or not isinstance(p2, np.ndarray):
        raise TypeError("p1 and p2 should be numpy arrays")

    if torch.cuda.is_available():
        torch_device = "cuda:0"
        print("Using GPU")
    else:
        torch_device = "cpu"

    spatial_dim = len(p1.shape)
    v_shape = (T,) + p1.shape + (spatial_dim,)
    z_shape = (T,) + p1.shape

    # Initialization of v and z
    if v0 is None:
        v = np.zeros(v_shape)
    else:
        if v_shape != v0.shape:
            raise TypeError(f"The shape of v0 should be: {v_shape}")
        v = v0

    if z0 is None:
        z = np.zeros(z_shape)
    else:
        if z_shape != z0.shape:
            raise TypeError(f"The shape of z0 should be: {z_shape}")
        z = z0

    vz = np.concatenate([v.flatten(), z.flatten()])

    # Constraint initialization
    if H is not None and F is not None:
        if H.shape[0] != T + 1:
            raise TypeError("The first dimension of H should be T+1")
        if H.shape[1:] != p1.shape:
            raise TypeError("The spatial shape of H does not match p1.shape")
        if F.shape != (T + 1,):
            raise TypeError("The shape of F should be (T+1,)")

        H_torch = torch.from_numpy(H)

        # Forward difference in time
        Fprime = T * (np.roll(F, -1) - F)
        Fprime_torch = torch.from_numpy(Fprime)

        # Forward difference in time
        dHdt = T * (np.roll(H, -1, 0) - H)
        dHdt_torch = torch.from_numpy(dHdt)

        # Central difference in space
        gradH = [
            (np.roll(H, -1, i + 1) - np.roll(H, 1, i + 1)) / (2 * dx[i])
            for i in range(spatial_dim)
        ]

        gradH_torch = [torch.from_numpy(component) for component in gradH]
    else:
        H_torch = None
        Fprime_torch = None
        dHdt_torch = None
        gradH_torch = None

    # If p1, p2 does not have the dtype `float64`, this will create a copy of them.
    # Otherwise, the torch version share the memory with the original.
    # We make a torch version to pass it to torchdiffeq.
    p1_torch = torch.from_numpy(p1).to(device=torch_device, dtype=torch.float64)
    p2_torch = torch.from_numpy(p2).to(device=torch_device, dtype=torch.float64)

    def loss_torch(_vz: torch.Tensor):
        """The loss function to be passed to torch.autograd.grad."""

        _v = _vz[: math.prod(v_shape)].reshape(v_shape)
        _z = _vz[math.prod(v_shape) :].reshape(z_shape)

        # Solve the continuity equation
        divpz = functools.partial(
            _div_plus_pz_grid,
            v=_v,
            z=_z,
            dx=dx,
            T=T,
            H=H_torch,
            Fprime=Fprime_torch,
            dHdt=dHdt_torch,
            gradH=gradH_torch,
        )
        _p = torchdiffeq.odeint(
            divpz,
            p1_torch,
            torch.linspace(0, 1, steps=T + 1),
            method=solver,
        )

        _v, _z = _batch_project_affine(
            _p, _v, _z, H_torch, dHdt_torch, gradH_torch, Fprime_torch, dx
        )

        # Absolute value on p to avoid divergence to negative infinity
        loss = _WFR_energy(torch.abs(_p[:-1]), _v, _z, delta) + rel * torch.norm(
            _p[-1] - p2_torch
        )
        return loss

    def loss_scipy(_vz: np.ndarray):
        """The loss function for optimization to be passed to the scipy optimizer."""
        loss = loss_torch(
            torch.from_numpy(_vz).to(device=torch_device, dtype=torch.float64)
        )
        return float(loss.detach().cpu().numpy())

    def grad_loss_scipy(_vz: np.ndarray):
        """The gradient of the loss function to be passed to the scipy optimizer."""
        _vz_torch = torch.from_numpy(_vz).to(device=torch_device, dtype=torch.float64)
        _vz_torch.requires_grad_()
        (gradient,) = torch.autograd.grad(loss_torch(_vz_torch), _vz_torch)
        return gradient.detach().cpu().numpy().flatten().astype("float64")

    vz_optimal, _, _ = sp.optimize.fmin_l_bfgs_b(
        loss_scipy,
        vz,
        fprime=grad_loss_scipy,
        maxiter=num_iter,
        **optim_params,
    )

    v = vz_optimal[: math.prod(v_shape)].reshape(v_shape)
    z = vz_optimal[math.prod(v_shape) :].reshape(z_shape)
    torch_v = torch.from_numpy(v)
    torch_z = torch.from_numpy(z)

    # Solve the continuity equation one last time to get the final solution
    divpz = functools.partial(
        _div_plus_pz_grid,
        v=torch_v,
        z=torch_z,
        dx=dx,
        T=T,
        H=H_torch,
        dHdt=dHdt_torch,
        gradH=gradH_torch,
        Fprime=Fprime_torch,
    )
    torch_p = torchdiffeq.odeint(
        divpz, p1_torch, torch.linspace(0, 1, steps=T + 1), method=solver
    )

    # Apply the constraint
    torch_v, torch_z = _batch_project_affine(
        torch_p, torch_v, torch_z, H_torch, dHdt_torch, gradH_torch, Fprime_torch, dx
    )

    # Calculate the WFR distance (or the square root of the optimal energy)
    prod_dx = math.prod(dx)
    dt = 1.0 / T
    wfr = torch.sqrt(
        0.5 * _WFR_energy(torch_p[:-1], torch_v, torch_z, delta) * prod_dx * dt
    ).numpy()

    return float(wfr), torch_p.numpy(), torch_v.numpy(), torch_z.numpy()


def wfr_grid_scipy_tunerel(
    p1: np.ndarray,
    p2: np.ndarray,
    delta: float,
    T: float,
    rel_start: float = 0.0,
    rel_stop: float = 10.0,
    rel_step: float = 1.0,
    dx: list[float] = None,
    H: np.ndarray = None,
    F: np.ndarray = None,
    num_iter: int = 1000,
    solver: str = "euler",
    optim: str = "lbfgs",
    **optim_params,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Run wfr_grid_scipy and automatically tune the relaxation parameter. We gradually \
    increase the relaxation parameter and stops when the WFR distance reaches plateau. \
    The iterable for the relaxation parameters is created by np.arange. The tuning\
    terminates when the latest wfr is lower than the previous.

    Args:
        p1 (np.ndarray): The discretized density to calculate the distance between \
        p2. The size of p1 and p2 implicitly defines the size of the grid.

        p1 (np.ndarray): The discretized density to calculate the distance between \
        p2. The size of p1 and p2 implicitly defines the size of the grid.

        delta (float): The interpolation parameter for WFR.

        T (int):  The grid size in time. The step size in time is defined by 1/T.

        rel_start (float): The initial of the relaxation constant.

        rel_stop (float): The last relaxation constant to check.

        rel_step (float): The step of the relaxation parameter.

        dx (list of floats), default = None: The step size in each spatial direction \
            for the grid. The number of elements should match the number of \
            dimensions of p1, p2. if None, dx = [1/N_1, ..., 1/N_n] where \
            (N_1, ..., N_n) is the shape of p1,p2.

        H (np.ndarray of shape (T+1, N_1,..,N_n)): The H function in the affine \
        constraint int H(t,x) dp_t = F(t). (N_1,...,N_n) is the shape of p1, p2. \
        H[i] corresponds to H(i/T, x).

        F (np.ndarray of shape (T+1,)): The H function in the affine constraint \
        int H(t,x) dp_t = F(t).  F[i] corresponds to F(i/T).

        num_iter (int), default = 1000: The maximal number of iterations.

        solver (str), default = 'euler': The ODE solver used for torchdiffeq.

        optim (str), default = 'lbfgs': The scipy optimizer to use. \
            Currently, only `lbfgs` is supported.

    Returns:
        wfr (float): The Waserstein-Fisher-Rao distance between p1 and p2 at the \
        relaxation constant found.

        p (np.ndarray of shape (T+1, N_1,...., N_n)): The interpolated measures at \
            each time. p[i] is the measure at time t=i/T. (N_1, ..., N_n) is the shape
            of p1 or p2.

        v (np.ndarray of shape (T, N_1,...., N_n, n)): The velocity vector field at\
            each time. v[i] is the vector field at time t=i/T.

        z (np.ndarray of shape (T, N_1,...., N_n)): The source field at\
            each time. z[i] is the source field at time t=i/T.

        best_rel (float) : The relaxation constant found.
    """

    rels = np.arange(rel_start, rel_stop, rel_step)
    wfr_prev = 0.0

    if dx is None:
        dx = [1.0 / n for n in p1.shape]

    for rel in rels:
        wfr, p, v, z = wfr_grid_scipy(
            p1, p2, delta, rel, T, dx, H, F, num_iter, solver, optim, **optim_params
        )

        if wfr <= wfr_prev:  # If no longer increasing
            best_rel = max(rel_start, rel - rel_step)  # Get the previous rel
            break

        wfr_prev = wfr

    # See if best_rel is undefined
    try:
        best_rel
    except NameError:
        print(
            "Did not terminate until the last rel. Try increasing rel_stop and/or\
             rel_step."
        )
        best_rel = rels[-1]

    return wfr, p, v, z, best_rel
