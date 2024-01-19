import torch
import torchdiffeq
import numpy as np
import scipy as sp
import functools
import math


def _WFR_energy(
    p: torch.Tensor, v: torch.Tensor, z: torch.Tensor, delta: torch.FloatTensor
):
    """Calculates the Wasserstein-Fisher-Rao energy for the path (p, v, z).

    This function calculates the (scaled) WFR energy or D_delta(mu) in the
    original paper by Σ(p|v|²+δ²pz²)

    Args:
        p (torch.Tensor of size (T, N1, N2, ..., N_n)): The mass distribution over time.

        v (torch.Tensor of size (T, N1, N2, ..., N_n, n)): The velocity field over time.

        z (torch. Tensor of size (T, N1, N2, ..., N_n))): The source function over time.

        delta: the interpolation parameter for WFR.

    Returns:
        wfr: torch.FloatTensor of size (1,)
            The scaled Wasserstein-Fisher-Rao energy.
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


def _div_plus_pz_grid(
    t: torch.Tensor, p: torch.Tensor, v: torch.Tensor, z: torch.Tensor, dx: list, T: int
):
    """Calculates -div(pv)+pz given p, v and z where div is the Euclidean divergence.

    t: torch.Tensor of shape (1,)
        The time to evaluate.

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
    """
    # For a given t, the following lines finds the index of the closest discretization
    # points i/T where i = 0,...,T-1. We use the vector field at the closest point as the
    # value at t.

    time_step_num = torch.round(t * T).int()
    time_step_num = min(time_step_num, T - 1)  # avoid rounding to t=T

    spatial_dim = len(dx)

    # pre_div represents the list of (pv_i(t, ...x_i+dx_i...)-pv_i(t, ...x_i-dx_i...))
    # /2dx_i
    # in each dimension i.e. a numerical approximation of
    # (∂pv_1/∂x1, ∂pv_2/∂x2, ..., ∂pv_n/∂xn)
    pre_div = [
        (
            torch.roll(p * v[time_step_num, ..., i], -1, i)
            - torch.roll(p * v[time_step_num, ..., i], 1, i)
        )
        / (2 * dx[i])
        for i in range(spatial_dim)
    ]

    divpv = sum(pre_div)

    return -divpv + p * z[time_step_num]


def dynamic_wfr_relaxed_grid(
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
    **optim_params
):
    """Calculates the Wasserstein-Fisher-Rao distance between two (discretized)
    measures defined on a rectangular grid.
    The difference between the `dynamic_wfr_relaxed_grid` function is that we use
    the scipy optimizer for optimization. We still use PyTorch for gradient
    calculation.

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

        num_iter (int), default = 1000: The number of iterations.

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

    if rel <= 0:
        raise ValueError("The relaxation constant should be positive")

    if dx is None:
        dx = [1.0 / n for n in p1.shape]

    if len(dx) != len(p1.shape):
        raise TypeError("The spatial dimension of dx and p1, p2 should match")

    if torch.cuda.is_available():
        device = "cuda:0"
        print("Using GPU")
    else:
        device = "cpu"

    spatial_dim = len(p1.shape)
    v_shape = (T,) + p1.shape + (spatial_dim,)
    z_shape = (T,) + p1.shape

    # Initialization of v and z
    v = torch.zeros(v_shape, requires_grad=True).to(device)
    z = torch.zeros(z_shape, requires_grad=True).to(device)

    # Initialize the optimizer
    optimizer = optim_class([v, z], **optim_params)

    for _ in range(num_iter):
        optimizer.zero_grad()

        # Solve the continuity equation
        divpz = functools.partial(_div_plus_pz_grid, v=v, z=z, dx=dx, T=T)
        p = torchdiffeq.odeint(
            divpz, p1, torch.arange(0, 1.0 + 1.0 / T, 1.0 / T), method=solver
        )

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
        divpz, p1, torch.arange(0, 1.0 + 1.0 / T, 1.0 / T), method=solver
    ).detach()
    prod_dx = math.prod(dx)
    dt = 1.0 / T
    wfr = torch.sqrt(0.5 * _WFR_energy(p[:-1], v, z, delta) * prod_dx * dt).detach()

    return wfr, p, v, z


def dynamic_wfr_relaxed_grid_scipy(
    p1: torch.Tensor,
    p2: torch.Tensor,
    delta: float,
    rel: float,
    T: float,
    dx: list[float] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    num_iter: int = 1000,
    solver: str = "euler",
    optim: str = "lbfgs",
    **optim_params
):
    """Calculates the Wasserstein-Fisher-Rao distance between two (discretized)
    measures defined on a rectangular grid.
    The difference between the `dynamic_wfr_relaxed_grid` function is that we use
    the scipy optimizer for optimization. We still use PyTorch for gradient
    calculation.

    The implementation is heavily influenced by Emanuel Hartman's code.
    The reason why he didn't use PyTorch LBFGS was that he had issues with the line \
    search, saying that 'it only does only one line search'.

    Args:
        p1 (torch.Tensor): The discretized measure to calculate the distance between \
        p2. The size of p1 and p2 implicitly defines the size of the grid.

        p1 (torch.Tensor): The discretized measure to calculate the distance between \
        p2. The size of p1 and p2 implicitly defines the size of the grid.

        delta (float): The interpolation parameter for WFR.

        rel (float): The relaxation constant.

        T (int):  The grid size in time. The step size in time is defined by 1/T.

        dx (list of floats), default = None: The step size in each spatial direction \
            for the grid. The number of elements should match the number of \
            dimensions of p1, p2. if None, dx = [1/N_1, ..., 1/N_n] where \
            (N_1, ..., N_n) is the shape of p1,p2.

        atol (float), default = 1e-8: The absolute tolerance for convergence check.

        rtol (float), default = 1e-5: The relative tolerance for convergence check.

        num_iter (int), default = 1000: The number of iterations.

        solver (str), default = 'euler': The ODE solver used for torchdiffeq.

        optim (str), default = 'lbfgs': The scipy optimizer to use. \
            Currently, only `lbfgs` is supported.

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

    if rel <= 0:
        raise ValueError("The relaxation constant should be positive")

    if dx is None:
        dx = [1.0 / n for n in p1.shape]

    if len(dx) != len(p1.shape):
        raise TypeError("The spatial dimension of dx and p1, p2 should match")

    if optim != "lbfgs":
        raise NotImplementedError("Currently, only lbfgs is supported.")

    spatial_dim = len(p1.shape)
    v_shape = (T,) + p1.shape + (spatial_dim,)
    z_shape = (T,) + p1.shape

    # Initialization of v and z
    v = torch.zeros(v_shape)
    z = torch.zeros(z_shape)
    vz = torch.cat([v.flatten(), z.flatten()], dim=0)

    def obj(_vz: np.ndarray):
        """The objective function for optimization to be passed to the optimizer.

        _vz (np.ndarray): The concatenation of flattened v and z.

        """

        # ToDo: apply constraint here
        _v = torch.from_numpy(_vz[: len(v_shape)].reshape(v_shape))
        _z = torch.from_numpy(_vz[len(v_shape) :].reshape(z_shape))

        # Solve the continuity equation
        divpz = functools.partial(_div_plus_pz_grid, v=_v, z=_z, dx=dx, T=T)
        _p = torchdiffeq.odeint(
            divpz, p1, torch.arange(0, 1.0 + 1.0 / T, 1.0 / T), method=solver
        )

        loss = _WFR_energy(_p[:-1], _v, _z, delta) + rel * torch.norm(_p[-1] - p2)
        return float(loss.detach().cpu().numpy())

    def grad_obj(_vz: np.ndarray):
        """The gradient of the objective function at _vz."""
