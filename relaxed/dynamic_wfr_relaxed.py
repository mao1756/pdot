import torch
import torchdiffeq


def WFR_energy(p: torch.Tensor, v: torch.Tensor, z: torch.Tensor):
    """Calculates the Wasserstein-Fisher-Rao energy for the path (p, v, z).

    This function calculates the WFR energy or D_delta(mu) in the original paper by
    (1/2)Σ(p|v|²+δ²pz²)dxdt

    Args:
        p: torch.Tensor of size (T, N1, N2, ..., N_n)
            The mass distribution over time.
        v: torch.Tensor of size (T, N1, N2, ..., N_n, n)
            The velocity field over time.
        z: (T, N1, N2, ..., N_n)
    """


def dynamic_wfr_relaxed_unconstrained_grid(
    p1: torch.Tensor, p2: torch.Tensor, rel: float, dx: list, dt: float
):
    """Calculates the Wasserstein-Fisher-Rao distance between two (discretized) measures defined on a rectangular grid.

    Args:
        p1, p2: torch.Tensor
            The discretized measures to calculate the distance.
            The size of p1 and p2 implicitly defines the size of the grid.

        rel: float
            The relaxation constant.

        dx: list of floats
            The step size in each spatial direction for the grid.
            The number of elements should match the number of dimensions of p1, p2.

        dt: float
            The step size in time.

    Returns:
        WFR: torch.FloatTensor
            The Waserstein-Fisher-Rao distance between p1 and p2.
    """
