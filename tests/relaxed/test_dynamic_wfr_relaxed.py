import torch
import numpy as np
import pytest
import relaxed.dynamic_wfr_relaxed as dynamic_wfr_relaxed
import math
import random

"""
Test of _WFR_energy
"""


def test_WFR_energy_1Dinput():
    p = torch.tensor([3.0]).reshape(1, 1)
    v = torch.tensor([4.0]).reshape(1, 1, 1)
    z = torch.tensor([-1]).reshape(1, 1)
    energy = dynamic_wfr_relaxed._WFR_energy(p, v, z, delta=1)
    torch.testing.assert_close(energy, torch.tensor(51.0))


def test_WFR_energy_zeroinput():
    p = torch.tensor([0]).reshape(1, 1)
    v = torch.tensor([4.0]).reshape(1, 1, 1)
    z = torch.tensor([-1]).reshape(1, 1)
    energy = dynamic_wfr_relaxed._WFR_energy(p, v, z, delta=1)
    torch.testing.assert_close(energy, torch.tensor(0.0))


def test_WFR_energy_1dlonginput():
    p = torch.tensor([[0.1, 3.0, 0.5], [0.2, 0.3, 0.2], [7.0, 3.0, 4.0]])
    v = torch.tensor([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [0.0, 0.0, 1]]).reshape(3, 3, 1)
    z = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.2, 0.3]])
    energy = dynamic_wfr_relaxed._WFR_energy(p, v, z, delta=1)
    torch.testing.assert_close(energy, torch.tensor(14.893))


def test_WFR_energy_wrong_size():
    p = torch.tensor([[0.1, 3.0, 0.5], [0.2, 0.3, 0.2], [7.0, 3.0, 4.0]])
    v = torch.tensor(
        [
            [[1.0, 1], [1.0, 1], [1.0, 1]],
            [[2.0, 1], [3.0, 3], [4.0, 3]],
            [[0.0, 2], [0.0, 1], [3.0, 4.0]],
        ]
    ).reshape(3, 3, 2)
    z = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.2, 0.3]])
    with pytest.raises(TypeError):
        dynamic_wfr_relaxed._WFR_energy(p, v, z, delta=1)


def test_WFR_energy_incorrect_v():
    p = torch.tensor([[0.1, 3.0, 0.5], [0.2, 0.3, 0.2], [7.0, 3.0, 4.0]])
    v = torch.tensor([4.0]).reshape(1, 1, 1)
    z = torch.tensor([-1]).reshape(1, 1)
    with pytest.raises(TypeError):
        dynamic_wfr_relaxed._WFR_energy(p, v, z, delta=1)


"""
Test of _project_affine
"""


def test_project_affine_allzero():
    """Pass the zero input for all of the variables to check if it gives us any error."""

    time_step_num = 5
    T = 10
    n = 3
    grid_shape = (10, 61, 21)
    p = torch.zeros(grid_shape)
    v = torch.zeros(grid_shape + (n,))
    z = torch.zeros(grid_shape)
    H = torch.zeros((T + 1,) + grid_shape)
    F = torch.zeros((T + 1,))
    dx = [1 / s for s in grid_shape]
    spatial_dim = len(dx)
    # Forward difference in time
    Fprime = T * (torch.roll(F, -1) - F)
    Fprime = Fprime[time_step_num]

    # Forward difference in time
    dHdt = T * (torch.roll(H, -1, 0) - H)
    dHdt = dHdt[time_step_num]

    # Central difference in space
    gradH = [
        (torch.roll(H[time_step_num], -1, i) - torch.roll(H[time_step_num], 1, i))
        / (2 * dx[i])
        for i in range(spatial_dim)
    ]

    v_new, z_new = dynamic_wfr_relaxed._project_affine(
        p, v, z, H, dHdt, gradH, Fprime, dx
    )

    torch.testing.assert_close(v_new, v)
    torch.testing.assert_close(z_new, z)


def test_project_affine_constantmass():
    """Pass a random v, z and enforce the constraint rho_t(omega) = 1, that is, H=1, F=1.
    We expect the inner product between p and z to be zero (= net mass change is zero).
    """
    time_step_num = 7
    T = 100
    N1 = 31
    N2 = 57
    grid_shape = (N1, N2)
    n = len(grid_shape)
    p = torch.abs(torch.randn(grid_shape))
    v = torch.randn(grid_shape + (n,))
    z = torch.randn(grid_shape)
    H = torch.ones((T + 1,) + grid_shape)
    F = torch.ones((T + 1,))
    dx = [1.0 / s for s in grid_shape]

    spatial_dim = len(dx)
    # Forward difference in time
    Fprime = T * (torch.roll(F, -1) - F)
    Fprime = Fprime[time_step_num]

    # Forward difference in time
    dHdt = T * (torch.roll(H, -1, 0) - H)
    dHdt = dHdt[time_step_num]

    # Central difference in space
    gradH = [
        (torch.roll(H[time_step_num], -1, i) - torch.roll(H[time_step_num], 1, i))
        / (2 * dx[i])
        for i in range(spatial_dim)
    ]

    all(torch.allclose(grad_component, torch.zeros(N1, N2)) for grad_component in gradH)

    v_new, z_new = dynamic_wfr_relaxed._project_affine(
        p, v, z, H[time_step_num], dHdt, gradH, Fprime, dx
    )

    torch.testing.assert_close(v_new, v)
    torch.testing.assert_close((z_new * p).sum(), torch.tensor(0.0))


def test_project_affine_hi_dim_allzero():
    """Pass the zero input for all of the variables to check if it gives us any error."""

    time_step_num = 5
    T = 10
    n = 3
    k = 10
    grid_shape = (10, 61, 21)
    p = torch.zeros(grid_shape)
    v = torch.zeros(grid_shape + (n,))
    z = torch.zeros(grid_shape)
    H = torch.zeros((k,) + (T + 1,) + grid_shape)
    F = torch.zeros((k,) + (T + 1,))
    dx = [1 / s for s in grid_shape]
    spatial_dim = len(dx)
    # Forward difference in time
    Fprime = T * (torch.roll(F, -1) - F)
    Fprime = Fprime[:, time_step_num]

    # Forward difference in time
    dHdt = T * (torch.roll(H, -1, 0) - H)
    dHdt = dHdt[:, time_step_num]

    # Central difference in space
    gradH = [
        [
            (
                torch.roll(H[constraint, time_step_num], -1, i)
                - torch.roll(H[constraint, time_step_num], 1, i)
            )
            / (2 * dx[i])
            for i in range(spatial_dim)
        ]
        for constraint in range(k)
    ]

    # Check if it outputs ValuEerror

    try:
        v_new, z_new = dynamic_wfr_relaxed._project_affine_hi_dim(
            p, v, z, H[:, time_step_num], dHdt, gradH, Fprime, dx
        )
    except ValueError:
        pass
    else:
        raise AssertionError("_project_affine_hi_dim did not raise ValueError")


"""
Test of _project_affine_hi_dim
"""


def test_project_affine_hi_dim_split_one():
    """Pass v and z with all ones to enforce the constraint
    (1,1,1,0,0,0)*z = 0
    (0,0,0,1,1,1)*z = 0
    so that z will be zero. As a side effect v will also change but we don't check it \
    here.
    """
    k = 2
    time_step_num = 2
    T = 5
    N = 6
    grid_shape = (N,)
    n = len(grid_shape)
    p = torch.tensor([1, 1, 1, 1, 1, 1])
    v = torch.ones(grid_shape + (n,))
    z = torch.ones(grid_shape)
    H_template_1 = [1, 1, 1, 0, 0, 0]
    H_template_2 = [0, 0, 0, 1, 1, 1]
    H = torch.stack([torch.tensor(H_template_1), torch.tensor(H_template_2)], dim=0)
    H = H.unsqueeze(1).expand(k, T + 1, N)
    F = torch.zeros((k,) + (T + 1,))
    dx = [1 / s for s in grid_shape]
    spatial_dim = len(dx)
    # Forward difference in time
    Fprime = T * (torch.roll(F, -1, 1) - F)
    Fprime = Fprime[:, time_step_num]

    # Forward difference in time
    dHdt = T * (torch.roll(H, -1, 1) - H)
    dHdt = dHdt[:, time_step_num, ...]

    # Central difference in space
    gradH = [
        [
            (
                torch.roll(H[constraint, time_step_num], -1, i)
                - torch.roll(H[constraint, time_step_num], 1, i)
            )
            / (2 * dx[i])
            for i in range(spatial_dim)
        ]
        for constraint in range(k)
    ]

    v_new, z_new = dynamic_wfr_relaxed._project_affine_hi_dim(
        p, v, z, H[:, time_step_num], dHdt, gradH, Fprime, dx
    )

    # torch.testing.assert_close(v_new, torch.zeros(grid_shape + (n,)))
    torch.testing.assert_close(z_new, torch.zeros(grid_shape))


def test_project_affine_hi_dim_constantmass():
    """Pass a random v, z and enforce the constraint rho_t(omega) = 1, that is, H=1, F=1.
    We expect the inner product between p and z to be zero (= net mass change is zero).
    """
    k = 1
    time_step_num = 7
    T = 100
    N1 = 31
    N2 = 57
    grid_shape = (N1, N2)
    n = len(grid_shape)
    p = torch.abs(torch.randn(grid_shape))
    v = torch.randn(grid_shape + (n,))
    z = torch.randn(grid_shape)
    H = torch.ones((k,) + (T + 1,) + grid_shape)
    F = torch.ones((k,) + (T + 1,))
    dx = [1 / s for s in grid_shape]
    spatial_dim = len(dx)
    # Forward difference in time
    Fprime = T * (torch.roll(F, -1, 1) - F)
    Fprime = Fprime[:, time_step_num]

    # Forward difference in time
    dHdt = T * (torch.roll(H, -1, 1) - H)
    dHdt = dHdt[:, time_step_num]

    # Central difference in space
    gradH = [
        [
            (
                torch.roll(H[constraint, time_step_num], -1, i)
                - torch.roll(H[constraint, time_step_num], 1, i)
            )
            / (2 * dx[i])
            for i in range(spatial_dim)
        ]
        for constraint in range(k)
    ]

    v_new, z_new = dynamic_wfr_relaxed._project_affine_hi_dim(
        p, v, z, H[:, time_step_num], dHdt, gradH, Fprime, dx
    )

    torch.testing.assert_close(v_new, v)
    torch.testing.assert_close(
        (z_new * p).sum(), torch.tensor(0.0), atol=1e-4, rtol=1e-4
    )


"""
Test of _div_plus_pz_grid
"""


def test_div_plus_pz_grid_linear():
    # Tries to calculate the diveregnce of f(x)=x which should be 1
    t = torch.tensor(0)
    N1 = 10
    p = -torch.ones(N1)
    v = torch.arange(0, 1.0, 1.0 / 10).reshape(1, -1, 1)
    dx = [1.0 / 10]
    z = torch.zeros((1, 10))
    T = 1

    div = dynamic_wfr_relaxed._div_plus_pz_grid(t, p, v, z, dx, T)
    # Excluding the endpoint because of periodicity
    torch.testing.assert_close(div[1:-1], torch.ones(N1)[1:-1])


def test_div_plus_pz_grid_sin():
    # Tries to calculate the divergence of f(x) = sin(x)
    # This should be cos(x)
    t = torch.tensor(0)
    N1 = 20
    p = -torch.ones(N1)
    xs = torch.arange(0, 2 * math.pi, 2 * math.pi / N1)
    v = torch.sin(xs).reshape(1, -1, 1)
    dx = [2 * math.pi / N1]
    z = torch.zeros((1, N1))
    T = 1

    div = dynamic_wfr_relaxed._div_plus_pz_grid(t, p, v, z, dx, T)
    torch.testing.assert_close(div, torch.cos(xs), atol=(2 * math.pi / N1) ** 2, rtol=0)


def test_div_plus_pz_grid_sintime():
    # Tries to calculate the divergence of f(x) = sin(x-t) for random t
    # expected: cos(x-t)
    T = 10
    t = torch.tensor(random.random())
    N1 = 20
    p = -torch.ones(N1)
    xs = torch.arange(0, 2 * math.pi, 2 * math.pi / N1).reshape(1, -1, 1)
    ts = torch.arange(0, T - 1).reshape(-1, 1, 1) / T
    xsts = xs - ts
    v = torch.sin(xsts)
    dx = [2 * math.pi / N1]
    z = torch.zeros((T, N1))

    div = dynamic_wfr_relaxed._div_plus_pz_grid(t, p, v, z, dx, T)
    torch.testing.assert_close(
        div, torch.cos(xs.squeeze() - t), atol=(2 * math.pi / N1) ** 2 + 1.0 / T, rtol=0
    )


def test_div_plus_pz_grid_linear_upwind1():
    # Tries to calculate the diveregnce of f(x)=x which should be 1
    t = torch.tensor(0)
    N1 = 10
    p = -torch.ones(N1)
    v = torch.arange(0, 1.0, 1.0 / 10).reshape(1, -1, 1)
    dx = [1.0 / 10]
    z = torch.zeros((1, 10))
    T = 1

    div = dynamic_wfr_relaxed._div_plus_pz_grid(t, p, v, z, dx, T, scheme="upwind1")
    # Excluding the endpoint because of periodicity
    torch.testing.assert_close(div[1:-1], torch.ones(N1)[1:-1])


def test_div_plus_pz_grid_sin_upwind1():
    # Tries to calculate the divergence of f(x) = sin(x)
    # This should be cos(x)
    t = torch.tensor(0)
    N1 = 20
    p = -torch.ones(N1)
    xs = torch.arange(0, 2 * math.pi, 2 * math.pi / N1)
    v = torch.sin(xs).reshape(1, -1, 1)
    dx = [2 * math.pi / N1]
    z = torch.zeros((1, N1))
    T = 1

    div = dynamic_wfr_relaxed._div_plus_pz_grid(t, p, v, z, dx, T, scheme="upwind1")
    torch.testing.assert_close(div, torch.cos(xs), atol=2 * math.pi / N1, rtol=0)


def test_div_plus_pz_grid_sintime_upwind1():
    # Tries to calculate the divergence of f(x) = sin(x-t) for random t
    # expected: cos(x-t)
    T = 10
    t = torch.tensor(random.random())
    N1 = 20
    p = -torch.ones(N1)
    xs = torch.arange(0, 2 * math.pi, 2 * math.pi / N1).reshape(1, -1, 1)
    ts = torch.arange(0, (T - 1) / T, 1.0 / T).reshape(-1, 1, 1)
    xsts = xs - ts
    v = torch.sin(xsts)
    dx = [2 * math.pi / N1]
    z = torch.zeros((T, N1))

    div = dynamic_wfr_relaxed._div_plus_pz_grid(t, p, v, z, dx, T, scheme="upwind1")
    torch.testing.assert_close(
        div, torch.cos(xs.squeeze() - t), atol=2 * math.pi / N1 + 1.0 / T, rtol=0
    )


def test_div_plus_pz_grid_linear_lax_wendroff():
    # Tries to calculate the diveregnce of f(x)=x which should be 1
    t = torch.tensor(0)
    N1 = 10
    p = -torch.ones(N1)
    v = torch.arange(0, 1.0, 1.0 / 10).reshape(1, -1, 1)
    dx = [1.0 / 10]
    z = torch.zeros((1, 10))
    T = 1

    div = dynamic_wfr_relaxed._div_plus_pz_grid(
        t, p, v, z, dx, T, scheme="lax-wendroff"
    )
    # Excluding the endpoint because of periodicity
    torch.testing.assert_close(div[1:-1], torch.ones(N1)[1:-1])


"""
Test of wfr_grid
"""


def test_wfr_grid_samedist():
    # p1=p2=uniform. expected = no error, wfr=0, p=1, v=z=0.
    p1 = torch.ones(10, 10)
    p2 = p1
    delta = 1
    rel = 1
    T = 100
    lr = 1
    wfr, p, v, z = dynamic_wfr_relaxed.wfr_grid(p1, p2, delta, rel=rel, T=T, lr=lr)
    p_objective = torch.ones((T + 1, 10, 10))
    v_objective = torch.zeros((T, 10, 10, 2))
    z_objective = torch.zeros((T, 10, 10))

    torch.testing.assert_close(wfr, torch.tensor(0.0))
    torch.testing.assert_close(p, p_objective)
    torch.testing.assert_close(v, v_objective)
    torch.testing.assert_close(z, z_objective)


"""
Test of wfr_grid_scipy
"""


def test_wfr_grid_scipy_samedist():
    # p1=p2=uniform. expected = no error, wfr=0, p=1, v=z=0.
    p1 = np.ones((10, 10))
    p2 = p1
    delta = 1
    rel = 1
    T = 100
    wfr, p, v, z = dynamic_wfr_relaxed.wfr_grid_scipy(p1, p2, delta, rel=rel, T=T)
    p_objective = np.ones((T + 1, 10, 10))
    v_objective = np.zeros((T, 10, 10, 2))
    z_objective = np.zeros((T, 10, 10))

    np.testing.assert_allclose(wfr, np.array(0.0))
    np.testing.assert_allclose(p, p_objective)
    np.testing.assert_allclose(v, v_objective)
    np.testing.assert_allclose(z, z_objective)


def test_wfr_grid_scipy_samedist_constrained():
    # p1=p2=uniform. expected = no error, wfr=0, p=1, v=z=0.
    p1 = np.ones((10, 10))
    p2 = p1
    delta = 1
    rel = 1
    T = 131
    H = np.zeros((T + 1, 10, 10))
    F = np.zeros((T + 1,))

    wfr, p, v, z = dynamic_wfr_relaxed.wfr_grid_scipy(
        p1, p2, delta, rel=rel, T=T, H=H, F=F
    )
    p_objective = np.ones((T + 1, 10, 10))
    v_objective = np.zeros((T, 10, 10, 2))
    z_objective = np.zeros((T, 10, 10))

    np.testing.assert_allclose(wfr, np.array(0.0))
    np.testing.assert_allclose(p, p_objective)
    np.testing.assert_allclose(v, v_objective)
    np.testing.assert_allclose(z, z_objective)


def test_wfr_grid_scipy_samedist_constrained_hidim():
    # p1=p2=uniform. expected = no error, wfr=0, p=1, v=z=0.
    # To test _projec_affine_hi_dim, we need to pass H and F with an additional dimension
    p1 = np.ones((10, 10))
    p2 = p1
    delta = 1
    rel = 1
    T = 131
    H = np.ones((1, T + 1, 10, 10))
    F = np.ones(
        (
            1,
            T + 1,
        )
    )

    wfr, p, v, z = dynamic_wfr_relaxed.wfr_grid_scipy(
        p1, p2, delta, rel=rel, T=T, H=H, F=F
    )
    p_objective = np.ones((T + 1, 10, 10))
    v_objective = np.zeros((T, 10, 10, 2))
    z_objective = np.zeros((T, 10, 10))

    np.testing.assert_allclose(wfr, np.array(0.0))
    np.testing.assert_allclose(p, p_objective)
    np.testing.assert_allclose(v, v_objective)
    np.testing.assert_allclose(z, z_objective)


def test_wfr_grid_scipy_wrong_initial_cond():
    # p1=p2=uniform, v=z=wrong size, expected = TypeError
    p1 = np.ones((10, 10))
    p2 = p1
    delta = 1
    rel = 1
    T = 100
    v0 = np.zeros((T, 10))
    z0 = np.zeros((T, 10, 10))

    with pytest.raises(TypeError):
        dynamic_wfr_relaxed.wfr_grid_scipy(p1, p2, delta, rel=rel, T=T, v0=v0, z0=z0)
