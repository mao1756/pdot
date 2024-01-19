import torch
import pytest
import dynamic_wfr_relaxed
import math
import random

"""
Test of _WFR_energy
"""


def test_WFR_energy_1Dinput():
    p = torch.tensor([3.]).reshape(1, 1)
    v = torch.tensor([4.]).reshape(1, 1, 1)
    z = torch.tensor([-1]).reshape(1, 1)
    energy = dynamic_wfr_relaxed._WFR_energy(
        p,
        v,
        z,
        delta=1
    )
    torch.testing.assert_close(energy, torch.tensor(51.))


def test_WFR_energy_zeroinput():
    p = torch.tensor([0]).reshape(1, 1)
    v = torch.tensor([4.]).reshape(1, 1, 1)
    z = torch.tensor([-1]).reshape(1, 1)
    energy = dynamic_wfr_relaxed._WFR_energy(
        p,
        v,
        z,
        delta=1
    )
    torch.testing.assert_close(energy, torch.tensor(0.))


def test_WFR_energy_1dlonginput():
    p = torch.tensor([[0.1, 3., 0.5], [0.2, 0.3, 0.2], [7., 3., 4.]])
    v = torch.tensor([[1., 1., 1.], [2., 3., 4.], [0., 0., 1]]).reshape(3, 3, 1)
    z = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.2, 0.3]])
    energy = dynamic_wfr_relaxed._WFR_energy(
        p,
        v,
        z,
        delta=1
    )
    torch.testing.assert_close(energy, torch.tensor(14.893))


def test_WFR_energy_wrong_size():
    p = torch.tensor([[0.1, 3., 0.5], [0.2, 0.3, 0.2], [7., 3., 4.]])
    v = torch.tensor([[[1., 1], [1., 1], [1., 1]], [[2., 1], [3., 3], [4., 3]],
                     [[0., 2], [0., 1], [3., 4.]]]).reshape(3, 3, 2)
    z = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.2, 0.3]])
    with pytest.raises(TypeError):
        dynamic_wfr_relaxed._WFR_energy(
            p,
            v,
            z,
            delta=1
        )


def test_WFR_energy_incorrect_v():
    p = torch.tensor([[0.1, 3., 0.5], [0.2, 0.3, 0.2], [7., 3., 4.]])
    v = torch.tensor([4.]).reshape(1, 1, 1)
    z = torch.tensor([-1]).reshape(1, 1)
    with pytest.raises(TypeError):
        dynamic_wfr_relaxed._WFR_energy(
            p,
            v,
            z,
            delta=1
        )


"""
Test of _div_plus_pz_grid
"""


def test_div_plus_pz_grid_linear():
    # Tries to calculate the diveregnce of f(x)=x which should be 1
    t = 0
    N1 = 10
    p = -torch.ones(N1)
    v = torch.arange(0, 1., 1./10).reshape(1, -1, 1)
    dx = [1./10]
    z = torch.zeros((1, 10))
    T = 1

    div = dynamic_wfr_relaxed._div_plus_pz_grid(t, p, v, z, dx, T)
    # Excluding the endpoint because of periodicity
    torch.testing.assert_close(div[1:-1], torch.ones(N1)[1:-1])


def test_div_plus_pz_grid_sin():
    # Tries to calculate the divergence of f(x) = sin(x)
    # This should be cos(x)
    t = 0
    N1 = 20
    p = -torch.ones(N1)
    xs = torch.arange(0, 2*math.pi, 2*math.pi/N1)
    v = torch.sin(xs).reshape(1, -1, 1)
    dx = [2*math.pi/N1]
    z = torch.zeros((1, N1))
    T = 1

    div = dynamic_wfr_relaxed._div_plus_pz_grid(t, p, v, z, dx, T)
    torch.testing.assert_close(div, torch.cos(xs), atol=(2*math.pi/N1)**2, rtol=0)


def test_div_plus_pz_grid_sintime():
    # Tries to calculate the divergence of f(x) = sin(x-t) for random t
    # expected: cos(x-t)
    T = 10
    t = random.random()
    N1 = 20
    p = -torch.ones(N1)
    xs = torch.arange(0, 2*math.pi, 2*math.pi/N1).reshape(1, -1, 1)
    ts = torch.arange(0, (T-1)/T, 1./T).reshape(-1, 1, 1)
    xsts = xs-ts
    v = torch.sin(xsts)
    dx = [2*math.pi/N1]
    z = torch.zeros((T, N1))

    div = dynamic_wfr_relaxed._div_plus_pz_grid(t, p, v, z, dx, T)
    torch.testing.assert_close(div, torch.cos(xs.squeeze()-t), atol=(2*math.pi/N1)**2
                               + 1./T, rtol=0)


"""
Test of dynamic_wfr_relaxed_grid
"""


def test_dynamic_wfr_relaxed_grid_samedist():
    # p1=p2=uniform. expected = no error, wfr=0, p=1, v=z=0.
    p1 = torch.ones(10, 10)
    p2 = p1
    delta = 1
    rel = 1
    T = 100
    lr = 1
    wfr, p, v, z = dynamic_wfr_relaxed.dynamic_wfr_relaxed_grid(
        p1,
        p2,
        delta,
        rel=rel,
        T=T,
        lr=lr
    )
    p_objective = torch.ones((T+1, 10, 10))
    v_objective = torch.zeros((T, 10, 10, 2))
    z_objective = torch.zeros((T, 10, 10))

    torch.testing.assert_close(wfr, torch.tensor(0.))
    torch.testing.assert_close(p, p_objective)
    torch.testing.assert_close(v, v_objective)
    torch.testing.assert_close(z, z_objective)
