import proximal.grids as g
import proximal.backend_extension as pb
import numpy as np
import random
import torch
import math


class TestVar:
    def test_proj_positive_numpy(self):
        N = random.randint(2, 4)
        cs = tuple([random.randint(1, 20) for _ in range(N)])
        ll = tuple([random.random() for _ in range(N)])
        D = [np.random.randn(*cs) for _ in range(N)]
        Z = np.random.randn(*cs)
        x = g.Var(cs, ll, D, Z)
        x.proj_positive()
        assert np.all(x.D[0] >= 0)
        assert isinstance(x.nx, pb.NumpyBackend_ext)

    def test_proj_positive_torch(self):
        N = random.randint(2, 4)
        cs = tuple([random.randint(1, 20) for _ in range(N)])
        ll = tuple([random.random() for _ in range(N)])
        D = [torch.randn(*cs) for _ in range(N)]
        Z = torch.randn(*cs)
        x = g.Var(cs, ll, D, Z)
        x.proj_positive()
        assert torch.all(x.D[0] >= 0)
        assert isinstance(x.nx, pb.TorchBackend_ext)

    def test_dilate_grid_numpy(self):
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        D1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = np.array([[9.0, 10.0], [11.0, 12.0]])
        x = g.Var(cs, ll, D, Z)
        x.dilate_grid(2.0)
        assert np.allclose(x.D[0], np.array([[0.5, 1], [1.5, 2]]))
        assert np.allclose(x.D[1], D1)
        assert np.allclose(x.Z, np.array([[4.5, 5], [5.5, 6]]))
        assert np.allclose(x.ll, (1.0, 2.0))

    def test_dilate_grid_torch(self):
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        D1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
        x = g.Var(cs, ll, D, Z)
        x.dilate_grid(2.0)
        assert torch.allclose(x.D[0], torch.tensor([[0.5, 1], [1.5, 2]]))
        assert torch.allclose(x.D[1], D1)
        assert torch.allclose(x.Z, torch.tensor([[4.5, 5], [5.5, 6]]))
        assert torch.allclose(torch.tensor(x.ll), torch.tensor([1.0, 2.0]))

    def test_add_numpy(self):
        N = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        D1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = np.array([[9.0, 10.0], [11.0, 12.0]])
        x = g.Var(cs, ll, D, Z)
        y = g.Var(cs, ll, D, Z)
        z = x + y

        assert z.N == N
        assert z.cs == cs
        assert z.ll == ll
        assert np.allclose(z.D[0], 2 * D0)
        assert np.allclose(z.D[1], 2 * D1)
        assert np.allclose(z.Z, 2 * Z)

    def test_add_torch(self):
        N = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        D1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
        x = g.Var(cs, ll, D, Z)
        y = g.Var(cs, ll, D, Z)
        z = x + y

        assert z.N == N
        assert z.cs == cs
        assert z.ll == ll
        assert torch.allclose(z.D[0], 2 * D0)
        assert torch.allclose(z.D[1], 2 * D1)
        assert torch.allclose(z.Z, 2 * Z)

    def test_iadd_numpy(self):
        N = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        D1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = np.array([[9.0, 10.0], [11.0, 12.0]])
        x = g.Var(cs, ll, D, Z)
        y = g.Var(cs, ll, D, Z)
        x += y

        assert x.N == N
        assert x.cs == cs
        assert x.ll == ll
        assert np.allclose(x.D[0], 2 * D0)
        assert np.allclose(x.D[1], 2 * D1)
        assert np.allclose(x.Z, 2 * Z)

    def test_iadd_torch(self):
        N = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        D1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
        x = g.Var(cs, ll, D, Z)
        y = g.Var(cs, ll, D, Z)
        x += y

        assert x.N == N
        assert x.cs == cs
        assert x.ll == ll
        assert torch.allclose(x.D[0], 2 * D0)
        assert torch.allclose(x.D[1], 2 * D1)
        assert torch.allclose(x.Z, 2 * Z)

    def test_sub_numpy(self):
        N = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        D1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = np.array([[9.0, 10.0], [11.0, 12.0]])
        x = g.Var(cs, ll, D, Z)
        y = g.Var(cs, ll, D, Z)
        z = x - y

        assert z.N == N
        assert z.cs == cs
        assert z.ll == ll
        assert np.allclose(z.D[0], np.zeros(cs))
        assert np.allclose(z.D[1], np.zeros(cs))
        assert np.allclose(z.Z, np.zeros(cs))

    def test_sub_torch(self):
        N = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        D1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
        x = g.Var(cs, ll, D, Z)
        y = g.Var(cs, ll, D, Z)
        z = x - y

        assert z.N == N
        assert z.cs == cs
        assert z.ll == ll
        assert torch.allclose(z.D[0], torch.zeros(cs))
        assert torch.allclose(z.D[1], torch.zeros(cs))
        assert torch.allclose(z.Z, torch.zeros(cs))


class TestCvar:
    def test_energy_WFR_numpy(self):
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        D1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = np.array([[9.0, 10.0], [11.0, 12.0]])
        x = g.Cvar(cs, ll, D, Z)
        expected_energy = (0.5 * D1**2 / D0 + 0.5 * Z**2 / D0).sum() / np.prod(cs)
        assert np.allclose(x.energy(1.0, 2.0, 2.0), expected_energy)

    def test_energy_WFR_torch(self):
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        D1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
        x = g.Cvar(cs, ll, D, Z)
        expected_energy = (0.5 * D1**2 / D0 + 0.5 * Z**2 / D0).sum() / torch.prod(
            torch.tensor(cs)
        )
        assert torch.allclose(x.energy(1.0, 2.0, 2.0), expected_energy)

    def test_add(self):  # test for Cvar
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        D1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = np.array([[9.0, 10.0], [11.0, 12.0]])
        x = g.Cvar(cs, ll, D, Z)
        y = g.Cvar(cs, ll, D, Z)
        z = x + y

        assert z.cs == cs
        assert z.ll == ll
        assert isinstance(z, g.Cvar)
        assert np.allclose(z.D[0], 2 * D0)
        assert np.allclose(z.D[1], 2 * D1)
        assert np.allclose(z.Z, 2 * Z)

    def test_iadd(self):  # test for Cvar
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        D1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = np.array([[9.0, 10.0], [11.0, 12.0]])
        x = g.Cvar(cs, ll, D, Z)
        y = g.Cvar(cs, ll, D, Z)
        x += y

        assert x.cs == cs
        assert x.ll == ll
        assert isinstance(x, g.Cvar)
        assert np.allclose(x.D[0], 2 * D0)
        assert np.allclose(x.D[1], 2 * D1)
        assert np.allclose(x.Z, 2 * Z)

    def test_sub(self):  # test for Cvar
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        D1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = np.array([[9.0, 10.0], [11.0, 12.0]])
        x = g.Cvar(cs, ll, D, Z)
        y = g.Cvar(cs, ll, D, Z)
        z = x - y

        assert z.cs == cs
        assert z.ll == ll
        assert isinstance(z, g.Cvar)
        assert np.allclose(z.D[0], np.zeros(cs))
        assert np.allclose(z.D[1], np.zeros(cs))
        assert np.allclose(z.Z, np.zeros(cs))

    def test_mul(self):
        cs = (2, 2)
        ll = (1.0, 1.0)
        D0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        D1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        D = [D0, D1]
        Z = np.array([[9.0, 10.0], [11.0, 12.0]])
        x = g.Cvar(cs, ll, D, Z)
        z = x * 2

        assert z.cs == cs
        assert z.ll == ll
        assert isinstance(z, g.Cvar)
        assert np.allclose(z.D[0], 2 * D0)
        assert np.allclose(z.D[1], 2 * D1)
        assert np.allclose(z.Z, 2 * Z)


class TestSvar:
    def test_projBC_2D_numpy(self):
        cs = (3, 2)  # a centered grid of 3 points in time, 2 points in space
        ll = (1.0, 1.0)
        rho0 = np.array([1.0, 2.0])
        rho1 = np.array([5.0, 6.0])
        N = len(rho0.shape) + 1
        shapes_staggered = g.get_staggered_shape(cs)
        D = [np.zeros(shapes_staggered[k]) for k in range(N)]
        D[0] = g.linear_interpolation(rho0, rho1, cs[0])
        Z = np.zeros(cs)
        x = g.Svar(cs, ll, D, Z)

        assert x.D[0].shape == (4, 2)
        assert x.D[1].shape == (3, 3)
        x.proj_BC(rho0, rho1)
        assert np.allclose(x.D[0][0], rho0)
        assert np.allclose(x.D[0][1], (2.0 / 3) * rho0 + (1.0 / 3) * rho1)
        assert np.allclose(x.D[0][-1], rho1)
        assert np.allclose(x.D[1][:, 0], np.zeros(3))
        assert np.allclose(x.D[1][:, -1], np.zeros(3))
        assert np.allclose(x.Z[0], np.zeros((3, 2)))

    def test_projBC_2D_torch(self):
        cs = (3, 2)  # a centered grid of 3 points in time, 2 points in space
        ll = (1.0, 1.0)
        rho0 = torch.tensor([1.0, 2.0])
        rho1 = torch.tensor([5.0, 6.0])
        N = len(rho0.shape) + 1
        shapes_staggered = g.get_staggered_shape(cs)
        D = [torch.zeros(shapes_staggered[k]) for k in range(N)]
        D[0] = g.linear_interpolation(rho0, rho1, cs[0])
        Z = torch.zeros(cs)
        x = g.Svar(cs, ll, D, Z)

        assert x.D[0].shape == (4, 2)
        assert x.D[1].shape == (3, 3)
        x.proj_BC(rho0, rho1)
        assert torch.allclose(x.D[0][0], rho0)
        assert torch.allclose(x.D[0][1], (2.0 / 3) * rho0 + (1.0 / 3) * rho1)
        assert torch.allclose(x.D[0][-1], rho1)
        assert torch.allclose(x.D[1][:, 0], torch.zeros(3))
        assert torch.allclose(x.D[1][:, -1], torch.zeros(3))
        assert torch.allclose(x.Z[0], torch.zeros((3, 2)))

    def test_remainder_CE_2D_zero_numpy(self):
        # D[1] = 0, Z = 0 or no change from initial values
        cs = (3, 2)  # a centered grid of 3 points in time, 2 points in space
        ll = (1.0, 1.0)
        rho0 = torch.tensor([1.0, 2.0])
        rho1 = torch.tensor([5.0, 6.0])
        N = len(rho0.shape) + 1
        shapes_staggered = g.get_staggered_shape(cs)
        D = [torch.zeros(shapes_staggered[k]) for k in range(N)]
        D[0] = g.linear_interpolation(rho0, rho1, cs[0])
        Z = torch.zeros(cs)
        x = g.Svar(cs, ll, D, Z)
        np.allclose(
            x.remainder_CE(),
            np.array(
                [
                    [(7.0 - 3.0) / 3.0, (10.0 - 6.0) / 3],
                    [(11.0 - 7.0) / 3.0, (16.0 - 10.0) / 3.0],
                    [(15.0 - 11.0) / 3.0, (18.0 - 14.0) / 3.0],
                ]
            )
            * 3.0,
        )

    def test_remainder_CE_2D_zero_torch(self):
        # D[1] = 0, Z = 0 or no change from initial values
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = torch.tensor([1.0, 2.0])
        rho1 = torch.tensor([5.0, 6.0])
        N = len(rho0.shape) + 1
        shapes_staggered = g.get_staggered_shape(cs)
        D = [torch.zeros(shapes_staggered[k]) for k in range(N)]
        D[0] = g.linear_interpolation(rho0, rho1, cs[0])
        Z = torch.zeros(cs)
        x = g.Svar(cs, ll, D, Z)
        torch.allclose(
            x.remainder_CE(),
            torch.tensor(
                [
                    [(7.0 - 3.0) / 3.0, (10.0 - 6.0) / 3],
                    [(11.0 - 7.0) / 3.0, (16.0 - 10.0) / 3.0],
                    [(15.0 - 11.0) / 3.0, (18.0 - 14.0) / 3.0],
                ]
            )
            * 3.0,
        )


class TestCSvar:
    def test_interp_2D_numpy(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = np.array([1.0, 2.0])
        rho1 = np.array([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        D1 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        x.U.D[1] = D1
        x.interp_()
        D0_target = np.array(
            [[5.0 / 3, 8.0 / 3.0], [3.0, 4.0], [13.0 / 3.0, 16.0 / 3.0]]
        )
        D1_target = np.array(
            [
                [9.0 / 2.0, 11.0 / 2.0],
                [15.0 / 2.0, 17.0 / 2.0],
                [21.0 / 2.0, 23.0 / 2.0],
            ]
        )
        assert np.allclose(x.V.D[0], D0_target)
        assert np.allclose(x.V.D[1], D1_target)

    def test_interp_2D_torch(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = torch.tensor([1.0, 2.0])
        rho1 = torch.tensor([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        D1 = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        x.U.D[1] = D1
        x.interp_()
        D0_target = torch.tensor(
            [[5.0 / 3, 8.0 / 3.0], [3.0, 4.0], [13.0 / 3.0, 16.0 / 3.0]]
        )
        D1_target = torch.tensor(
            [
                [9.0 / 2.0, 11.0 / 2.0],
                [15.0 / 2.0, 17.0 / 2.0],
                [21.0 / 2.0, 23.0 / 2.0],
            ]
        )
        assert torch.allclose(x.V.D[0], D0_target)
        assert torch.allclose(x.V.D[1], D1_target)

    def test_dist_from_interp_numpy(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = np.array([1.0, 2.0])
        rho1 = np.array([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        D1 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        x.U.D[1] = D1
        D1_interpolated = np.array(
            [
                [9.0 / 2.0, 11.0 / 2.0],
                [15.0 / 2.0, 17.0 / 2.0],
                [21.0 / 2.0, 23.0 / 2.0],
            ]
        )
        dist = (D1_interpolated**2).sum() * np.prod(ll) / np.prod(cs)
        assert np.allclose(x.dist_from_interp(), dist)

    def test_dist_from_interp_torch(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = torch.tensor([1.0, 2.0])
        rho1 = torch.tensor([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        D1 = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        x.U.D[1] = D1
        D1_interpolated = torch.tensor(
            [
                [9.0 / 2.0, 11.0 / 2.0],
                [15.0 / 2.0, 17.0 / 2.0],
                [21.0 / 2.0, 23.0 / 2.0],
            ]
        )
        dist = (D1_interpolated**2).sum() * math.prod(ll) / math.prod(cs)
        assert torch.allclose(x.dist_from_interp(), dist)

    def test_add_numpy(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = np.array([1.0, 2.0])
        rho1 = np.array([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        x_org = g.CSvar(rho0, rho1, cs[0], ll)
        z = x + x
        assert np.allclose(z.U.D[0], 2 * x_org.U.D[0])
        assert np.allclose(z.U.D[1], 2 * x_org.U.D[1])
        assert np.allclose(z.V.D[0], 2 * x_org.V.D[0])
        assert np.allclose(z.V.D[1], 2 * x_org.V.D[1])

    def test_add_torch(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = torch.tensor([1.0, 2.0])
        rho1 = torch.tensor([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        x_org = g.CSvar(rho0, rho1, cs[0], ll)
        z = x + x
        assert torch.allclose(z.U.D[0], 2 * x_org.U.D[0])
        assert torch.allclose(z.U.D[1], 2 * x_org.U.D[1])
        assert torch.allclose(z.V.D[0], 2 * x_org.V.D[0])
        assert torch.allclose(z.V.D[1], 2 * x_org.V.D[1])

    def test_iadd_numpy(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = np.array([1.0, 2.0])
        rho1 = np.array([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        x_org = g.CSvar(rho0, rho1, cs[0], ll)
        x += x
        assert np.allclose(x.U.D[0], 2 * x_org.U.D[0])
        assert np.allclose(x.U.D[1], 2 * x_org.U.D[1])
        assert np.allclose(x.V.D[0], 2 * x_org.V.D[0])
        assert np.allclose(x.V.D[1], 2 * x_org.V.D[1])

    def test_iadd_torch(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = torch.tensor([1.0, 2.0])
        rho1 = torch.tensor([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        x_org = g.CSvar(rho0, rho1, cs[0], ll)
        x += x
        assert torch.allclose(x.U.D[0], 2 * x_org.U.D[0])
        assert torch.allclose(x.U.D[1], 2 * x_org.U.D[1])
        assert torch.allclose(x.V.D[0], 2 * x_org.V.D[0])
        assert torch.allclose(x.V.D[1], 2 * x_org.V.D[1])

    def test_sub_numpy(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = np.array([1.0, 2.0])
        rho1 = np.array([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        z = x - x
        assert np.allclose(z.U.D[0], np.zeros((4, 2)))
        assert np.allclose(z.U.D[1], np.zeros((3, 3)))
        assert np.allclose(z.V.D[0], np.zeros(cs))
        assert np.allclose(z.V.D[1], np.zeros(cs))

    def test_sub_torch(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = torch.tensor([1.0, 2.0])
        rho1 = torch.tensor([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        z = x - x
        assert torch.allclose(z.U.D[0], torch.zeros((4, 2)))
        assert torch.allclose(z.U.D[1], torch.zeros((3, 3)))
        assert torch.allclose(z.V.D[0], torch.zeros(cs))
        assert torch.allclose(z.V.D[1], torch.zeros(cs))

    def test_mul_numpy(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = np.array([1.0, 2.0])
        rho1 = np.array([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        z = x * 2
        assert np.allclose(z.U.D[0], 2 * x.U.D[0])
        assert np.allclose(z.U.D[1], 2 * x.U.D[1])
        assert np.allclose(z.V.D[0], 2 * x.V.D[0])
        assert np.allclose(z.V.D[1], 2 * x.V.D[1])

    def test_mul_torch(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = torch.tensor([1.0, 2.0])
        rho1 = torch.tensor([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        z = x * 2
        assert torch.allclose(z.U.D[0], 2 * x.U.D[0])
        assert torch.allclose(z.U.D[1], 2 * x.U.D[1])
        assert torch.allclose(z.V.D[0], 2 * x.V.D[0])
        assert torch.allclose(z.V.D[1], 2 * x.V.D[1])

    def test_rmul_numpy(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = np.array([1.0, 2.0])
        rho1 = np.array([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        z = 2 * x
        assert np.allclose(z.U.D[0], 2 * x.U.D[0])
        assert np.allclose(z.U.D[1], 2 * x.U.D[1])
        assert np.allclose(z.V.D[0], 2 * x.V.D[0])
        assert np.allclose(z.V.D[1], 2 * x.V.D[1])

    def test_rmul_torch(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        rho0 = torch.tensor([1.0, 2.0])
        rho1 = torch.tensor([5.0, 6.0])
        x = g.CSvar(rho0, rho1, cs[0], ll)
        z = 2 * x
        assert torch.allclose(z.U.D[0], 2 * x.U.D[0])
        assert torch.allclose(z.U.D[1], 2 * x.U.D[1])
        assert torch.allclose(z.V.D[0], 2 * x.V.D[0])
        assert torch.allclose(z.V.D[1], 2 * x.V.D[1])


class TestFunctions:
    def test_interpT_2D_numpy(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        D1 = np.array([[4.0, 6.0], [7.0, 9.0], [10.0, 12.0]])
        x = g.Cvar(cs, ll, [D0, D1], np.ones(cs))

        assert g.interpT(x).D[0].shape == (4, 2)
        assert g.interpT(x).D[1].shape == (3, 3)
        assert g.interpT(x).Z.shape == cs
        assert np.allclose(
            g.interpT(x).D[0],
            np.array(
                [[1.0 / 2.0, 2.0 / 2.0], [2.0, 3.0], [4.0, 5.0], [5.0 / 2.0, 6.0 / 2.0]]
            ),
        )
        assert np.allclose(
            g.interpT(x).D[1],
            np.array(
                [
                    [4.0 / 2.0, 5.0, 6.0 / 2.0],
                    [7.0 / 2.0, 8.0, 9.0 / 2.0],
                    [10.0 / 2.0, 11.0, 12.0 / 2.0],
                ]
            ),
        )

    def test_interpT_2D_torch(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        D1 = torch.tensor([[4.0, 6.0], [7.0, 9.0], [10.0, 12.0]])
        x = g.Cvar(cs, ll, [D0, D1], torch.ones(cs))

        assert g.interpT(x).D[0].shape == (4, 2)
        assert g.interpT(x).D[1].shape == (3, 3)
        assert g.interpT(x).Z.shape == cs
        assert torch.allclose(
            g.interpT(x).D[0],
            torch.tensor(
                [[1.0 / 2.0, 2.0 / 2.0], [2.0, 3.0], [4.0, 5.0], [5.0 / 2.0, 6.0 / 2.0]]
            ),
        )
        assert torch.allclose(
            g.interpT(x).D[1],
            torch.tensor(
                [
                    [4.0 / 2.0, 5.0, 6.0 / 2.0],
                    [7.0 / 2.0, 8.0, 9.0 / 2.0],
                    [10.0 / 2.0, 11.0, 12.0 / 2.0],
                ]
            ),
        )

    def test_interpT_inplace_2D_numpy(self):
        cs = (2, 2)
        ll = (1.0, 1.0)
        x = g.CSvar(np.array([1.0, 2.0]), np.array([5.0, 6.0]), cs[0], ll)

        D0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        # D1 = np.array([[0.0, 2.0, 0.0], [2.0, 4.0, 0.0]])
        # x = g.Cvar(cs, ll, [D0, D1], np.ones(cs))

        np.testing.assert_allclose(x.U.D[0], D0)
        x.V.D[0] = np.array(
            [
                [2.0, 3.0],
                [4.0, 5.0],
            ]
        )
        x.V.D[1] = np.array(
            [
                [1.0, 1.0],
                [3.0, 2.0],
            ]
        )

        g.interpT_(x.U, x.V)

        assert x.U.D[0].shape == (3, 2)
        assert x.U.D[1].shape == (2, 3)
        assert x.U.Z.shape == cs
        np.testing.assert_allclose(
            x.U.D[0],
            np.array(
                [
                    [1.0, 3.0 / 2.0],
                    [3.0, 4.0],
                    [2.0, 5.0 / 2.0],
                ]
            ),
        )
        np.testing.assert_allclose(
            x.U.D[1],
            np.array(
                [
                    [0.5, 1.0, 0.5],
                    [1.5, 2.5, 1.0],
                ]
            ),
        )

    def test_interpT_inplace_2D_torch(self):
        cs = (2, 2)
        ll = (1.0, 1.0)
        x = g.CSvar(torch.tensor([1.0, 2.0]), torch.tensor([5.0, 6.0]), cs[0], ll)

        D0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        torch.testing.assert_allclose(x.U.D[0], D0)
        x.V.D[0] = torch.tensor(
            [
                [2.0, 3.0],
                [4.0, 5.0],
            ]
        )
        x.V.D[1] = torch.tensor(
            [
                [1.0, 1.0],
                [3.0, 2.0],
            ]
        )

        g.interpT_(x.U, x.V)

        assert x.U.D[0].shape == (3, 2)
        assert x.U.D[1].shape == (2, 3)
        assert x.U.Z.shape == cs
        torch.testing.assert_allclose(
            x.U.D[0],
            torch.tensor(
                [
                    [1.0, 3.0 / 2.0],
                    [3.0, 4.0],
                    [2.0, 5.0 / 2.0],
                ]
            ),
        )
        torch.testing.assert_allclose(
            x.U.D[1],
            torch.tensor(
                [
                    [0.5, 1.0, 0.5],
                    [1.5, 2.5, 1.0],
                ]
            ),
        )
