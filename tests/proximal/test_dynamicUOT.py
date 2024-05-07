import proximal.backend_extension as be
import proximal.dynamicUOT as dyn
import numpy as np
import torch


class TestRoot:
    def test_root_numpy_positive_delta(self):
        coeff = np.array([1, -4, 6, -24])
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), np.array([4.0]))

    def test_root_torch_positive_delta(self):
        coeff = torch.tensor([1, -4, 6, -24])
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), torch.tensor([4.0]))

    def test_root_numpy_negative_delta(self):
        coeff = np.array([1, -5, 1, -5])
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), np.array([5.0]))

    def test_root_torch_negative_delta(self):
        coeff = torch.tensor([1, -5, 1, -5])
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), torch.tensor([5.0]))

    def test_root_numpy_zero_delta(self):
        coeff = np.array([1, -3, 3, -1])
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), np.array([1.0]))

    def test_root_torch_zero_delta(self):
        coeff = torch.tensor([1, -3, 3, -1])
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), torch.tensor([1.0]))

    def test_root_numpy_multi_input(self):
        a = np.array([1, 1, 1])
        b = np.array([-4, -5, -3])
        c = np.array([6, 1, 3])
        d = np.array([-24, -5, -1])
        nx = be.get_backend_ext(a, b, c, d)
        assert np.allclose(dyn.root(a, b, c, d, nx), np.array([4.0, 5.0, 1.0]))


class TestProxB:
    def test_proxB_numpy_small(self):
        R = np.array([[1, 2], [3, 4]]).astype(np.float64)
        M = np.array([[5, 6], [7, 8]]).astype(np.float64)
        Z = np.array([[9, 10], [11, 12]]).astype(np.float64)
        M = [M, Z]
        nx = be.get_backend_ext(R, M[0], M[1])
        gamma = 1.0
        destR = np.zeros_like(R)
        destM = [np.zeros_like(M[0]), np.zeros_like(M[1])]
        dyn.proxB_(destR, destM, R, M, gamma=gamma, nx=nx)
        assert np.allclose(
            destR,
            np.array(
                [
                    [3.55474548856287, 4.36366412618957],
                    [5.20656376605373, 6.07669399212356],
                ]
            ),
        )
        assert np.allclose(
            destM[0],
            np.array(
                [
                    [3.90224382184357, 4.88136172235257],
                    [5.87216175264550, 6.86952862326616],
                ]
            ),
        )
        assert np.allclose(
            destM[1],
            np.array(
                [
                    [7.02403887931843, 8.13560287058762],
                    [9.22768275415721, 10.3042929348992],
                ]
            ),
        )

    def test_proxB_torch_small(self):
        R = torch.tensor([[1, 2], [3, 4]]).float()
        M = torch.tensor([[5, 6], [7, 8]]).float()
        Z = torch.tensor([[9, 10], [11, 12]]).float()
        M = [M, Z]
        nx = be.get_backend_ext(R, M[0], M[1])
        gamma = 1.0
        destR = torch.zeros_like(R)
        destM = [torch.zeros_like(M[0]), torch.zeros_like(M[1])]
        dyn.proxB_(destR, destM, R, M, gamma=gamma, nx=nx)
        assert torch.allclose(
            destR,
            torch.tensor(
                [
                    [3.55474548856287, 4.36366412618957],
                    [5.20656376605373, 6.07669399212356],
                ]
            ),
        )
        assert torch.allclose(
            destM[0],
            torch.tensor(
                [
                    [3.90224382184357, 4.88136172235257],
                    [5.87216175264550, 6.86952862326616],
                ]
            ),
        )
        assert torch.allclose(
            destM[1],
            torch.tensor(
                [
                    [7.02403887931843, 8.13560287058762],
                    [9.22768275415721, 10.3042929348992],
                ]
            ),
        )


class TestPoisson:
    def test_poisson_numpy_nosource_zero(self):
        f = np.zeros((3, 3))
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, (1.0, 1.0), False, nx)
        assert np.allclose(f, np.zeros((3, 3)))

    def test_poisson_torch_nosource_zero(self):
        f = torch.zeros((3, 3))
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, (1.0, 1.0), False, nx)
        assert torch.allclose(f, torch.zeros((3, 3)))

    def test_poisson_numpy_source_zero(self):
        f = np.zeros((3, 3))
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, (1.0, 1.0), True, nx)
        assert np.allclose(f, np.zeros((3, 3)))

    def test_poisson_torch_source_zero(self):
        f = torch.zeros((3, 3))
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, (1.0, 1.0), True, nx)
        assert torch.allclose(f, torch.zeros((3, 3)))

    def test_poisson_numpy_nosource_nonzero(self):
        N_0 = 100
        N_1 = 100
        tol = (1 / N_0) ** 2 + (1 / N_1) ** 2
        ll = (1.0, 1.0)
        t = ll[0] * (np.arange(N_0) + 0.5) / N_0
        x = ll[1] * (np.arange(N_1) + 0.5) / N_1
        T, X = np.meshgrid(t, x)
        f = 8 * np.pi**2 * np.cos(np.pi * (2 * T - 1)) * np.cos(np.pi * (2 * X - 1))
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, ll, False, nx)
        np.testing.assert_allclose(
            f,
            np.cos(np.pi * (2 * T - 1)) * np.cos(np.pi * (2 * X - 1)),
            atol=tol,
            rtol=tol,
        )

    def test_poisson_torch_nosource_nonzero(self):
        N_0 = 100
        N_1 = 100
        tol = (1 / N_0) ** 2 + (1 / N_1) ** 2
        ll = (1.0, 1.0)
        t = ll[0] * (torch.arange(N_0) + 0.5) / N_0
        x = ll[1] * (torch.arange(N_1) + 0.5) / N_1
        T, X = torch.meshgrid(t, x)
        f = (
            8
            * np.pi**2
            * torch.cos(np.pi * (2 * T - 1))
            * torch.cos(np.pi * (2 * X - 1))
        )
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, ll, False, nx)
        torch.testing.assert_allclose(
            f,
            torch.cos(np.pi * (2 * T - 1)) * torch.cos(np.pi * (2 * X - 1)),
            atol=tol,
            rtol=tol,
        )


class TestMinusInterior:
    def test_minus_interior_numpy_small(self):
        M = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [6, 7, 8, 9], [10, 11, 12, 13]]
        ).astype(np.float64)
        dest = M.copy()
        dpk = np.array([[6, 6, 7, 8], [6, 7, 8, 9]]).astype(np.float64)
        cs = (3, 4)
        dim = 0
        dyn.minus_interior_(dest, M, dpk, cs, dim)
        assert np.allclose(
            dest,
            np.array([[1, 2, 3, 4], [-1, 0, 0, 0], [0, 0, 0, 0], [10, 11, 12, 13]]),
        )

    def test_minus_interior_torch_small(self):
        M = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [6, 7, 8, 9], [10, 11, 12, 13]]
        ).float()
        dest = M.clone()
        dpk = torch.tensor([[6, 6, 7, 8], [6, 7, 8, 9]]).float()
        cs = (3, 4)
        dim = 0
        dyn.minus_interior_(dest, M, dpk, cs, dim)
        assert torch.allclose(
            dest,
            torch.tensor(
                [[1, 2, 3, 4], [-1, 0, 0, 0], [0, 0, 0, 0], [10, 11, 12, 13]],
                dtype=torch.float32,
            ),
        )


class TestProjCE:
    pass
