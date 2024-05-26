import proximal.backend_extension as be
import proximal.dynamicUOT as dyn
import proximal.grids as gr
import numpy as np
import torch


class TestRoot:
    def test_root_numpy_positive_delta(self):
        coeff = np.array([1, -4, 6, -24]).astype(np.float32)
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), np.array([4.0]))

    def test_root_torch_positive_delta(self):
        coeff = torch.tensor([1, -4, 6, -24]).float()
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), torch.tensor([4.0]))

    def test_root_numpy_negative_delta(self):
        coeff = np.array([1, -5, 1, -5]).astype(np.float32)
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), np.array([5.0]))

    def test_root_torch_negative_delta(self):
        coeff = torch.tensor([1, -5, 1, -5]).float()
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), torch.tensor([5.0]))

    def test_root_numpy_zero_delta(self):
        coeff = np.array([1, -3, 3, -1]).astype(np.float32)
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), np.array([1.0]))

    def test_root_torch_zero_delta(self):
        coeff = torch.tensor([1, -3, 3, -1]).float()
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), torch.tensor([1.0]))

    def test_root_numpy_multi_input(self):
        a = np.array([1, 1, 1]).astype(np.float32)
        b = np.array([-4, -5, -3]).astype(np.float32)
        c = np.array([6, 1, 3]).astype(np.float32)
        d = np.array([-24, -5, -1]).astype(np.float32)
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

    def test_posison_numpy_source_nonzero(self):
        N_0 = 100
        N_1 = 100
        tol = (1 / N_0) ** 2 + (1 / N_1) ** 2
        ll = (1.0, 1.0)
        t = ll[0] * (np.arange(N_0) + 0.5) / N_0
        x = ll[1] * (np.arange(N_1) + 0.5) / N_1
        T, X = np.meshgrid(t, x)
        f = (
            (8 * np.pi**2 + 1)
            * np.cos(np.pi * (2 * T - 1))
            * np.cos(np.pi * (2 * X - 1))
        )
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, ll, True, nx)
        np.testing.assert_allclose(
            f,
            np.cos(np.pi * (2 * T - 1)) * np.cos(np.pi * (2 * X - 1)),
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
    def test_projCE_numpy_small(self):
        rho0 = np.array([1, 2, 3, 4]).astype(np.float32)
        rho1 = np.array([5, 6, 7, 8]).astype(np.float32)
        T = 5
        cs = (T, 4)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs)
        D = [np.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=np.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, True)
        np.testing.assert_allclose(U.remainder_CE(), 0, atol=1e-5, rtol=1e-8)

    def test_projCE_torch_small(self):
        rho0 = torch.tensor([1, 2, 3, 4]).float()
        rho1 = torch.tensor([5, 6, 7, 8]).float()
        T = 5
        cs = (T, 4)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs)
        D = [torch.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=torch.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, True)
        torch.testing.assert_close(
            U.remainder_CE(), torch.zeros(cs), atol=1e-5, rtol=1e-8
        )

    def test_proJCE_numpy_large(self):
        rho0 = np.random.rand(256).astype(np.float32)
        rho1 = np.random.rand(256).astype(np.float32)
        T = 100
        cs = (T, 256)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs)
        D = [np.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=np.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, True)
        np.testing.assert_allclose(U.remainder_CE(), 0, atol=1e-5, rtol=1e-8)


class TestInvQ_Mul_A_:
    def test_invQ_Mul_A_numpy_small1(self):
        src = np.array([[1, 2], [3, 4]]).astype(np.float32)
        Q = np.array([[1, 2], [0, 1]]).astype(np.float32)
        nx = be.get_backend_ext(src, Q)
        dyn.invQ_mul_A_(src, Q, src, 0, nx)
        np.testing.assert_allclose(src, np.array([[-5, -6], [3, 4]]))

    def test_invQ_Mul_A_torch_small1(self):
        src = torch.tensor([[1, 2], [3, 4]]).float()
        Q = torch.tensor([[1, 2], [0, 1]]).float()
        nx = be.get_backend_ext(src, Q)
        dyn.invQ_mul_A_(src, Q, src, 0, nx)
        torch.testing.assert_close(src, torch.tensor([[-5, -6], [3, 4]]).float())

    def test_invQ_Mul_A_numpy_small2(self):
        src = np.array([[1, 2], [3, 4]]).astype(np.float32)
        Q = np.array([[1, 2], [0, 1]]).astype(np.float32)
        nx = be.get_backend_ext(src, Q)
        dyn.invQ_mul_A_(src, Q, src, 1, nx)
        np.testing.assert_allclose(src, np.array([[-3, 2], [-5, 4]]))

    def test_invQ_Mul_A_torch_small2(self):
        src = torch.tensor([[1, 2], [3, 4]]).float()
        Q = torch.tensor([[1, 2], [0, 1]]).float()
        nx = be.get_backend_ext(src, Q)
        dyn.invQ_mul_A_(src, Q, src, 1, nx)
        torch.testing.assert_close(src, torch.tensor([[-3, 2], [-5, 4]]).float())


class TestProjInterp_:
    def test_projInterp_numpy_small(self):
        rho0 = np.array([1, 2]).astype(np.float32)
        rho1 = np.array([5, 6]).astype(np.float32)
        T = 2
        ll = (1.0, 1.0)
        x = gr.CSvar(rho0, rho1, T, ll)
        y = gr.CSvar(rho0, rho1, T, ll)
        x.U.D[1] = np.array([[0, 2, 0], [2, 4, 0]]).astype(np.float32)
        x.interp_()

        assert x.U.N == 2

        np.testing.assert_allclose(
            x.U.D[0], np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        )
        np.testing.assert_allclose(
            x.V.D[0], np.array([[2, 3], [4, 5]]).astype(np.float32)
        )
        np.testing.assert_allclose(
            x.V.D[1], np.array([[1, 1], [3, 2]]).astype(np.float32)
        )

        identities = [
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32),
        ]
        Q = [
            np.array([[1, 2, 3], [0, 1, 5], [0, 0, 1]]).astype(np.float32),
            np.array([[1, 7, 8], [0, 1, 9], [0, 0, 1]]).astype(np.float32),
        ]
        dyn.projinterp_(y, x, identities)
        dyn.projinterp_(x, x, Q)
        np.testing.assert_allclose(y.U.D[0], np.array([[2, 3.5], [6, 8], [7, 8.5]]))
        np.testing.assert_allclose(y.U.D[1], np.array([[0.5, 3, 0.5], [3.5, 6.5, 1.0]]))
        np.testing.assert_allclose(
            x.U.D[0], np.array([[39, 47], [-29, -34.5], [7, 8.5]])
        )
        np.testing.assert_allclose(x.U.D[1], np.array([[7, -1.5, 0.5], [13, -2.5, 1]]))
        np.testing.assert_allclose(x.U.Z, np.array([[0, 0], [0, 0]]))


class TestProjInterp_Constraint:
    def test_projInterp_constraint_numpy_small(self):
        rho0 = np.array([1, 2]).astype(np.float32)
        rho1 = np.array([5, 6]).astype(np.float32)
        T = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        x = gr.CSvar(rho0, rho1, T, ll)
        y = gr.CSvar(rho0, rho1, T, ll)
        x.U.D[1] = np.array([[0, 2, 0], [2, 4, 0]]).astype(np.float32)
        x.interp_()
        identities = [
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32),
        ]
        H = np.ones(cs, dtype=np.float32)
        F = np.array([7, 8], dtype=np.float32)
        dyn.projinterp_constraint_(y, x, identities, np.eye(2), H, F)
        np.testing.assert_allclose(
            y.U.D[0],
            np.array([[2.53125, 4.03125], [6.6875, 8.6875], [7.15625, 8.65625]]),
        )
        np.testing.assert_allclose(y.U.D[1], np.array([[0.5, 3, 0.5], [3.5, 6.5, 1.0]]))

    def test_projInterp_constraint_torch_small(self):
        rho0 = torch.tensor([1, 2]).float()
        rho1 = torch.tensor([5, 6]).float()
        T = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        x = gr.CSvar(rho0, rho1, T, ll)
        y = gr.CSvar(rho0, rho1, T, ll)
        x.U.D[1] = torch.tensor([[0, 2, 0], [2, 4, 0]]).float()
        x.interp_()
        identities = [
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float(),
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float(),
        ]
        H = torch.ones(cs, dtype=torch.float32)
        F = torch.tensor([7, 8], dtype=torch.float32)
        dyn.projinterp_constraint_(y, x, identities, torch.eye(2), H, F)
        torch.testing.assert_close(
            y.U.D[0],
            torch.tensor([[2.53125, 4.03125], [6.6875, 8.6875], [7.15625, 8.65625]]),
        )
        torch.testing.assert_close(
            y.U.D[1], torch.tensor([[0.5, 3, 0.5], [3.5, 6.5, 1.0]])
        )

    def test_projInterp_constraint_numpy_small_2(self):
        rho0 = np.array([1, 2]).astype(np.float32)
        rho1 = np.array([5, 6]).astype(np.float32)
        T = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        x = gr.CSvar(rho0, rho1, T, ll)
        x.U.D[1] = np.array([[0, 2, 0], [2, 4, 0]]).astype(np.float32)
        x.interp_()
        Q = [
            np.array([[1, 2, 3], [0, 1, 5], [0, 0, 1]]).astype(np.float32),
            np.array([[1, 7, 8], [0, 1, 9], [0, 0, 1]]).astype(np.float32),
        ]
        H = np.ones(cs, dtype=np.float32)
        F = np.array([7, 8], dtype=np.float32)
        dyn.projinterp_constraint_(x, x, Q, np.array([[1, 3], [0, 1]]), H, F)
        np.testing.assert_allclose(
            x.U.D[0],
            np.array([[78.65625, 86.65625], [-63.65625, -69.15625], [12.0, 13.5]]),
        )
        np.testing.assert_allclose(x.U.D[1], np.array([[7, -1.5, 0.5], [13, -2.5, 1]]))

    def test_projInterp_constraint_torch_small_2(self):
        rho0 = torch.tensor([1, 2]).float()
        rho1 = torch.tensor([5, 6]).float()
        T = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        x = gr.CSvar(rho0, rho1, T, ll)
        x.U.D[1] = torch.tensor([[0, 2, 0], [2, 4, 0]]).float()
        x.interp_()
        Q = [
            torch.tensor([[1, 2, 3], [0, 1, 5], [0, 0, 1]]).float(),
            torch.tensor([[1, 7, 8], [0, 1, 9], [0, 0, 1]]).float(),
        ]
        H = torch.ones(cs, dtype=torch.float32)
        F = torch.tensor([7, 8], dtype=torch.float32)
        dyn.projinterp_constraint_(
            x, x, Q, torch.tensor([[1, 3], [0, 1]]).float(), H, F
        )
        torch.testing.assert_close(
            x.U.D[0],
            torch.tensor([[78.65625, 86.65625], [-63.65625, -69.15625], [12.0, 13.5]]),
        )
        torch.testing.assert_close(
            x.U.D[1], torch.tensor([[7, -1.5, 0.5], [13, -2.5, 1]])
        )


class TestPrecomputeProjectInterp:
    def test_precompute_project_interp_numpy_small(self):
        cs = (3, 4)
        z = np.zeros(cs, dtype=np.float64)
        Q = dyn.precomputeProjInterp(cs, z, z)
        np.testing.assert_allclose(
            Q[0],
            np.array(
                [
                    [5.0 / 4.0, 1.0 / 4.0, 0.0, 0.0],
                    [1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0],
                    [0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0],
                    [0.0, 0.0, 1.0 / 4.0, 5.0 / 4.0],
                ]
            ),
        )
        np.testing.assert_allclose(
            Q[1],
            np.array(
                [
                    [5.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0],
                    [1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0, 0.0],
                    [0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0],
                    [0.0, 0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0],
                    [0.0, 0.0, 0.0, 1.0 / 4.0, 5.0 / 4.0],
                ]
            ),
        )

    def test_precompute_project_interp_torch_small(self):
        cs = (3, 4)
        z = torch.zeros(cs, dtype=torch.float64)
        Q = dyn.precomputeProjInterp(cs, z, z)
        torch.testing.assert_close(
            Q[0],
            torch.tensor(
                [
                    [5.0 / 4.0, 1.0 / 4.0, 0.0, 0.0],
                    [1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0],
                    [0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0],
                    [0.0, 0.0, 1.0 / 4.0, 5.0 / 4.0],
                ],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(
            Q[1],
            torch.tensor(
                [
                    [5.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0],
                    [1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0, 0.0],
                    [0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0],
                    [0.0, 0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0],
                    [0.0, 0.0, 0.0, 1.0 / 4.0, 5.0 / 4.0],
                ],
                dtype=torch.float64,
            ),
        )


class TestprecomputeHQH:
    def test_precompute_HQH_numpy_small(self):
        cs = (3, 4)
        ll = (1.0, 1.0)
        z = np.zeros(cs, dtype=np.float64)
        Q = dyn.precomputeProjInterp(cs, z, z)
        H = np.ones(cs, dtype=np.float64)
        Q = dyn.precomputeHQH(Q[0], H, cs, ll)
        np.testing.assert_allclose(
            Q[:, 0],
            np.array([0.0784313725490196, 0.0294117647058824, -0.0049019607843137]),
        )

    def test_precompute_HQH_torch_small(self):
        cs = (3, 4)
        ll = (1.0, 1.0)
        z = torch.zeros(cs)
        Q = dyn.precomputeProjInterp(cs, z, z)
        H = torch.ones(cs)
        Q = dyn.precomputeHQH(Q[0], H, cs, ll)
        torch.testing.assert_close(
            Q[:, 0],
            torch.tensor([0.0784313725490196, 0.0294117647058824, -0.0049019607843137]),
        )


class TestStepDR:
    def test_step_DR_numpy_small(self):
        def prox1(y, x):
            y.U.D[0][...] = 2 * x.U.D[0]

        def prox2(z, w):
            pass

        rho0 = np.array([1, 2]).astype(np.float32)
        rho1 = np.array([5, 6]).astype(np.float32)
        T = 2
        ll = (1.0, 1.0)
        w, x, y, z = [gr.CSvar(rho0, rho1, T, ll) for _ in range(4)]
        w.U.D[0] = np.array([[1, 1], [1, 1], [1, 1]]).astype(np.float32)
        z.U.D[0] = np.array([[0, 0], [0, 0], [0, 0]]).astype(np.float32)

        assert isinstance(x.U, gr.Svar)

        w, x, y, z = dyn.stepDR(w, x, y, z, prox1, prox2, 1.0)
        np.testing.assert_allclose(
            y.U.D[0], np.array([[-2.0, -2.0], [-2.0, -2.0], [-2.0, -2.0]])
        )


class TestComputeGeodesic:

    def test_computeGeodesic_numpy_samedist(self):
        # rho0 = rho1, so everything should be one and the momentum should be zero
        rho0 = np.array([1, 1, 1]).astype(np.float32)
        rho1 = np.array([1, 1, 1]).astype(np.float32)
        T = 5
        ll = (1.0, 1.0)
        z, list = dyn.computeGeodesic(rho0, rho1, T, ll)
        np.testing.assert_allclose(z.U.D[0], np.ones((T + 1, 3)))
        np.testing.assert_allclose(z.U.D[1], np.zeros((T, 4)))
        np.testing.assert_allclose(z.U.Z, np.zeros((T, 3)))

    def test_computeGeodesic_numpy_samedist_constraint(self):
        # rho0 = rho1, so everything should be one and the momentum should be zero
        rho0 = np.array([1, 1, 1]).astype(np.float32)
        rho1 = np.array([1, 1, 1]).astype(np.float32)
        T = 5
        ll = (1.0, 1.0)
        H = np.ones((T, 3), dtype=np.float32)
        F = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        z, list = dyn.computeGeodesic(rho0, rho1, T, ll, H, F)
        np.testing.assert_allclose(z.U.D[0], np.ones((T + 1, 3)))
        np.testing.assert_allclose(z.U.D[1], np.zeros((T, 4)))
        np.testing.assert_allclose(z.U.Z, np.zeros((T, 3)))
