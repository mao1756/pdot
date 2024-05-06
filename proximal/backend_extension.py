import ot.backend as ob
import numpy as np
import scipy as sp

try:
    import torch
except ImportError:
    torch = False

try:
    import torch_dct
except ImportError:
    torch_dct = False

"""
Extension of the backend module to include new functions.
Currently only numpy and torch backends are supported.
"""


class NumpyBackend_ext(ob.NumpyBackend):
    def __init__(self):
        super().__init__()

    def roll(self, x, shift, axis=None):
        return np.roll(x, shift, axis)

    def diff(self, x, n=1, axis=-1, prepend=None, append=None):
        if prepend is None and append is None:
            return np.diff(x, n=n, axis=axis)
        elif prepend is None:
            return np.diff(x, n=n, axis=axis, append=append)
        elif append is None:
            return np.diff(x, n=n, axis=axis, prepend=prepend)
        else:
            return np.diff(x, n=n, axis=axis, prepend=prepend, append=append)

    def dct(self, x, axis=-1, norm="ortho"):
        return sp.fft.dct(x, axis=axis, norm=norm)

    def idct(self, x, axis=-1, norm="ortho"):
        return sp.fft.idct(x, axis=axis, norm=norm)


class TorchBackend_ext(ob.TorchBackend):
    def __init__(self):
        super().__init__()

    def roll(self, x, shift, axis=None):
        return torch.roll(x, shift, axis)

    def diff(self, x, n=1, axis=-1, prepend=None, append=None):
        if prepend is None and append is None:
            return torch.diff(x, n=n, axis=axis)
        elif prepend is None:
            return torch.diff(x, n=n, axis=axis, append=append)
        elif append is None:
            return torch.diff(x, n=n, axis=axis, prepend=prepend)
        else:
            return torch.diff(x, n=n, axis=axis, prepend=prepend, append=append)

    def dct(self, x, axis=-1, norm="ortho"):
        if torch_dct:
            transposed = torch.transpose(x, -1, axis)
            dct_d = torch_dct.dct(transposed, norm=norm)
            return torch.transpose(dct_d, -1, axis)
        else:
            raise ValueError("torch_dct not available")

    def idct(self, x, axis=-1, norm="ortho"):
        if torch_dct:
            transposed = torch.transpose(x, -1, axis)
            idct_d = torch_dct.idct(transposed, norm=norm)
            return torch.transpose(idct_d, -1, axis)
        else:
            raise ValueError("torch_dct not available")


def get_backend_ext(*args):
    nx = ob.get_backend(*args)

    # Add new functions to the backend
    if isinstance(nx, ob.NumpyBackend):
        nx = NumpyBackend_ext()
    elif isinstance(nx, ob.TorchBackend):
        nx = TorchBackend_ext()
    else:
        raise ValueError(f"Backend not supported, backend={nx.__class__}")

    return nx
