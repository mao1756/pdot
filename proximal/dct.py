from ot.backend import Backend

"""
Implementation of DCT-2 & 3 in backend agnostic code using ot.backend

The implementation is based on the following reference:
https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft
https://dsp.stackexchange.com/questions/51311/computation-of-the-inverse-dct-idct-using-dct-or-ifft
"""

# ToDo: FFT and IFFT is not on the backend, so we need a new algo or use scipy fft for
# the time being


def dct(x, nx: Backend):
    """Compute the Discrete Cosine Transform of the 1D array x.

    Args:
        x (array): The input array.
        nx (Backend): The backend module used for computation such as numpy or torch.

    """
    v = nx.empty_like(x)
    N = x.shape[0]
    v[: (N - 1) // 2 + 1] = x[::2]

    if N % 2:  # odd length
        v[(N - 1) // 2 + 1 :] = x[-2::-2]
    else:  # even length
        v[(N - 1) // 2 + 1 :] = x[::-2]

    V = nx.fft(v)

    k = nx.arange(N)
    V *= 2 * nx.exp(-1j * nx.pi * k / (2 * N))
    return V.real


def idct(x, nx: Backend):
    N = x.shape[0]

    # Inverse IDCT Using FFT
    vGrid = nx.arange(N)

    vShiftGrid = nx.exp((1j * nx.pi * vGrid) / (2 * N))
    vShiftGrid = vShiftGrid * nx.sqrt(2 * N)
    vShiftGrid[0] = vShiftGrid[0] / nx.sqrt(2)

    vTmp = vShiftGrid * x
    vTmp = nx.ifft(vTmp).real

    vX = nx.zeros(N)

    vX[0::2] = vTmp[: N // 2]  # even indices
    vX[1::2] = vTmp[-1 : -(N // 2 + 1) : -1]  # odd indices in reverse order

    return vX
