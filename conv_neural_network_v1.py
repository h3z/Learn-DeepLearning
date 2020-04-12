import numpy as np


def zero_pad(A, pad):
    return np.pad(A, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)


def convolution(A_prev_slice, W, b):
    return np.sum(np.multiply(A_prev_slice, W) + b)


def conv_forward(A_prev, W, b, hparameters):
    cache = (A_prev, W, b, hparameters)

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = (n_H_prev - f + 2 * pad) // stride + 1
    n_W = (n_W_prev - f + 2 * pad) // stride + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev = zero_pad(A_prev, pad)

    for h in range(n_H):
        for w in range(n_W):
            for i in range(m):
                A_prev_pad = A_prev[i]
                for j in range(n_C):
                    A_prev_slice = A_prev_pad[h * stride: h * stride + f, w * stride: w * stride + f, :]
                    Z[i, h, w, j] = convolution(A_prev_slice, W[..., j], b[..., j])

    return Z, cache

