"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

-------------------------------------------------------------------

Vectorized implementation of rANS based on https://arxiv.org/abs/1402.3392
Adapted from https://github.com/j-towns/craystack/blob/master/craystack/rans.py
"""

from warnings import warn

import numpy as np


rng = np.random.default_rng(0)
rans_l = 1 << 31  # the lower bound of the normalisation interval
atleast_1d = lambda x: np.atleast_1d(x).astype('uint64')


def base_message(shape, randomize=False):
    """
    Returns a base ANS message of given shape. If randomize=True,
    populates the lower bits of the head with samples from a Bernoulli(1/2)
    distribution. The tail is empty.
    """
    assert shape and np.prod(shape), 'Shape must be an int > 0' \
                                     'or tuple with length > 0.'
    head = np.full(shape, rans_l, "uint64")
    if randomize:
        head += rng.integers(0, rans_l, size=shape, dtype='uint64')
    return head, ()


def stack_extend(stack, arr):
    return arr, stack


def stack_slice(stack, n):
    slc = []
    while n > 0:
        if stack:
            arr, stack = stack
        else:
            warn(f'Popping from empty message. Generating {32*n} bits of random data.')
            arr, stack = rng.integers(1 << 32, size=n, dtype='uint32'), ()
        if n >= len(arr):
            slc.append(arr)
            n -= len(arr)
        else:
            slc.append(arr[:n])
            stack = arr[n:], stack
            break
    return stack, np.concatenate(slc)


def encode(ans_state, starts, freqs, precisions):
    head, tail = ans_state
    starts, freqs, precisions = map(atleast_1d, (starts, freqs, precisions))
    idxs = head >= ((rans_l // precisions) << 32) * freqs
    if np.any(idxs):
        tail = stack_extend(tail, np.uint32(head[idxs]))
        head = np.copy(head)  # Ensure no side-effects
        head[idxs] >>= 32

    # calculate next state s' = 2^r * (s // p) + (s % p) + c
    head_div_freqs, head_mod_freqs = np.divmod(head, freqs)
    return head_div_freqs*precisions + head_mod_freqs + starts, tail


def decode(ans_state, precisions):
    precisions = atleast_1d(precisions)
    head_, tail_ = ans_state

    # s' mod 2^r
    cfs = head_ % precisions

    def pop(starts, freqs):
        starts, freqs = map(atleast_1d, (starts, freqs))

        # calculate previous state  s = p*(s' // 2^r) + (s' % 2^r) - c
        head = freqs * (head_ // precisions) + cfs - starts

        # check which entries need renormalizing
        idxs = head < rans_l

        # how many 32*n bits do we need from the tail?
        n = np.sum(idxs)
        if n > 0:
            # new_head = 32*n bits from the tail
            # tail = previous tail, with 32*n less bits
            tail, new_head = stack_slice(tail_, n)

            # update LSBs of head, where needed
            head[idxs] = (head[idxs] << 32) | new_head
        else:
            tail = tail_
        return head, tail
    return cfs, pop


def flatten(ans_state):
    """Flatten a vrans state ans_state into a 1d numpy array."""
    head, ans_state = np.ravel(ans_state[0]), ans_state[1]
    out = [np.uint32(head >> 32), np.uint32(head)]
    while ans_state:
        head, ans_state = ans_state
        out.append(head)
    return np.concatenate(out)


def unflatten(arr, shape):
    """Unflatten a 1d numpy array into a vrans state."""
    size = np.prod(shape)
    head = np.uint64(arr[:size]) << 32 | np.uint64(arr[size:2 * size])
    return np.reshape(head, shape), (arr[2 * size:], ())
