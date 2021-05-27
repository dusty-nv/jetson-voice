#!/usr/bin/env python3
# coding: utf-8

import numpy as np

    
def softmax(x, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of x.

    Parameters
    ----------
      x: ND-Array. Probably should be floats.
    
      theta (optional): float parameter, used as a multiplier
                        prior to exponentiation. Default = 1.0
        
      axis (optional): axis to compute values along. Default is the
                       first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    y = np.atleast_2d(x)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(x.shape) == 1: p = p.flatten()

    return p


def normalize_logits(logits):
    """
    Normalize logits such that they are distributed between [0,1]
    """
    return np.exp(logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True)))           

              