"""Pointwise math for scalar fields; numeric inputs stay numeric."""

from __future__ import annotations

import builtins

import numpy as np

from .expr import Scalar, as_scalar, call, max_, min_

pi = float(np.pi)
inf = float(np.inf)


def _symbolic(*values):
    return any(isinstance(value, Scalar) for value in values)


def _unary(name, value, numeric):
    return call(name, value) if _symbolic(value) else numeric(value)


def abs(value):
    return _unary("abs", value, builtins.abs)


def sqrt(value):
    return _unary("sqrt", value, np.sqrt)


def sin(value):
    return _unary("sin", value, np.sin)


def cos(value):
    return _unary("cos", value, np.cos)


def tan(value):
    return _unary("tan", value, np.tan)


def floor(value):
    return _unary("floor", value, np.floor)


def ceil(value):
    return _unary("ceil", value, np.ceil)


def exp(value):
    return _unary("exp", value, np.exp)


def log(value):
    return _unary("log", value, np.log)


def atan2(y, x):
    return call("atan2", y, x) if _symbolic(y, x) else np.arctan2(y, x)


def minimum(a, b):
    return min_(a, b) if _symbolic(a, b) else np.minimum(a, b)


def maximum(a, b):
    return max_(a, b) if _symbolic(a, b) else np.maximum(a, b)


def hypot(a, b):
    return sqrt(a * a + b * b)


def clip(value, low, high):
    return (
        call("clip", value, low, high) if _symbolic(value, low, high) else np.clip(value, low, high)
    )


def where(condition, yes, no):
    if _symbolic(condition, yes, no):
        if as_scalar(condition).kind != "bool":
            raise TypeError("where() condition must be a boolean field")
        return call("where", condition, yes, no, kind=as_scalar(yes).kind)
    return np.where(condition, yes, no)


def logical_xor(a, b):
    return call("xor", a, b, kind="bool") if _symbolic(a, b) else np.logical_xor(a, b)


def logical_and(a, b):
    return call("and", a, b, kind="bool") if _symbolic(a, b) else np.logical_and(a, b)


def logical_or(a, b):
    return call("or", a, b, kind="bool") if _symbolic(a, b) else np.logical_or(a, b)


def full_like(value, fill_value, dtype=None):
    if _symbolic(value):
        result = as_scalar(fill_value)
        if dtype is bool and result.kind != "bool":
            return as_scalar(bool(fill_value))
        return result
    return np.full_like(value, fill_value, dtype=dtype)


def zeros_like(value, dtype=None):
    return full_like(value, False if dtype is bool else 0, dtype=dtype)


def array(values, dtype=None):
    return np.asarray(values, dtype=dtype)


def interp(value, xp, fp):
    """Piecewise linear interpolation with endpoint clamping."""
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    if xp.ndim != 1 or fp.ndim != 1 or len(xp) != len(fp) or len(xp) < 1:
        raise ValueError("interp() requires matching nonempty one-dimensional knot arrays")
    if np.any(np.diff(xp) <= 0):
        raise ValueError("interp() knots must increase strictly")
    if not _symbolic(value):
        return np.interp(value, xp, fp)
    result = as_scalar(float(fp[-1]))
    for index in range(len(xp) - 2, -1, -1):
        slope = float((fp[index + 1] - fp[index]) / (xp[index + 1] - xp[index]))
        segment = float(fp[index]) + slope * (value - float(xp[index]))
        result = where(value < float(xp[index + 1]), segment, result)
    return where(value < float(xp[0]), float(fp[0]), result)
