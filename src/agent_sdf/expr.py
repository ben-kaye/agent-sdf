"""Pointwise scalar expressions shared by CPU evaluation and GLSL rendering."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np


@dataclass(frozen=True, eq=False)
class Scalar:
    op: str
    args: tuple = ()
    kind: str = "float"

    def _binary(self, op, other, *, kind="float"):
        return Scalar(op, (self, as_scalar(other)), kind)

    def _reverse(self, op, other, *, kind="float"):
        return Scalar(op, (as_scalar(other), self), kind)

    def __add__(self, other):
        return self._binary("add", other)

    def __radd__(self, other):
        return self._reverse("add", other)

    def __sub__(self, other):
        return self._binary("sub", other)

    def __rsub__(self, other):
        return self._reverse("sub", other)

    def __mul__(self, other):
        return self._binary("mul", other)

    def __rmul__(self, other):
        return self._reverse("mul", other)

    def __truediv__(self, other):
        return self._binary("div", other)

    def __rtruediv__(self, other):
        return self._reverse("div", other)

    def __pow__(self, other):
        return self._binary("pow", other)

    def __rpow__(self, other):
        return self._reverse("pow", other)

    def __neg__(self):
        return Scalar("neg", (self,))

    def __pos__(self):
        return self

    def __abs__(self):
        return Scalar("abs", (self,))

    def __lt__(self, other):
        return self._binary("lt", other, kind="bool")

    def __le__(self, other):
        return self._binary("le", other, kind="bool")

    def __gt__(self, other):
        return self._binary("gt", other, kind="bool")

    def __ge__(self, other):
        return self._binary("ge", other, kind="bool")

    def __eq__(self, other):
        return self._binary("eq", other, kind="bool")

    def __ne__(self, other):
        return self._binary("ne", other, kind="bool")

    def __and__(self, other):
        return self._binary("and", other, kind="bool")

    def __rand__(self, other):
        return self._reverse("and", other, kind="bool")

    def __or__(self, other):
        return self._binary("or", other, kind="bool")

    def __ror__(self, other):
        return self._reverse("or", other, kind="bool")

    def __xor__(self, other):
        return self._binary("xor", other, kind="bool")

    def __rxor__(self, other):
        return self._reverse("xor", other, kind="bool")

    def __invert__(self):
        return Scalar("not", (self,), "bool")

    def __bool__(self):
        raise TypeError("a field condition cannot be used as a Python bool; use where()")


def as_scalar(value) -> Scalar:
    if isinstance(value, Scalar):
        return value
    if isinstance(value, (bool, np.bool_)):
        return Scalar("const", (bool(value),), "bool")
    if isinstance(value, (Real, np.number)):
        return Scalar("const", (float(value),))
    raise TypeError(f"expected a scalar field or number, got {type(value).__name__}")


def coordinates() -> tuple[Scalar, Scalar, Scalar]:
    return tuple(Scalar("var", (name,)) for name in "xyz")


def call(op: str, *args, kind: str = "float") -> Scalar:
    return Scalar(op, tuple(as_scalar(arg) for arg in args), kind)


def _reduce(op, values):
    values = tuple(values)
    if not values:
        raise ValueError(f"{op}_ requires at least one field")
    result = as_scalar(values[0])
    for value in values[1:]:
        result = call(op, result, value)
    return result


def min_(*values):
    """Pointwise minimum of scalar fields or constants."""
    if len(values) == 1 and isinstance(values[0], (tuple, list)):
        values = tuple(values[0])
    return _reduce("min", values)


def max_(*values):
    """Pointwise maximum of scalar fields or constants."""
    if len(values) == 1 and isinstance(values[0], (tuple, list)):
        values = tuple(values[0])
    return _reduce("max", values)


_EVAL = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "pow": np.power,
    "neg": np.negative,
    "abs": np.abs,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
    "eq": np.equal,
    "ne": np.not_equal,
    "and": np.logical_and,
    "or": np.logical_or,
    "xor": np.logical_xor,
    "not": np.logical_not,
    "min": np.minimum,
    "max": np.maximum,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "floor": np.floor,
    "ceil": np.ceil,
    "exp": np.exp,
    "log": np.log,
    "atan2": np.arctan2,
    "where": np.where,
    "clip": np.clip,
}


def evaluate(expression, x, y, z):
    """Evaluate a field at scalar or NumPy array coordinates."""
    expression = as_scalar(expression)
    cache = {}
    variables = {"x": x, "y": y, "z": z}

    def visit(node):
        key = id(node)
        if key in cache:
            return cache[key]
        if node.op == "const":
            result = node.args[0]
        elif node.op == "var":
            result = variables[node.args[0]]
        else:
            result = _EVAL[node.op](*(visit(arg) for arg in node.args))
        cache[key] = result
        return result

    return visit(expression)


_BINARY_GLSL = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
    "and": "&&",
    "or": "||",
    "xor": "!=",
}


def _literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if not isfinite(value):
        bits = np.asarray(value, dtype=np.float32).view(np.uint32).item()
        return f"uintBitsToFloat({bits}u)"
    result = repr(float(value))
    return result if "." in result or "e" in result else result + ".0"


def _glsl_op(node, args):
    op = node.op
    if op in _BINARY_GLSL:
        return f"({args[0]} {_BINARY_GLSL[op]} {args[1]})"
    if op == "neg":
        return f"(-{args[0]})"
    if op == "not":
        return f"(!{args[0]})"
    if op == "where":
        return f"({args[0]} ? {args[1]} : {args[2]})"
    if op == "clip":
        return f"clamp({args[0]}, {args[1]}, {args[2]})"
    if op == "atan2":
        return f"atan({args[0]}, {args[1]})"
    if op == "pow" and node.args[1].op == "const" and node.args[1].args[0].is_integer():
        power = int(node.args[1].args[0])
        if power == 0:
            return "1.0"
        if 0 < power <= 16:
            return f"({' * '.join([args[0]] * power)})"
        if -16 <= power < 0:
            return f"(1.0 / ({' * '.join([args[0]] * -power)}))"
    if op in _EVAL:
        return f"{op}({', '.join(args)})"
    raise ValueError(f"unsupported scalar operation: {op}")


def to_glsl(expression) -> str:
    """Lower a small scalar field into an inline GLSL expression."""
    expression = as_scalar(expression)
    cache = {}

    def visit(node):
        key = id(node)
        if key not in cache:
            if node.op == "const":
                cache[key] = _literal(node.args[0])
            elif node.op == "var":
                cache[key] = node.args[0]
            else:
                cache[key] = _glsl_op(node, [visit(arg) for arg in node.args])
        return cache[key]

    return visit(expression)


def compile_glsl(expression) -> tuple[list[str], str]:
    """Lower a field into shared GLSL declarations and its result name."""
    expression = as_scalar(expression)
    lines = []
    cache = {}

    def visit(node):
        key = id(node)
        if key not in cache:
            if node.op == "const":
                cache[key] = _literal(node.args[0])
            elif node.op == "var":
                cache[key] = node.args[0]
            else:
                args = [visit(arg) for arg in node.args]
                name = f"f{len(lines)}"
                typ = "bool" if node.kind == "bool" else "float"
                lines.append(f"{typ} {name} = {_glsl_op(node, args)};")
                cache[key] = name
        return cache[key]

    return lines, visit(expression)
