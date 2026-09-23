"""Translate bounded, pointwise JAX geometry to GLSL ES 3.00."""

from __future__ import annotations

from itertools import product
from math import isfinite

import jax
import jax.numpy as jnp
import numpy as np

from .dsl import Model, Shape

_MAX_ELEMENTS = 4096
_MAX_STATEMENTS = 60000
_MAX_SCAN = 128


def _type(dtype):
    kind = np.dtype(dtype).kind
    if kind == "b":
        return "bool"
    if kind in "iu":
        if np.dtype(dtype).itemsize != 4:
            raise ValueError("GLSL viewer supports only 32-bit integer geometry")
        return "uint" if kind == "u" else "int"
    if kind == "f" and np.dtype(dtype).itemsize == 4:
        return "float"
    raise ValueError(f"GLSL viewer does not support geometry dtype {dtype}; use float32")


def _literal(value, dtype):
    typ = _type(dtype)
    if typ == "bool":
        return "true" if value else "false"
    if typ in ("int", "uint"):
        value = int(value)
        if typ == "int" and value == -2147483648:
            return "(-2147483647 - 1)"
        return str(value) + ("u" if typ == "uint" else "")
    value = float(value)
    if not isfinite(value):
        bits = np.asarray(value, dtype=np.float32).view(np.uint32).item()
        return f"uintBitsToFloat({bits}u)"
    result = repr(value)
    return result if "." in result or "e" in result else result + ".0"


class _Compiler:
    def __init__(self):
        self.lines = []
        self.cache = {}

    def atom(self, value, dtype):
        return value if isinstance(value, str) else _literal(value, dtype)

    def emit(self, expression, dtype):
        typ = _type(dtype)
        key = (typ, expression)
        if key in self.cache:
            return self.cache[key]
        if len(self.lines) >= _MAX_STATEMENTS:
            raise ValueError("GLSL scene is too large; simplify geometry or use the JAX viewer")
        name = f"v{len(self.lines)}"
        self.lines.append(f"  {typ} {name} = {expression};")
        self.cache[key] = name
        return name

    def array(self, value):
        result = np.asarray(value, dtype=object)
        if result.size > _MAX_ELEMENTS:
            raise ValueError(f"GLSL geometry arrays are limited to {_MAX_ELEMENTS} elements")
        return result

    def run(self, closed, inputs):
        graph = closed.jaxpr if hasattr(closed, "jaxpr") else closed
        constants = closed.consts if hasattr(closed, "consts") else ()
        env = dict(zip(graph.constvars, map(self.array, constants), strict=True))
        env.update(zip(graph.invars, inputs, strict=True))

        def read(var):
            return self.array(var.val) if hasattr(var, "val") else env[var]

        for equation in graph.eqns:
            values = [read(var) for var in equation.invars]
            for var in equation.outvars:
                if np.prod(var.aval.shape, dtype=int) > _MAX_ELEMENTS:
                    raise ValueError("GLSL geometry array exceeds the supported size")
                _type(var.aval.dtype)
            outputs = self.equation(equation, values)
            env.update(zip(equation.outvars, outputs, strict=True))
        return [read(var) for var in graph.outvars]

    def elementwise(self, equation, values):
        name, params = equation.primitive.name, equation.params
        output = equation.outvars[0].aval
        arrays = [np.broadcast_to(value, output.shape) for value in values]
        result = np.empty(output.shape, dtype=object)
        input_types = [v.aval.dtype for v in equation.invars]
        for index in np.ndindex(output.shape):
            a = [
                self.atom(value[index], dtype)
                for value, dtype in zip(arrays, input_types, strict=True)
            ]
            if name in ("add", "sub", "mul", "div", "lt", "le", "gt", "ge", "eq", "ne"):
                operator = dict(
                    add="+",
                    sub="-",
                    mul="*",
                    div="/",
                    lt="<",
                    le="<=",
                    gt=">",
                    ge=">=",
                    eq="==",
                    ne="!=",
                )[name]
                expr = f"({a[0]} {operator} {a[1]})"
            elif name in ("min", "max", "pow", "atan2"):
                expr = f"{'atan' if name == 'atan2' else name}({a[0]}, {a[1]})"
            elif name in (
                "abs",
                "sqrt",
                "exp",
                "exp2",
                "log",
                "log2",
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "sinh",
                "cosh",
                "tanh",
                "floor",
                "ceil",
                "sign",
            ):
                expr = f"{name}({a[0]})"
            elif name in ("neg", "not"):
                operator = "-" if name == "neg" else ("!" if _type(output.dtype) == "bool" else "~")
                expr = f"({operator}{a[0]})"
            elif name in ("and", "or", "xor"):
                boolean = _type(output.dtype) == "bool"
                operator = {
                    "and": "&&" if boolean else "&",
                    "or": "||" if boolean else "|",
                    "xor": "!=" if boolean else "^",
                }[name]
                expr = f"({a[0]} {operator} {a[1]})"
            elif name == "select_n":
                if len(a) != 3:
                    raise ValueError("GLSL viewer supports two-way jnp.where selections")
                condition = (
                    a[0]
                    if _type(input_types[0]) == "bool"
                    else f"({a[0]} != {_literal(0, input_types[0])})"
                )
                expr = f"({condition} ? {a[2]} : {a[1]})"
            elif name in ("square", "integer_pow"):
                power = 2 if name == "square" else params["y"]
                if abs(power) > 16:
                    raise ValueError("GLSL integer powers are limited to magnitude 16")
                base = " * ".join([a[0]] * abs(power)) or _literal(1, output.dtype)
                expr = f"({_literal(1, output.dtype)} / ({base}))" if power < 0 else f"({base})"
            elif name == "convert_element_type":
                if input_types[0] == output.dtype:
                    result[index] = arrays[0][index]
                    continue
                expr = f"{_type(output.dtype)}({a[0]})"
            elif name == "rsqrt":
                expr = f"inversesqrt({a[0]})"
            elif name == "is_finite":
                expr = f"(!isnan({a[0]}) && !isinf({a[0]}))"
            elif name in ("lt_to", "le_to"):
                op = "<" if name == "lt_to" else "<="
                if _type(input_types[0]) == "float":
                    keys = []
                    for term in a:
                        bits = self.emit(f"floatBitsToUint({term})", np.uint32)
                        keys.append(
                            self.emit(
                                f"(({bits} & 2147483648u) != 0u ? ~{bits} : ({bits} ^ 2147483648u))",
                                np.uint32,
                            )
                        )
                    expr = f"({keys[0]} {op} {keys[1]})"
                else:
                    expr = f"({a[0]} {op} {a[1]})"
            elif name == "rem":
                if _type(output.dtype) == "float":
                    expr = f"({a[0]} - trunc({a[0]} / {a[1]}) * {a[1]})"
                else:
                    expr = f"({a[0]} % {a[1]})"
            elif name == "clamp":
                expr = f"clamp({a[1]}, {a[0]}, {a[2]})"
            else:
                raise ValueError(
                    f"GLSL viewer cannot translate JAX operation '{name}'; simplify this field or use the JAX viewer"
                )
            result[index] = self.emit(expr, output.dtype)
        return [result]

    def equation(self, equation, values):
        name, p = equation.primitive.name, equation.params
        if equation.effects:
            raise ValueError("GLSL geometry cannot contain callbacks or side effects")
        if name in ("jit", "pjit", "closed_call", "custom_jvp_call"):
            return self.run(p.get("jaxpr", p.get("call_jaxpr")), values)
        if name == "stop_gradient":
            return values
        # Evaluate input-independent work once in Python, preserving JAX dtypes.
        if all(not any(isinstance(item, str) for item in value.flat) for value in values):
            inputs = [
                jnp.asarray(value.tolist(), dtype=var.aval.dtype)
                for value, var in zip(values, equation.invars, strict=True)
            ]
            try:
                result = equation.primitive.bind(*inputs, **p)
                if not equation.primitive.multiple_results:
                    result = [result]
                return [self.array(np.asarray(item)) for item in result]
            except NotImplementedError, TypeError:
                pass
        if name == "slice":
            strides = p["strides"] or (1,) * len(p["start_indices"])
            slices = tuple(
                slice(a, b, c)
                for a, b, c in zip(p["start_indices"], p["limit_indices"], strides, strict=True)
            )
            return [values[0][slices]]
        if name in ("squeeze", "reshape"):
            value = values[0]
            if name == "reshape" and p.get("dimensions") is not None:
                value = value.transpose(p["dimensions"])
            return [value.reshape(equation.outvars[0].aval.shape)]
        if name == "transpose":
            return [values[0].transpose(p["permutation"])]
        if name == "broadcast_in_dim":
            shape = [1] * len(p["shape"])
            for axis, size in zip(p["broadcast_dimensions"], values[0].shape, strict=True):
                shape[axis] = size
            return [np.broadcast_to(values[0].reshape(shape), p["shape"])]
        if name == "stack":
            return [np.stack(values, axis=p["axis"])]
        if name in ("argmin", "argmax"):
            return [self.argreduce(equation, values[0])]
        if name == "concatenate":
            return [np.concatenate(values, axis=p["dimension"])]
        if name.startswith("reduce_"):
            return [self.reduce(equation, values[0])]
        if name == "dynamic_slice":
            return [self.dynamic_slice(equation, values)]
        if name == "dot_general":
            return [self.dot(equation, values)]
        if name == "scan":
            count = p["length"]
            if count > _MAX_SCAN:
                raise ValueError(f"GLSL static loops are limited to {_MAX_SCAN} iterations")
            body = p["jaxpr"]
            graph = body.jaxpr if hasattr(body, "jaxpr") else body
            # New JAX records flattened loop state in ft_in instead of num_carry.
            if "num_consts" in p:
                nc, ns = p["num_consts"], p["num_carry"]
            elif "ft_in" in p:
                groups = p["ft_in"].unpack()
                nc, ns = len(groups[0]), len(groups[1])
            else:
                raise ValueError("Unsupported JAX scan layout in GLSL geometry")
            const, carry, xs = values[:nc], values[nc : nc + ns], values[nc + ns :]
            collected = [[] for _ in graph.outvars[ns:]]
            indices = range(count - 1, -1, -1) if p["reverse"] else range(count)
            for i in indices:
                outputs = self.run(body, const + carry + [value[i] for value in xs])
                carry = outputs[:ns]
                for bucket, value in zip(collected, outputs[ns:], strict=True):
                    bucket.append(value)
            return carry + [
                np.stack(bucket[::-1] if p["reverse"] else bucket) for bucket in collected
            ]
        return self.elementwise(equation, values)

    def argreduce(self, equation, value):
        axes = equation.params["axes"]
        if len(axes) != 1:
            raise ValueError("GLSL argmin/argmax require one axis")
        axis = axes[0]
        output = equation.outvars[0].aval
        dtype = equation.invars[0].aval.dtype
        result = np.empty(output.shape, dtype=object)
        for index in np.ndindex(output.shape):
            selection = list(index)
            selection.insert(axis, slice(None))
            terms = list(value[tuple(selection)].flat)
            best, best_index = self.atom(terms[0], dtype), "0"
            for i, term in enumerate(terms[1:], 1):
                term = self.atom(term, dtype)
                op = "<" if equation.primitive.name == "argmin" else ">"
                condition = f"({term} {op} {best})"
                if _type(dtype) == "float":
                    condition = f"({condition} || (isnan({term}) && !isnan({best})))"
                condition = self.emit(condition, np.bool_)
                best = self.emit(f"({condition} ? {term} : {best})", dtype)
                best_index = self.emit(f"({condition} ? {i} : {best_index})", output.dtype)
            result[index] = best_index
        return result

    def reduce(self, equation, value):
        name, axes = equation.primitive.name, equation.params["axes"]
        supported = {
            "reduce_sum": "+",
            "reduce_prod": "*",
            "reduce_min": "min",
            "reduce_max": "max",
            "reduce_or": "||",
            "reduce_and": "&&",
        }
        if name not in supported:
            raise ValueError(f"GLSL viewer cannot translate JAX reduction '{name}'")
        output = equation.outvars[0].aval
        result = np.empty(output.shape, dtype=object)
        kept = [axis for axis in range(value.ndim) if axis not in axes]
        dtype = equation.invars[0].aval.dtype
        for index in np.ndindex(output.shape):
            selection = [slice(None)] * value.ndim
            for axis, coordinate in zip(kept, index, strict=True):
                selection[axis] = coordinate
            terms = list(value[tuple(selection)].flat)
            if not terms:
                identity = {
                    "reduce_sum": 0,
                    "reduce_prod": 1,
                    "reduce_min": np.inf,
                    "reduce_max": -np.inf,
                    "reduce_or": False,
                    "reduce_and": True,
                }[name]
                result[index] = identity
                continue
            acc = self.atom(terms[0], dtype)
            operator = supported[name]
            for term in terms[1:]:
                term = self.atom(term, dtype)
                expr = (
                    f"{operator}({acc}, {term})"
                    if operator in ("min", "max")
                    else f"({acc} {operator} {term})"
                )
                acc = self.emit(expr, output.dtype)
            result[index] = acc
        return result

    def dynamic_slice(self, equation, values):
        source, *starts = values
        sizes = equation.params["slice_sizes"]
        output = equation.outvars[0].aval
        starts = [
            self.atom(value.item(), var.aval.dtype)
            for value, var in zip(starts, equation.invars[1:], strict=True)
        ]
        starts = [
            self.emit(
                f"int(clamp({start}, {_literal(0, var.aval.dtype)}, {_literal(size - width, var.aval.dtype)}))",
                np.int32,
            )
            for start, size, width, var in zip(
                starts, source.shape, sizes, equation.invars[1:], strict=True
            )
        ]
        result = np.empty(sizes, dtype=object)
        for index in np.ndindex(sizes):
            choices = list(
                product(
                    *(
                        range(size - width + 1)
                        for size, width in zip(source.shape, sizes, strict=True)
                    )
                )
            )
            expr = self.atom(
                source[tuple(a + b for a, b in zip(choices[-1], index, strict=True))], output.dtype
            )
            for choice in reversed(choices[:-1]):
                condition = " && ".join(
                    f"{start} == {coordinate}"
                    for start, coordinate in zip(starts, choice, strict=True)
                )
                item = source[tuple(a + b for a, b in zip(choice, index, strict=True))]
                expr = self.emit(
                    f"({condition} ? {self.atom(item, output.dtype)} : {expr})", output.dtype
                )
            result[index] = expr
        return result

    def dot(self, equation, values):
        (la, ra), (lb, rb) = equation.params["dimension_numbers"]
        if lb or rb:
            raise ValueError("GLSL viewer does not support batched dot products inside fields")
        left, right = values
        lf = [i for i in range(left.ndim) if i not in la]
        rf = [i for i in range(right.ndim) if i not in ra]
        output = equation.outvars[0].aval
        result = np.empty(output.shape, dtype=object)
        for index in np.ndindex(output.shape):
            li, ri = [0] * left.ndim, [0] * right.ndim
            for axis, coordinate in zip(lf, index[: len(lf)], strict=True):
                li[axis] = coordinate
            for axis, coordinate in zip(rf, index[len(lf) :], strict=True):
                ri[axis] = coordinate
            acc = _literal(0, output.dtype)
            for contract in product(*(range(left.shape[axis]) for axis in la)):
                for axis, coordinate in zip(la, contract, strict=True):
                    li[axis] = coordinate
                for axis, coordinate in zip(ra, contract, strict=True):
                    ri[axis] = coordinate
                a = self.atom(left[tuple(li)], equation.invars[0].aval.dtype)
                b = self.atom(right[tuple(ri)], equation.invars[1].aval.dtype)
                acc = self.emit(f"({acc} + {a} * {b})", output.dtype)
            result[index] = acc
        return result


def compile_glsl(scene):
    """Return a shader snippet defining vec2 mapScene(vec3 p): distance, material.

    Geometry is traced at one float32 point and translated without sampling.
    Unsupported JAX primitives and oversized static programs raise ValueError.
    """
    if isinstance(scene, Model):
        scene = scene.scene()
    if not isinstance(scene, Shape) or scene.dim != 3:
        raise ValueError("GLSL scene must be a 3D Shape or Model")
    try:
        graph = jax.make_jaxpr(scene.sample)(jnp.zeros(3, dtype=jnp.float32))
        compiler = _Compiler()
        outputs = compiler.run(graph, [np.asarray(["p.x", "p.y", "p.z"], dtype=object)])
        if len(outputs) != 2 or any(value.shape for value in outputs):
            raise ValueError("GLSL geometry must return one distance and one material per point")
        distance, material = [
            compiler.atom(value.item(), var.aval.dtype)
            for value, var in zip(outputs, graph.jaxpr.outvars, strict=True)
        ]
        return "\n".join(
            [
                "vec2 mapScene(vec3 p) {",
                *compiler.lines,
                f"  return vec2(float({distance}), float({material}));",
                "}",
            ]
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"GLSL geometry translation failed: {exc}. Use the JAX viewer for unsupported fields."
        ) from exc
