#!/usr/bin/env python3
"""Generate torch::experimental::Tensor methods from gen-namespace.cpp.

The R torchgen pipeline already resolves the PyTorch schema into concrete C++
overloads and Lantern symbol names. Reusing that output keeps this facade in
lockstep with the Tensor API without introducing a second schema parser.
"""

from pathlib import Path
import argparse
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "gen-namespace.cpp"
OUTPUT = ROOT / "inst" / "include" / "torch" / "experimental_tensor_methods.h"
NAMESPACE_OUTPUT = ROOT / "inst" / "include" / "torch" / "experimental_namespace_functions.h"
FACADE = ROOT / "inst" / "include" / "torch" / "experimental.h"
LANTERN = ROOT / "inst" / "include" / "lantern" / "lantern.h"

SKIP = {
    "add_self_Tensor_other_Tensor",
    "detach_self_Tensor",
    "div_self_Tensor_other_Tensor",
    "matmul_self_Tensor_other_Tensor",
    "mul_self_Tensor_other_Tensor",
    "neg_self_Tensor",
    "relu_self_Tensor",
    "relu__self_Tensor",
    "requires_grad__self_Tensor",
    "sigmoid_self_Tensor",
    "sub_self_Tensor_other_Tensor",
    "t_self_Tensor",
    "transpose_self_Tensor_dim0_int64_t_dim1_int64_t",
    "zero__self_Tensor",
}

RETURN_TYPES = {
    "XPtrTorchTensor": "Tensor",
    "XPtrTorchTensorList": "::torch::TensorList",
    "XPtrTorchbool": "bool",
    "XPtrTorchint64_t": "std::int64_t",
    "XPtrTorchdouble": "double",
    "XPtrTorchScalar": "::torch::Scalar",
    "Rcpp::XPtr<XPtrTorchQScheme>": "Rcpp::XPtr<XPtrTorchQScheme>",
    "Rcpp::List": "Rcpp::List",
    "void": "void",
    "XPtrTorchScalarType": "::torch::ScalarType",
}

SUBNAMESPACES = {"fft", "linalg", "special"}


def split_parameters(signature):
    result, start, depth = [], 0, 0
    for index, character in enumerate(signature):
        if character in "<({[":
            depth += 1
        elif character in ">)}]":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(signature[start:index].strip())
            start = index + 1
    tail = signature[start:].strip()
    if tail:
        result.append(tail)
    return result


def public_parameter(parameter):
    match = re.match(r"(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)$", parameter)
    if not match:
        raise ValueError(f"Cannot parse parameter: {parameter}")
    type_name, name = match.groups()
    if type_name in {"XPtrTorchTensor", "XPtrTorchIndexTensor"}:
        type_name = "const Tensor&"
    elif type_name in {"XPtrTorchint64_t", "XPtrTorchindex_int64_t"}:
        type_name = "std::int64_t"
    elif type_name == "XPtrTorchbool":
        type_name = "bool"
    elif type_name == "XPtrTorchdouble":
        type_name = "double"
    return f"{type_name} {name}"


def adapt_parameter(body, parameter):
    match = re.match(r"(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)$", parameter)
    type_name, name = match.groups()
    adapters = {
        "XPtrTorchint64_t": "integer",
        "XPtrTorchindex_int64_t": "integer",
        "XPtrTorchbool": "boolean",
        "XPtrTorchdouble": "floating",
    }
    if type_name in adapters:
        body = re.sub(
            rf"\b{re.escape(name)}\.get\(\)",
            f"{adapters[type_name]}({name}).get()",
            body,
        )
    return body


def transform_body(body):
    body = re.sub(
        r"lantern_Tensor_([A-Za-z0-9_]+)\(",
        lambda match: (
            'detail::call<void*>("_lantern_Tensor_'
            + match.group(1)
            + '", '
        ),
        body,
    )
    body = body.replace(
        "lantern_vector_get(",
        'detail::call<void*>("_lantern_vector_get", ',
    )
    body = body.replace(
        "return XPtrTorchTensor(r_out);",
        "return Tensor(::torch::Tensor(r_out));",
    )
    body = body.replace(
        "return XPtrTorchbool(r_out);",
        "auto value = ::torch::bool_t(r_out);\n"
        'return detail::call<bool>("_lantern_bool_get", value.get());',
    )
    body = body.replace(
        "return XPtrTorchint64_t(r_out);",
        "auto value = ::torch::int64_t(r_out);\n"
        'return detail::call<std::int64_t>("_lantern_int64_t_get", value.get());',
    )
    body = body.replace(
        "return XPtrTorchdouble(r_out);",
        "auto value = ::torch::double_t(r_out);\n"
        'return detail::call<double>("_lantern_double_get", value.get());',
    )
    return re.sub(r"\bself\.get\(\)", "get()", body)


def functions(source, kind="method"):
    lines = source.splitlines()
    header = re.compile(
        rf"^(.+?)\s+(cpp_torch_{kind}_([^ (]+))\s*\((.*)\)\s*\{{$"
    )
    index = 0
    while index < len(lines):
        match = header.match(lines[index])
        if not match:
            index += 1
            continue
        return_type, _, encoded_name, signature = match.groups()
        body = []
        index += 1
        while index < len(lines) and lines[index] != "}":
            body.append(lines[index])
            index += 1
        yield return_type.strip(), encoded_name, signature, "\n".join(body)
        index += 1


def generated_source():
    methods = []
    seen = set()
    for return_type, encoded_name, signature, body in functions(SOURCE.read_text()):
        method_name = encoded_name.split("_self_", 1)[0]
        if encoded_name in SKIP:
            continue
        if return_type not in RETURN_TYPES:
            raise ValueError(f"Unhandled return type: {return_type}")

        parameters = split_parameters(signature)
        self_parameters = [
            index for index, value in enumerate(parameters) if value.endswith(" self")
        ]
        if len(self_parameters) != 1:
            raise ValueError(f"Tensor method has no self argument: {encoded_name}")
        del parameters[self_parameters[0]]
        public_parameters = [public_parameter(value) for value in parameters]
        key = (method_name, tuple(public_parameters))
        if key in seen:
            continue
        seen.add(key)

        transformed = transform_body(body)
        for parameter in parameters:
            transformed = adapt_parameter(transformed, parameter)
        indented = "\n".join(f"  {line}" for line in transformed.splitlines())
        qualifier = "" if method_name.endswith("_") else " const"
        methods.append(
            f"{RETURN_TYPES[return_type]} {method_name}"
            f"({', '.join(public_parameters)}){qualifier} {{\n{indented}\n}}"
        )

    preamble = [
        "// Generated by tools/generate-experimental-tensor.py; do not edit.",
        f"// Generated methods: {len(methods)}",
        "",
    ]
    return "\n\n".join(preamble + methods) + "\n", len(methods)


def namespace_public_parameter(parameter):
    value = public_parameter(parameter)
    return value.replace("const Tensor&", "const experimental::Tensor&")


def namespace_name(encoded_name, parameters):
    name = encoded_name.removeprefix("namespace_")
    if parameters:
        parameter_name = re.match(
            r"(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)$", parameters[0]
        ).group(2)
        marker = f"_{parameter_name}_"
        if marker not in name:
            raise ValueError(f"Cannot find first parameter in {encoded_name}")
        name = name.rsplit(marker, 1)[0]
    namespace = None
    prefix = name.split("_", 1)[0]
    if prefix in SUBNAMESPACES:
        namespace = prefix
        name = name[len(prefix) + 1:]
    return namespace, name


def transform_namespace_body(body):
    body = body.replace(
        "lantern_vector_get(",
        'experimental::detail::call<void*>("_lantern_vector_get", ',
    )
    body = re.sub(
        r"auto r_out = lantern_([A-Za-z0-9_]+)\(",
        lambda match: (
            'auto r_out = experimental::detail::call<void*>("_lantern_'
            + match.group(1) + '", '
        ),
        body,
    )
    body = re.sub(
        r"lantern_([A-Za-z0-9_]+)\(",
        lambda match: (
            'experimental::detail::call_void("_lantern_'
            + match.group(1) + '", '
        ),
        body,
    )
    body = body.replace(
        "return XPtrTorchTensor(r_out);",
        "return experimental::Tensor(::torch::Tensor(r_out));",
    )
    body = body.replace(
        "return XPtrTorchbool(r_out);",
        "auto value = ::torch::bool_t(r_out);\n"
        'return experimental::detail::call<bool>("_lantern_bool_get", value.get());',
    )
    body = body.replace(
        "return XPtrTorchint64_t(r_out);",
        "auto converted_value = ::torch::int64_t(r_out);\n"
        'return experimental::detail::call<std::int64_t>("_lantern_int64_t_get", converted_value.get());',
    )
    body = body.replace(
        "return XPtrTorchdouble(r_out);",
        "auto value = ::torch::double_t(r_out);\n"
        'return experimental::detail::call<double>("_lantern_double_get", value.get());',
    )
    return body


def generated_namespace_source():
    groups = {None: [], "fft": [], "linalg": [], "special": []}
    seen = set()
    for return_type, encoded_name, signature, body in functions(
            SOURCE.read_text(), "namespace"):
        if return_type not in RETURN_TYPES:
            raise ValueError(f"Unhandled namespace return type: {return_type}")
        parameters = split_parameters(signature)
        namespace, function_name = namespace_name(encoded_name, parameters)
        # torch_types.h already exposes torch::index as a namespace.
        if namespace is None and function_name == "index":
            continue
        public_parameters = [namespace_public_parameter(x) for x in parameters]
        parameter_types = tuple(
            re.match(r"(.+?)\s+[A-Za-z_][A-Za-z0-9_]*$", x).group(1)
            for x in public_parameters
        )
        key = (namespace, function_name, parameter_types)
        if key in seen:
            raise ValueError(f"Duplicate public namespace overload: {key}")
        seen.add(key)
        transformed = transform_namespace_body(body)
        for parameter in parameters:
            transformed = adapt_parameter(transformed, parameter)
        for adapter in ("integer", "boolean", "floating"):
            transformed = re.sub(
                rf"(?<!::)\b{adapter}\(",
                f"experimental::detail::{adapter}(",
                transformed,
            )
        indented = "\n".join(f"  {line}" for line in transformed.splitlines())
        public_return = RETURN_TYPES[return_type]
        if public_return == "Tensor":
            public_return = "experimental::Tensor"
        groups[namespace].append(
            f"inline {public_return} {function_name}"
            f"({', '.join(public_parameters)}) {{\n{indented}\n}}"
        )

    lines = [
        "// Generated by tools/generate-experimental-tensor.py; do not edit.",
        f"// Generated namespace functions: {len(seen)}",
        "",
    ]
    lines.extend(groups[None])
    for namespace in ("fft", "linalg", "special"):
        lines.extend([f"namespace {namespace} {{", ""])
        lines.extend(groups[namespace])
        lines.extend(["", f"}}  // namespace {namespace}"])
    return "\n\n".join(lines) + "\n", len(seen)


def generate(check=False):
    source, count = generated_source()
    namespace_source, namespace_count = generated_namespace_source()
    if check:
        if (not OUTPUT.exists() or OUTPUT.read_text() != source or
                not NAMESPACE_OUTPUT.exists() or
                NAMESPACE_OUTPUT.read_text() != namespace_source):
            raise SystemExit(
                f"{OUTPUT} is stale; run tools/generate-experimental-tensor.py"
            )
        lantern_methods = set(re.findall(
            r"HOST_API\s+[^\n]+\s+lantern_Tensor_([A-Za-z0-9_]+)\s*\(",
            LANTERN.read_text(),
        ))
        covered_methods = set(re.findall(
            r'"_lantern_Tensor_([A-Za-z0-9_]+)"',
            FACADE.read_text() + source,
        ))
        # delete is an ownership implementation detail. set_requires_grad is
        # the legacy predecessor of the covered requires_grad_ schema method.
        allowed = {"delete", "set_requires_grad"}
        missing = sorted(lantern_methods - covered_methods - allowed)
        if missing:
            raise SystemExit("Uncovered Lantern Tensor methods: " + ", ".join(missing))
        print(
            f"Verified {count} generated overloads and "
            f"{namespace_count} namespace overloads; "
            f"{len(lantern_methods - allowed)} Lantern Tensor entry points"
        )
        return
    OUTPUT.write_text(source)
    NAMESPACE_OUTPUT.write_text(namespace_source)
    print(f"Generated {count} Tensor method overloads in {OUTPUT}")
    print(f"Generated {namespace_count} namespace overloads in {NAMESPACE_OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    generate(check=parser.parse_args().check)
