#!/usr/bin/env python3

"""
Tool: codegen/mojo_boilerplate.py
Purpose: Generate Mojo boilerplate code (structs, layers, basic patterns)

Language: Python
Justification:
  - Template processing with string substitution
  - String manipulation for code generation
  - No performance requirements (one-time generation)
  - No ML/AI computation involved

Reference: ADR-001
Last Review: 2025-11-16
"""

from typing import List, Tuple, Dict
import argparse
import sys


def generate_struct(name: str, fields: List[Tuple[str, str]], include_init: bool = True) -> str:
    """
    Generate a basic Mojo struct definition.

    Args:
        name: Struct name
        fields: List of (field_name, field_type) tuples
        include_init: Whether to include __init__ method

    Returns:
        Generated Mojo struct code
    """
    lines = [f"struct {name}:"]

    # Add fields
    for field_name, field_type in fields:
        lines.append(f"    var {field_name}: {field_type}")

    if include_init and fields:
        lines.append("")
        # Generate __init__
        params = ", ".join([f"{name}: {type_}" for name, type_ in fields])
        lines.append(f"    fn __init__(inout self, {params}):")
        for field_name, _ in fields:
            lines.append(f"        self.{field_name} = {field_name}")

    return "\n".join(lines)


def generate_layer(layer_type: str, params: Dict[str, str]) -> str:
    """
    Generate a basic Mojo neural network layer.

    Args:
        layer_type: Type of layer (Linear, Conv2D, etc.)
        params: Layer parameters as dict

    Returns:
        Generated Mojo layer code
    """
    if layer_type == "Linear":
        return _generate_linear_layer(params)
    elif layer_type == "Conv2D":
        return _generate_conv2d_layer(params)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def _generate_linear_layer(params: Dict[str, str]) -> str:
    """Generate a Linear (fully connected) layer."""
    in_features = params.get("in_features", "in_features")
    out_features = params.get("out_features", "out_features")

    template = f"""struct LinearLayer:
    var in_features: Int
    var out_features: Int
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]

    fn __init__(inout self, in_features: Int, out_features: Int):
        self.in_features = in_features
        self.out_features = out_features
        # TODO: Initialize weights and bias
        self.weights = Tensor[DType.float32](in_features, out_features)
        self.bias = Tensor[DType.float32](out_features)

    fn forward(self, borrowed input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # TODO: Implement matrix multiplication
        # output = input @ weights + bias
        return input  # Placeholder
"""
    return template


def _generate_conv2d_layer(params: Dict[str, str]) -> str:
    """Generate a Conv2D layer."""
    in_channels = params.get("in_channels", "in_channels")
    out_channels = params.get("out_channels", "out_channels")
    kernel_size = params.get("kernel_size", "3")

    template = f"""struct Conv2DLayer:
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]

    fn __init__(inout self, in_channels: Int, out_channels: Int, kernel_size: Int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # TODO: Initialize weights and bias
        let weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weights = Tensor[DType.float32](out_channels, in_channels, kernel_size, kernel_size)
        self.bias = Tensor[DType.float32](out_channels)

    fn forward(self, borrowed input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # TODO: Implement convolution
        return input  # Placeholder
"""
    return template


def main() -> int:
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate Mojo boilerplate code for common patterns"
    )

    subparsers = parser.add_subparsers(dest="command", help="Generation command")

    # Struct generation
    struct_parser = subparsers.add_parser("struct", help="Generate a struct")
    struct_parser.add_argument("name", help="Struct name")
    struct_parser.add_argument(
        "--fields",
        nargs="+",
        help="Fields as name:type pairs (e.g., x:Int y:Float64)"
    )
    struct_parser.add_argument(
        "--no-init",
        action="store_true",
        help="Don't generate __init__ method"
    )

    # Layer generation
    layer_parser = subparsers.add_parser("layer", help="Generate a neural network layer")
    layer_parser.add_argument(
        "layer_type",
        choices=["Linear", "Conv2D"],
        help="Layer type"
    )
    layer_parser.add_argument(
        "--params",
        nargs="+",
        help="Layer parameters as key:value pairs"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "struct":
            # Parse fields
            fields = []
            if args.fields:
                for field_spec in args.fields:
                    if ":" not in field_spec:
                        print(f"Error: Invalid field spec '{field_spec}'. Use name:type format.")
                        return 1
                    name, type_ = field_spec.split(":", 1)
                    fields.append((name.strip(), type_.strip()))

            code = generate_struct(args.name, fields, include_init=not args.no_init)
            print(code)

        elif args.command == "layer":
            # Parse params
            params = {}
            if args.params:
                for param_spec in args.params:
                    if ":" not in param_spec:
                        print(f"Error: Invalid param spec '{param_spec}'. Use key:value format.")
                        return 1
                    key, value = param_spec.split(":", 1)
                    params[key.strip()] = value.strip()

            code = generate_layer(args.layer_type, params)
            print(code)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
