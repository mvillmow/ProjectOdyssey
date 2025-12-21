#!/usr/bin/env python3
"""Fix kaiming_normal and xavier_normal signature issues."""

import re
from pathlib import Path


def fix_initializers(filepath):
    """Fix kaiming/xavier normal calls and mut self constructors."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    modified = False

    for i, line in enumerate(lines):
        # Fix mut self → out self in constructors
        if "fn __init__(" in line and i + 1 < len(lines):
            if "mut self" in lines[i + 1]:
                lines[i + 1] = lines[i + 1].replace("mut self", "out self")
                modified = True

        # Fix kaiming_normal calls:
        # Current bad pattern: kaiming_normal([shape], fan_in=value)
        # Should be: kaiming_normal(fan_in=value, fan_out=value, shape=[shape])
        if "kaiming_normal(" in line:
            # Look ahead to collect the full call (may span multiple lines)
            call_lines = [line]
            j = i + 1
            while j < len(lines) and ")" not in "".join(call_lines):
                call_lines.append(lines[j])
                j += 1

            call_text = "".join(call_lines)

            # Pattern: kaiming_normal(\n?      [shape],\n?      fan_in=value,\n?    )
            # Extract shape and fan_in
            shape_match = re.search(
                r"kaiming_normal\(\s*(\[[^\]]+\])\s*,\s*fan_in\s*=\s*([^,\)]+)",
                call_text,
                re.MULTILINE | re.DOTALL,
            )

            if shape_match:
                shape = shape_match.group(1).strip()
                fan_in_expr = shape_match.group(2).strip()

                # Parse shape to calculate fan_out
                # Shape format: [out_channels, in_channels, kernel_h, kernel_w]
                shape_clean = shape.replace("[", "").replace("]", "")
                shape_parts = [p.strip() for p in shape_clean.split(",")]

                if len(shape_parts) == 4:
                    out_ch, in_ch, k_h, k_w = shape_parts
                    fan_out_expr = f"{out_ch} * {k_h} * {k_w}" if k_h != "1" and k_w != "1" else out_ch

                    # Reconstruct the call
                    indent = " " * (len(line) - len(line.lstrip()))
                    new_call = f"{indent}kaiming_normal(\n"
                    new_call += f"{indent}    fan_in={fan_in_expr},\n"
                    new_call += f"{indent}    fan_out={fan_out_expr},\n"
                    new_call += f"{indent}    shape={shape},\n"
                    new_call += f"{indent})\n"

                    # Replace the original call
                    for k in range(i, j):
                        lines[k] = ""
                    lines[i] = new_call
                    modified = True

        # Fix xavier_normal calls similarly
        if "xavier_normal(" in line:
            call_lines = [line]
            j = i + 1
            while j < len(lines) and ")" not in "".join(call_lines):
                call_lines.append(lines[j])
                j += 1

            call_text = "".join(call_lines)

            # Pattern: xavier_normal([shape], fan_in=value, fan_out=value)
            match = re.search(
                r"xavier_normal\(\s*(\[[^\]]+\])\s*,\s*fan_in\s*=\s*([^,\)]+)\s*,\s*fan_out\s*=\s*([^,\)]+)",
                call_text,
                re.MULTILINE | re.DOTALL,
            )

            if match:
                shape = match.group(1).strip()
                fan_in_expr = match.group(2).strip()
                fan_out_expr = match.group(3).strip()

                indent = " " * (len(line) - len(line.lstrip()))
                new_call = f"{indent}xavier_normal(\n"
                new_call += f"{indent}    fan_in={fan_in_expr},\n"
                new_call += f"{indent}    fan_out={fan_out_expr},\n"
                new_call += f"{indent}    shape={shape},\n"
                new_call += f"{indent})\n"

                for k in range(i, j):
                    lines[k] = ""
                lines[i] = new_call
                modified = True

    if modified:
        with open(filepath, "w") as f:
            f.writelines(lines)

    return modified


def main():
    files = [
        "examples/googlenet-cifar10/model.mojo",
        "examples/mobilenetv1-cifar10/model.mojo",
    ]

    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️  File not found: {filepath}")
            continue

        print(f"Processing: {filepath}")
        if fix_initializers(filepath):
            print(f"✓ Fixed initializer signatures in {filepath}")
        else:
            print(f"  No changes needed in {filepath}")

    print("\n✓ All initializer signatures fixed!")


if __name__ == "__main__":
    main()
