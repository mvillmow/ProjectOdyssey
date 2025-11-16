#!/usr/bin/env bash
#
# Generate a test file from template
#
# Usage:
#   ./generate_test.sh <component-name> <test-type>
#
# Test types: unit, integration, performance
#
# Example:
#   ./generate_test.sh "tensor_ops" "unit"

set -euo pipefail

COMPONENT_NAME="${1:-}"
TEST_TYPE="${2:-unit}"

if [[ -z "$COMPONENT_NAME" ]]; then
    echo "Error: Component name required"
    echo "Usage: $0 <component-name> <test-type>"
    echo "Test types: unit, integration, performance"
    exit 1
fi

# Validate test type
case "$TEST_TYPE" in
    unit|integration|performance)
        ;;
    *)
        echo "Error: Invalid test type: $TEST_TYPE"
        echo "Valid types: unit, integration, performance"
        exit 1
        ;;
esac

# Sanitize component name
SAFE_NAME=$(echo "$COMPONENT_NAME" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

# Determine output directory
TEST_DIR="tests/$TEST_TYPE"
mkdir -p "$TEST_DIR"

# Determine file extension (prefer Mojo for ML code)
if [[ "$TEST_TYPE" == "unit" ]] || [[ "$TEST_TYPE" == "performance" ]]; then
    EXT="mojo"
    TEMPLATE="templates/unit_test_mojo.mojo"
else
    EXT="py"
    TEMPLATE="templates/integration_test.py"
fi

OUTPUT_FILE="$TEST_DIR/test_${SAFE_NAME}.$EXT"

# Check if file already exists
if [[ -f "$OUTPUT_FILE" ]]; then
    echo "Warning: File already exists: $OUTPUT_FILE"
    read -p "Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 1
    fi
fi

# Copy template
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_PATH="$(dirname "$SCRIPT_DIR")/$TEMPLATE"

if [[ -f "$TEMPLATE_PATH" ]]; then
    cp "$TEMPLATE_PATH" "$OUTPUT_FILE"
    # Replace placeholder with component name
    sed -i "s/ComponentName/${COMPONENT_NAME}/g" "$OUTPUT_FILE"
    sed -i "s/component_name/${SAFE_NAME}/g" "$OUTPUT_FILE"
else
    # Create basic template if template file doesn't exist
    if [[ "$EXT" == "mojo" ]]; then
        cat > "$OUTPUT_FILE" <<'EOF'
from testing import assert_equal, assert_true, assert_false

fn test_component_name_basic() raises:
    """Test basic functionality of component_name."""
    # Arrange
    # TODO: Setup test data

    # Act
    # TODO: Call function under test

    # Assert
    # TODO: Verify results
    pass

fn test_component_name_edge_case() raises:
    """Test edge case for component_name."""
    # TODO: Implement edge case test
    pass
EOF
    else
        cat > "$OUTPUT_FILE" <<'EOF'
import pytest

class TestComponentName:
    """Test suite for ComponentName."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        # TODO: Setup test data

        # Act
        # TODO: Call function under test

        # Assert
        # TODO: Verify results
        pass

    def test_edge_case(self):
        """Test edge case."""
        # TODO: Implement edge case test
        pass
EOF
    fi
    sed -i "s/ComponentName/${COMPONENT_NAME}/g" "$OUTPUT_FILE"
    sed -i "s/component_name/${SAFE_NAME}/g" "$OUTPUT_FILE"
fi

echo "✅ Test file created: $OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "1. Edit test file to add specific test cases"
echo "2. Run tests: mojo test $OUTPUT_FILE (or pytest $OUTPUT_FILE)"
echo "3. Implement code to make tests pass"
