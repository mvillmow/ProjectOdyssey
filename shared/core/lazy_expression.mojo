"""Lazy evaluation system for tensor expressions.

Implements expression templates using flat array-based storage for deferred computation.
Enables fusion of multiple operations into single kernels without intermediate allocations.

Architecture:
    - TensorExpr: Lazy expression wrapper that defers computation
    - ExprNode: Single node in expression tree (operation, operands, references)
    - Flat array storage: Avoids recursive struct limitation in Mojo
    - On-demand evaluation via evaluate()

Example:
    ```mojo
    from shared.core.lazy_expression import expr, evaluate

    # Build expression lazily (no computation yet)
    var expression = (expr(tensor1) + expr(tensor2)) * expr(tensor3)

    # Single fused evaluation (1 allocation instead of 3)
    var result = evaluate(expression)
    ```
"""

from collections import List
from shared.core.extensor import ExTensor
from shared.core.broadcasting import broadcast_shapes, compute_broadcast_strides


# ============================================================================
# Operation Type Aliases
# ============================================================================

alias OpType = Int
alias OP_LEAF: OpType = 0  # Terminal tensor node
alias OP_ADD: OpType = 1  # a + b
alias OP_SUB: OpType = 2  # a - b
alias OP_MUL: OpType = 3  # a * b
alias OP_DIV: OpType = 4  # a / b
alias OP_NEG: OpType = 5  # -a (unary negation)
alias OP_SCALAR_MUL: OpType = 6  # a * scalar
alias OP_SCALAR_DIV: OpType = 7  # a / scalar


# ============================================================================
# Expression Node Structure
# ============================================================================


struct ExprNode(Copyable, ImplicitlyCopyable, Movable):
    """Single node in lazy expression tree.

    Uses flat array storage pattern to work around Mojo's recursive struct
    limitations. All references use indices into Lists rather than pointers.

    Attributes:
        op: Operation type (OP_LEAF, OP_ADD, etc.)
        left_idx: Index of left child node (-1 if not used)
        right_idx: Index of right child node (-1 if not used/unary)
        tensor_idx: Index in _tensors list (-1 if not a leaf)
        scalar_idx: Index in _scalars list (-1 if not used)
    """

    var op: OpType
    var left_idx: Int
    var right_idx: Int
    var tensor_idx: Int
    var scalar_idx: Int

    fn __init__(out self, op: OpType):
        """Create a new expression node.

        Args:
            op: Operation type for this node.
        """
        self.op = op
        self.left_idx = -1
        self.right_idx = -1
        self.tensor_idx = -1
        self.scalar_idx = -1


# ============================================================================
# Main Lazy Expression Type
# ============================================================================


struct TensorExpr(Copyable, Movable):
    """Lazy tensor expression with deferred evaluation.

    Builds expression trees lazily without computing any values. Supports
    operator overloading to chain operations. Expressions are evaluated
    on-demand via evaluate() to produce fused kernels.

    Memory Model:
        - Stores references to tensors (not copies)
        - Stores scalar values directly
        - Builds tree structure via flat node array
        - Pre-computes broadcast shape at expression build time

    Attributes:
        _nodes: Flat array of expression nodes
        _root_idx: Index of root node in _nodes
        _tensors: Referenced tensor objects
        _scalars: Referenced scalar values
        _result_shape: Pre-computed broadcast shape
        _dtype: Common dtype for all operations
    """

    var _nodes: List[ExprNode]
    var _root_idx: Int
    var _tensors: List[ExTensor]
    var _scalars: List[Float64]
    var _result_shape: List[Int]
    var _dtype: DType

    fn __init__(out self, tensor: ExTensor) raises:
        """Create expression from a single tensor (entry point).

        Args:
            tensor: Source tensor.

        Raises:
            Error: If tensor is invalid.
        """
        self._nodes = List[ExprNode]()
        self._tensors = List[ExTensor]()
        self._scalars = List[Float64]()

        # Create leaf node for tensor
        var leaf = ExprNode(OP_LEAF)
        leaf.tensor_idx = 0
        self._nodes.append(leaf)
        self._root_idx = 0

        # Store tensor reference
        self._tensors.append(tensor)

        # Copy shape and dtype
        self._result_shape = List[Int]()
        for i in range(len(tensor.shape())):
            self._result_shape.append(tensor.shape()[i])

        self._dtype = tensor.dtype()

    fn shape(self) -> List[Int]:
        """Get the shape of the expression result.

        Returns:
            Shape of the computed result.
        """
        var result = List[Int]()
        for i in range(len(self._result_shape)):
            result.append(self._result_shape[i])
        return result

    fn dtype(self) -> DType:
        """Get the data type of the expression result.

        Returns:
            DType of the computed result.
        """
        return self._dtype

    fn numel(self) -> Int:
        """Get total number of elements in result.

        Returns:
            Product of all shape dimensions.
        """
        var total = 1
        for i in range(len(self._result_shape)):
            total *= self._result_shape[i]
        return total

    fn num_ops(self) -> Int:
        """Count number of operations in expression tree.

        Returns:
            Number of non-leaf nodes.
        """
        var count = 0
        for i in range(len(self._nodes)):
            if self._nodes[i].op != OP_LEAF:
                count += 1
        return count

    fn _add_leaf_node(mut self, tensor: ExTensor) raises -> Int:
        """Internal: Add a leaf node for a tensor.

        Args:
            tensor: Tensor to add as leaf.

        Returns:
            Index of new node.

        Raises:
            Error: If tensor shapes are not broadcastable.
        """
        # Check broadcastability with result shape
        var new_shape = broadcast_shapes(self._result_shape, tensor.shape())

        var leaf = ExprNode(OP_LEAF)
        leaf.tensor_idx = len(self._tensors)
        var node_idx = len(self._nodes)
        self._nodes.append(leaf)

        self._tensors.append(tensor)
        self._result_shape = new_shape

        return node_idx

    fn _add_binary_op(
        mut self, op: OpType, left_idx: Int, right_idx: Int
    ) -> Int:
        """Internal: Add a binary operation node.

        Args:
            op: Operation type.
            left_idx: Index of left operand.
            right_idx: Index of right operand.

        Returns:
            Index of new node.
        """
        var node = ExprNode(op)
        node.left_idx = left_idx
        node.right_idx = right_idx
        var node_idx = len(self._nodes)
        self._nodes.append(node)
        return node_idx

    fn _add_unary_op(mut self, op: OpType, operand_idx: Int) -> Int:
        """Internal: Add a unary operation node.

        Args:
            op: Operation type.
            operand_idx: Index of operand.

        Returns:
            Index of new node.
        """
        var node = ExprNode(op)
        node.left_idx = operand_idx
        node.right_idx = -1
        var node_idx = len(self._nodes)
        self._nodes.append(node)
        return node_idx

    fn _add_scalar_op(
        mut self, op: OpType, tensor_idx: Int, scalar: Float64
    ) -> Int:
        """Internal: Add a scalar operation node.

        Args:
            op: Operation type (OP_SCALAR_MUL or OP_SCALAR_DIV).
            tensor_idx: Index of tensor operand.
            scalar: Scalar value.

        Returns:
            Index of new node.
        """
        var node = ExprNode(op)
        node.left_idx = tensor_idx
        node.scalar_idx = len(self._scalars)
        var node_idx = len(self._nodes)
        self._nodes.append(node)
        self._scalars.append(scalar)
        return node_idx


# ============================================================================
# Expression Entry Point
# ============================================================================


fn expr(tensor: ExTensor) raises -> TensorExpr:
    """Create a lazy expression from a tensor (entry point).

    This is the main entry point for building lazy expressions. It wraps
    a tensor in an expression container that can be combined with other
    expressions using standard operators.

    Args:
        tensor: Tensor to wrap in lazy expression.

    Returns:
        Expression ready for operator chaining.

    Raises:
        Error: If tensor is invalid.

    Example:
        ```mojo
        var expr1 = expr(tensor_a)
        var expr2 = expr(tensor_b)
        var combined = expr1 + expr2  # No computation yet
        ```
    """
    return TensorExpr(tensor)


# ============================================================================
# Operator Overloading - Binary Operations
# ============================================================================


fn __add__(a: TensorExpr, b: TensorExpr) raises -> TensorExpr:
    """Add two lazy expressions element-wise.

    Args:
        a: First expression.
        b: Second expression.

    Returns:
        New expression representing a + b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var result = a
    # Compute new broadcast shape
    result._result_shape = broadcast_shapes(a._result_shape, b._result_shape)

    # Add b's tensors and nodes to result
    var b_offset = len(result._tensors)
    var b_scalar_offset = len(result._scalars)
    for i in range(len(b._tensors)):
        result._tensors.append(b._tensors[i])
    for i in range(len(b._scalars)):
        result._scalars.append(b._scalars[i])

    # Copy b's nodes with offset adjustments
    var b_node_offset = len(result._nodes)
    for i in range(len(b._nodes)):
        var node = b._nodes[i]
        if node.tensor_idx >= 0:
            node.tensor_idx += b_offset
        if node.left_idx >= 0:
            node.left_idx += b_node_offset
        if node.right_idx >= 0:
            node.right_idx += b_node_offset
        if node.scalar_idx >= 0:
            node.scalar_idx += b_scalar_offset
        result._nodes.append(node)

    # Create binary operation node
    var add_node = ExprNode(OP_ADD)
    add_node.left_idx = a._root_idx
    add_node.right_idx = b_node_offset + b._root_idx
    result._root_idx = len(result._nodes)
    result._nodes.append(add_node)

    return result


fn __add__(a: TensorExpr, b: ExTensor) raises -> TensorExpr:
    """Add tensor to lazy expression.

    Args:
        a: Expression.
        b: Tensor.

    Returns:
        New expression representing a + b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var b_expr = expr(b)
    return __add__(a, b_expr)


fn __add__(a: ExTensor, b: TensorExpr) raises -> TensorExpr:
    """Add lazy expression to tensor.

    Args:
        a: Tensor.
        b: Expression.

    Returns:
        New expression representing a + b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var a_expr = expr(a)
    return __add__(a_expr, b)


fn __sub__(a: TensorExpr, b: TensorExpr) raises -> TensorExpr:
    """Subtract two lazy expressions element-wise.

    Args:
        a: First expression.
        b: Second expression.

    Returns:
        New expression representing a - b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var result = a
    result._result_shape = broadcast_shapes(a._result_shape, b._result_shape)

    var b_offset = len(result._tensors)
    var b_scalar_offset = len(result._scalars)
    for i in range(len(b._tensors)):
        result._tensors.append(b._tensors[i])
    for i in range(len(b._scalars)):
        result._scalars.append(b._scalars[i])

    var b_node_offset = len(result._nodes)
    for i in range(len(b._nodes)):
        var node = b._nodes[i]
        if node.tensor_idx >= 0:
            node.tensor_idx += b_offset
        if node.left_idx >= 0:
            node.left_idx += b_node_offset
        if node.right_idx >= 0:
            node.right_idx += b_node_offset
        if node.scalar_idx >= 0:
            node.scalar_idx += b_scalar_offset
        result._nodes.append(node)

    var sub_node = ExprNode(OP_SUB)
    sub_node.left_idx = a._root_idx
    sub_node.right_idx = b_node_offset + b._root_idx
    result._root_idx = len(result._nodes)
    result._nodes.append(sub_node)

    return result


fn __sub__(a: TensorExpr, b: ExTensor) raises -> TensorExpr:
    """Subtract tensor from lazy expression.

    Args:
        a: Expression.
        b: Tensor.

    Returns:
        New expression representing a - b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var b_expr = expr(b)
    return __sub__(a, b_expr)


fn __sub__(a: ExTensor, b: TensorExpr) raises -> TensorExpr:
    """Subtract lazy expression from tensor.

    Args:
        a: Tensor.
        b: Expression.

    Returns:
        New expression representing a - b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var a_expr = expr(a)
    return __sub__(a_expr, b)


fn __mul__(a: TensorExpr, b: TensorExpr) raises -> TensorExpr:
    """Multiply two lazy expressions element-wise.

    Args:
        a: First expression.
        b: Second expression.

    Returns:
        New expression representing a * b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var result = a
    result._result_shape = broadcast_shapes(a._result_shape, b._result_shape)

    var b_offset = len(result._tensors)
    var b_scalar_offset = len(result._scalars)
    for i in range(len(b._tensors)):
        result._tensors.append(b._tensors[i])
    for i in range(len(b._scalars)):
        result._scalars.append(b._scalars[i])

    var b_node_offset = len(result._nodes)
    for i in range(len(b._nodes)):
        var node = b._nodes[i]
        if node.tensor_idx >= 0:
            node.tensor_idx += b_offset
        if node.left_idx >= 0:
            node.left_idx += b_node_offset
        if node.right_idx >= 0:
            node.right_idx += b_node_offset
        if node.scalar_idx >= 0:
            node.scalar_idx += b_scalar_offset
        result._nodes.append(node)

    var mul_node = ExprNode(OP_MUL)
    mul_node.left_idx = a._root_idx
    mul_node.right_idx = b_node_offset + b._root_idx
    result._root_idx = len(result._nodes)
    result._nodes.append(mul_node)

    return result


fn __mul__(a: TensorExpr, b: ExTensor) raises -> TensorExpr:
    """Multiply lazy expression by tensor.

    Args:
        a: Expression.
        b: Tensor.

    Returns:
        New expression representing a * b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var b_expr = expr(b)
    return __mul__(a, b_expr)


fn __mul__(a: ExTensor, b: TensorExpr) raises -> TensorExpr:
    """Multiply tensor by lazy expression.

    Args:
        a: Tensor.
        b: Expression.

    Returns:
        New expression representing a * b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var a_expr = expr(a)
    return __mul__(a_expr, b)


fn __truediv__(a: TensorExpr, b: TensorExpr) raises -> TensorExpr:
    """Divide two lazy expressions element-wise.

    Args:
        a: First expression (dividend).
        b: Second expression (divisor).

    Returns:
        New expression representing a / b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var result = a
    result._result_shape = broadcast_shapes(a._result_shape, b._result_shape)

    var b_offset = len(result._tensors)
    var b_scalar_offset = len(result._scalars)
    for i in range(len(b._tensors)):
        result._tensors.append(b._tensors[i])
    for i in range(len(b._scalars)):
        result._scalars.append(b._scalars[i])

    var b_node_offset = len(result._nodes)
    for i in range(len(b._nodes)):
        var node = b._nodes[i]
        if node.tensor_idx >= 0:
            node.tensor_idx += b_offset
        if node.left_idx >= 0:
            node.left_idx += b_node_offset
        if node.right_idx >= 0:
            node.right_idx += b_node_offset
        if node.scalar_idx >= 0:
            node.scalar_idx += b_scalar_offset
        result._nodes.append(node)

    var div_node = ExprNode(OP_DIV)
    div_node.left_idx = a._root_idx
    div_node.right_idx = b_node_offset + b._root_idx
    result._root_idx = len(result._nodes)
    result._nodes.append(div_node)

    return result


fn __truediv__(a: TensorExpr, b: ExTensor) raises -> TensorExpr:
    """Divide lazy expression by tensor.

    Args:
        a: Expression.
        b: Tensor.

    Returns:
        New expression representing a / b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var b_expr = expr(b)
    return __truediv__(a, b_expr)


fn __truediv__(a: ExTensor, b: TensorExpr) raises -> TensorExpr:
    """Divide tensor by lazy expression.

    Args:
        a: Tensor.
        b: Expression.

    Returns:
        New expression representing a / b.

    Raises:
        Error: If shapes are not broadcast-compatible.
    """
    var a_expr = expr(a)
    return __truediv__(a_expr, b)


# ============================================================================
# Operator Overloading - Unary Operations
# ============================================================================


fn __neg__(a: TensorExpr) -> TensorExpr:
    """Negate a lazy expression element-wise.

    Args:
        a: Expression to negate.

    Returns:
        New expression representing -a.
    """
    var result = a
    var neg_node = ExprNode(OP_NEG)
    neg_node.left_idx = a._root_idx
    result._root_idx = len(result._nodes)
    result._nodes.append(neg_node)
    return result


# ============================================================================
# Operator Overloading - Scalar Operations
# ============================================================================


fn __mul__(a: TensorExpr, scalar: Float64) -> TensorExpr:
    """Multiply lazy expression by scalar.

    Args:
        a: Expression.
        scalar: Scalar value.

    Returns:
        New expression representing a * scalar.
    """
    var result = a
    var scalar_mul_node = ExprNode(OP_SCALAR_MUL)
    scalar_mul_node.left_idx = a._root_idx
    scalar_mul_node.scalar_idx = len(result._scalars)
    result._root_idx = len(result._nodes)
    result._nodes.append(scalar_mul_node)
    result._scalars.append(scalar)
    return result


fn __mul__(scalar: Float64, a: TensorExpr) -> TensorExpr:
    """Multiply scalar by lazy expression.

    Args:
        scalar: Scalar value.
        a: Expression.

    Returns:
        New expression representing scalar * a.
    """
    return __mul__(a, scalar)


fn __truediv__(a: TensorExpr, scalar: Float64) -> TensorExpr:
    """Divide lazy expression by scalar.

    Args:
        a: Expression.
        scalar: Scalar divisor.

    Returns:
        New expression representing a / scalar.
    """
    var result = a
    var scalar_div_node = ExprNode(OP_SCALAR_DIV)
    scalar_div_node.left_idx = a._root_idx
    scalar_div_node.scalar_idx = len(result._scalars)
    result._root_idx = len(result._nodes)
    result._nodes.append(scalar_div_node)
    result._scalars.append(scalar)
    return result
