"""Attention mechanisms for transformer architectures.

This module provides pure functional implementations of attention operations.
All operations are stateless - caller manages parameters and state.

The core operation is scaled dot-product attention:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where:
    Q: Query tensor of shape (batch, seq_len, d_k)
    K: Key tensor of shape (batch, seq_len, d_k)
    V: Value tensor of shape (batch, seq_len, d_v)
    d_k: Key dimension (used for scaling)
"""

from .extensor import ExTensor, zeros, zeros_like
from .matrix import matmul, transpose
from .activation import softmax
from .arithmetic import multiply, divide, add
from math import sqrt


fn scaled_dot_product_attention(
    query: ExTensor,
    key: ExTensor,
    value: ExTensor,
    mask: ExTensor = ExTensor(),
    dropout_p: Float64 = 0.0,
) raises -> ExTensor:
    """Scaled dot-product attention.

    Computes attention weights from query-key similarity and applies them to values.
    This is the fundamental building block of transformer architectures.

    Args:
        `query`: Query tensor of shape (batch, seq_len, d_k) or (batch, heads, seq_len, d_k)
        `key`: Key tensor of shape (batch, seq_len, d_k) or (batch, heads, seq_len, d_k)
        `value`: Value tensor of shape (batch, seq_len, d_v) or (batch, heads, seq_len, d_v)
        `mask`: Optional attention mask. Use large negative values (-1e9) for positions
               to ignore. Shape: (batch, seq_len, seq_len) or (batch, heads, seq_len, seq_len)
        `dropout_p`: Dropout probability (not applied in this implementation)

    Returns:
        Attention output of shape (batch, seq_len, d_v) or (batch, heads, seq_len, d_v)

    Example:
        ```mojo
        from shared.core import scaled_dot_product_attention

        # Create Q, K, V tensors: (batch=2, seq_len=10, d_k=64)
        var query = ones([2, 10, 64], DType.float32)
        var key = ones([2, 10, 64], DType.float32)
        var value = ones([2, 10, 64], DType.float32)

        # Compute attention
        var output = scaled_dot_product_attention(query, key, value)
        # output shape: (2, 10, 64)
        ```

    Formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Note:
        - The scaling factor sqrt(d_k) prevents dot products from growing too large
        - Masking is applied before softmax (additive masking)
        - For causal (autoregressive) attention, use a lower-triangular mask
    """
    var q_shape = query.shape()
    var k_shape = key.shape()
    var v_shape = value.shape()

    # Support both 3D (batch, seq, dim) and 4D (batch, heads, seq, dim) inputs
    var ndim = len(q_shape)
    if ndim != 3 and ndim != 4:
        raise Error(
            "scaled_dot_product_attention: query must be 3D or 4D tensor"
        )

    # Get d_k for scaling (last dimension of query/key)
    var d_k = q_shape[ndim - 1]
    var scale = Float64(1.0) / sqrt(Float64(d_k))

    # Compute QK^T
    # For 3D: (batch, seq_q, d_k) @ (batch, d_k, seq_k) -> (batch, seq_q, seq_k)
    # For 4D: (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k) -> (batch, heads, seq_q, seq_k)
    var key_t = transpose(key)
    var scores = matmul(query, key_t)

    # Scale scores
    var scale_tensor = zeros_like(scores)
    var scale_ptr = scale_tensor._data
    var numel = scores.numel()

    if scores.dtype() == DType.float32:
        var scale_f32 = Float32(scale)
        for i in range(numel):
            scale_ptr.bitcast[Float32]()[i] = scale_f32
    elif scores.dtype() == DType.float64:
        for i in range(numel):
            scale_ptr.bitcast[Float64]()[i] = scale
    else:
        raise Error("scaled_dot_product_attention: only float32/64 supported")

    var scaled_scores = multiply(scores, scale_tensor)

    # Apply mask if provided (additive masking)
    var mask_shape = mask.shape()
    if len(mask_shape) > 0 and mask.numel() > 0:
        scaled_scores = add(scaled_scores, mask)

    # Apply softmax along the last dimension (key sequence length)
    var attention_weights = softmax(scaled_scores)

    # Apply attention weights to values
    # (batch, seq_q, seq_k) @ (batch, seq_k, d_v) -> (batch, seq_q, d_v)
    var output = matmul(attention_weights, value)

    return output


struct ScaledDotProductAttentionBackwardResult(Movable):
    """Result container for scaled_dot_product_attention_backward.

    Contains gradients for query, key, and value tensors.
    """

    var grad_query: ExTensor
    var grad_key: ExTensor
    var grad_value: ExTensor

    fn __init__(
        out self,
        grad_query: ExTensor,
        grad_key: ExTensor,
        grad_value: ExTensor,
    ):
        self.grad_query = grad_query
        self.grad_key = grad_key
        self.grad_value = grad_value

    fn __moveinit__(out self, owned existing: Self):
        self.grad_query = existing.grad_query^
        self.grad_key = existing.grad_key^
        self.grad_value = existing.grad_value^


fn scaled_dot_product_attention_backward(
    grad_output: ExTensor,
    query: ExTensor,
    key: ExTensor,
    value: ExTensor,
    attention_weights: ExTensor,
    mask: ExTensor = ExTensor(),
) raises -> ScaledDotProductAttentionBackwardResult:
    """Backward pass for scaled dot-product attention.

    Computes gradients with respect to query, key, and value tensors.

    Args:
        `grad_output`: Gradient w.r.t. attention output
        `query`: Original query tensor
        `key`: Original key tensor
        `value`: Original value tensor
        `attention_weights`: Attention weights from forward pass (after softmax)
        `mask`: Optional attention mask (same as forward pass)

    Returns:
        ScaledDotProductAttentionBackwardResult containing:
            - grad_query: Gradient w.r.t. query
            - grad_key: Gradient w.r.t. key
            - grad_value: Gradient w.r.t. value

    Example:
        ```mojo
        from shared.core import (
            scaled_dot_product_attention,
            scaled_dot_product_attention_backward,
        )

        # Forward pass
        var output = scaled_dot_product_attention(query, key, value)

        # ... compute loss and grad_output ...

        # Backward pass
        var result = scaled_dot_product_attention_backward(
            grad_output, query, key, value, attention_weights
        )
        # result.grad_query, result.grad_key, result.grad_value
        ```

    Note:
        Caller must save attention_weights from forward pass for use in backward.
        Pure functional: returns new tensors, does not modify inputs.
    """
    var q_shape = query.shape()
    var ndim = len(q_shape)
    var d_k = q_shape[ndim - 1]
    var scale = Float64(1.0) / sqrt(Float64(d_k))

    # Gradient w.r.t. value: attention_weights^T @ grad_output
    var attention_weights_t = transpose(attention_weights)
    var grad_value = matmul(attention_weights_t, grad_output)

    # Gradient w.r.t. attention weights: grad_output @ value^T
    var value_t = transpose(value)
    var grad_attention = matmul(grad_output, value_t)

    # Gradient through softmax
    # d_softmax/d_input = softmax * (grad - sum(grad * softmax))
    var grad_softmax = _softmax_backward(grad_attention, attention_weights)

    # Scale the gradient
    var scale_tensor = zeros_like(grad_softmax)
    var scale_ptr = scale_tensor._data
    var numel = grad_softmax.numel()

    if grad_softmax.dtype() == DType.float32:
        var scale_f32 = Float32(scale)
        for i in range(numel):
            scale_ptr.bitcast[Float32]()[i] = scale_f32
    elif grad_softmax.dtype() == DType.float64:
        for i in range(numel):
            scale_ptr.bitcast[Float64]()[i] = scale
    else:
        raise Error("attention backward: only float32/64 supported")

    var grad_scores = multiply(grad_softmax, scale_tensor)

    # Gradient w.r.t. query: grad_scores @ key
    var grad_query = matmul(grad_scores, key)

    # Gradient w.r.t. key: grad_scores^T @ query
    var grad_scores_t = transpose(grad_scores)
    var grad_key = matmul(grad_scores_t, query)

    return ScaledDotProductAttentionBackwardResult(
        grad_query, grad_key, grad_value
    )


fn _softmax_backward(
    grad_output: ExTensor, softmax_output: ExTensor
) raises -> ExTensor:
    """Internal helper for softmax backward pass.

    Computes gradient through softmax: d_softmax/d_input = s * (g - sum(g * s))
    where s = softmax_output and g = grad_output.
    """
    var shape = grad_output.shape()
    var result = zeros_like(grad_output)

    var grad_ptr = grad_output._data
    var softmax_ptr = softmax_output._data
    var result_ptr = result._data

    # Get dimensions for the last axis (softmax is applied along last axis)
    var ndim = len(shape)
    var last_dim = shape[ndim - 1]
    var batch_size = grad_output.numel() // last_dim

    if grad_output.dtype() == DType.float32:
        for b in range(batch_size):
            var offset = b * last_dim

            # Compute sum(grad * softmax) for this batch element
            var dot_sum = Float32(0.0)
            for i in range(last_dim):
                dot_sum += (
                    grad_ptr.bitcast[Float32]()[offset + i]
                    * softmax_ptr.bitcast[Float32]()[offset + i]
                )

            # Compute gradient: softmax * (grad - dot_sum)
            for i in range(last_dim):
                var s = softmax_ptr.bitcast[Float32]()[offset + i]
                var g = grad_ptr.bitcast[Float32]()[offset + i]
                result_ptr.bitcast[Float32]()[offset + i] = s * (g - dot_sum)

    elif grad_output.dtype() == DType.float64:
        for b in range(batch_size):
            var offset = b * last_dim

            var dot_sum = Float64(0.0)
            for i in range(last_dim):
                dot_sum += (
                    grad_ptr.bitcast[Float64]()[offset + i]
                    * softmax_ptr.bitcast[Float64]()[offset + i]
                )

            for i in range(last_dim):
                var s = softmax_ptr.bitcast[Float64]()[offset + i]
                var g = grad_ptr.bitcast[Float64]()[offset + i]
                result_ptr.bitcast[Float64]()[offset + i] = s * (g - dot_sum)

    else:
        raise Error("_softmax_backward: only float32/64 supported")

    return result


fn create_causal_mask(seq_len: Int, dtype: DType = DType.float32) raises -> ExTensor:
    """Create a causal (lower-triangular) attention mask.

    Returns a mask where positions that should be ignored have large negative
    values (-1e9) and valid positions have 0.

    Args:
        `seq_len`: Sequence length for the mask
        `dtype`: Data type for the mask tensor

    Returns:
        Mask tensor of shape (seq_len, seq_len) suitable for attention.

    Example:
        ```mojo
        from shared.core import create_causal_mask, scaled_dot_product_attention

        var mask = create_causal_mask(10)
        var output = scaled_dot_product_attention(query, key, value, mask=mask)
        ```

    Note:
        The mask is designed for additive masking before softmax.
        Position (i, j) is masked (set to -1e9) if j > i (future position).
    """
    var shape = List[Int]()
    shape.append(seq_len)
    shape.append(seq_len)

    var mask = zeros(shape, dtype)
    var mask_ptr = mask._data

    var neg_inf = Float64(-1e9)

    if dtype == DType.float32:
        var neg_inf_f32 = Float32(neg_inf)
        for i in range(seq_len):
            for j in range(seq_len):
                var idx = i * seq_len + j
                if j > i:
                    # Future position - mask it out
                    mask_ptr.bitcast[Float32]()[idx] = neg_inf_f32
                # else: keep as 0 (no masking)

    elif dtype == DType.float64:
        for i in range(seq_len):
            for j in range(seq_len):
                var idx = i * seq_len + j
                if j > i:
                    mask_ptr.bitcast[Float64]()[idx] = neg_inf

    else:
        raise Error("create_causal_mask: only float32/64 supported")

    return mask
