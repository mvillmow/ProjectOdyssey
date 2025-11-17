"""
Assertion helpers for ExTensor testing.

Provides custom assertion functions for validating tensor properties,
shapes, dtypes, values, and numerical accuracy.
"""

from tensor import Tensor, TensorShape
from utils.index import Index

# Note: ExTensor will be imported once implemented
# from extensor import ExTensor


fn assert_equal[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) raises:
    """
    Assert that two tensors are exactly equal.

    Checks shape, dtype, and all element values for exact equality.

    Args:
        a: First tensor
        b: Second tensor

    Raises:
        Error if tensors are not exactly equal
