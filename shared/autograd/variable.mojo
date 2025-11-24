"""Variable - Autograd-enabled tensor wrapper.

Provides automatic differentiation capabilities by wrapping ExTensor with
gradient tracking and computation graph recording.

This module implements a tape-based autograd system similar to PyTorch's eager
mode execution, where operations are recorded during the forward pass and
replayed in reverse during backward propagation.

Key Concepts:
- Variable wraps an ExTensor and adds requires_grad flag and grad storage
- Operations on Variables are recorded in a global gradient tape
- Calling .backward() triggers automatic gradient computation via chain rule
- Gradients accumulate across multiple backward passes (call .zero_grad() to reset)

Examples:
    # Create variables with gradient tracking
    var x = Variable(zeros(shape, dtype), requires_grad=True)
    var y = Variable(ones(shape, dtype), requires_grad=True)

    # Perform operations (recorded in tape)
    var z = x + y
    var loss = (z * z).sum()

    # Compute gradients automatically
    loss.backward()

    # Access gradients
    print(x.grad)  # dLoss/dx
    print(y.grad)  # dLoss/dy
"""

from ..core.extensor import ExTensor
from collections.optional import Optional


struct Variable:
    """Tensor wrapper with automatic differentiation support.

    Variable extends ExTensor with gradient tracking capabilities. Each Variable
    maintains:
    - data: The actual tensor values (ExTensor)
    - grad: Accumulated gradients (Optional[ExTensor])
    - requires_grad: Whether to track operations for this variable
    - grad_fn: Backward function to compute gradients (Optional)

    Attributes:
        data: The underlying ExTensor containing values
        grad: Optional gradient tensor (None until backward() is called)
        requires_grad: Flag indicating whether this Variable participates in autograd

    Note:
        Variables are immutable after creation. Operations create new Variables
        with updated data and grad_fn, following functional programming patterns.
    """

    var data: ExTensor
    var grad: Optional[ExTensor]
    var requires_grad: Bool
    # TODO: Add grad_fn field once we implement backward functions

    fn __init__(mut self, var data: ExTensor, requires_grad: Bool = False):
        """Initialize a Variable from an ExTensor.

        Args:
            data: The tensor values to wrap (ownership transferred)
            requires_grad: Whether to track gradients for this variable (default: False)

        Examples:
            var x = Variable(zeros(shape, dtype), requires_grad=True)
            var y = Variable(ones(shape, dtype))  # requires_grad=False by default
        """
        self.data = data^
        self.grad = None
        self.requires_grad = requires_grad

    fn __init__(mut self, var data: ExTensor, var grad: ExTensor, requires_grad: Bool = False):
        """Initialize a Variable with explicit gradient.

        Args:
            data: The tensor values to wrap (ownership transferred)
            grad: The gradient tensor (ownership transferred)
            requires_grad: Whether to track gradients for this variable

        Note:
            This constructor is primarily for internal use when creating
            intermediate Variables during operations.
        """
        self.data = data^
        self.grad = grad^
        self.requires_grad = requires_grad

    fn zero_grad(mut self):
        """Reset gradients to None.

        Should be called before each backward pass to prevent gradient accumulation
        from previous iterations.

        Examples:
            for epoch in range(num_epochs):
                optimizer.zero_grad()  # Clear previous gradients
                loss = forward(x)
                loss.backward()
                optimizer.step()
        """
        self.grad = None

    fn backward(mut self):
        """Compute gradients via automatic differentiation.

        Triggers backward pass through the computation graph, computing gradients
        for all Variables with requires_grad=True that were used to compute this
        Variable.

        The gradient of this Variable with respect to itself is initialized to
        ones (∂self/∂self = 1), then gradients are propagated backward through
        the graph using the chain rule.

        Note:
            Currently this is a placeholder. Full implementation requires:
            1. Gradient tape to record operations
            2. Backward functions for each operation
            3. Topological sort of computation graph
            4. Chain rule application in reverse order

        Raises:
            Error if this Variable is not a scalar (backward() requires scalar output)

        Examples:
            var x = Variable(data, requires_grad=True)
            var loss = compute_loss(x)
            loss.backward()  # Computes gradients for all inputs
            print(x.grad)    # ∂loss/∂x
        """
        # TODO: Implement backward pass
        # For now, this is a placeholder that will be implemented once we have:
        # 1. Gradient tape recording operations
        # 2. Backward functions registered for each operation
        # 3. Topological sorting of computation graph
        pass

    fn detach(self) -> Variable:
        """Create a new Variable with the same data but no gradient tracking.

        Useful for breaking the computation graph when you want to use values
        without tracking gradients.

        Returns:
            New Variable with requires_grad=False and no grad_fn

        Examples:
            var x = Variable(data, requires_grad=True)
            var y = x.detach()  # y shares data but doesn't track gradients
            var z = y + 1       # Operations on y don't affect x's gradient
        """
        return Variable(self.data, requires_grad=False)
