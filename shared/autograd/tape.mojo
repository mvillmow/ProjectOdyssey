"""Gradient Tape - Operation recording for automatic differentiation.

Implements a tape-based autograd system that records operations during forward
pass and replays them in reverse during backward propagation.

The gradient tape maintains a directed acyclic graph (DAG) of operations where:
- Nodes represent tensors (Variables)
- Edges represent operations with their backward functions
- Topological ordering enables correct gradient flow via chain rule

This design follows PyTorch's eager execution model where operations are
recorded as they happen, rather than building a static computation graph.

Key Components:
- TapeNode: Represents a single operation in the computation graph
- GradientTape: Global tape that records all operations
- backward(): Traverses tape in reverse, applying chain rule

Architecture:
    Forward Pass:
        x = Variable(...)  # No recording
        y = x + 2          # Records: TapeNode(op="add", inputs=[x], output=y, backward_fn=add_backward)
        z = y * 3          # Records: TapeNode(op="mul", inputs=[y], output=z, backward_fn=mul_backward)

    Backward Pass:
        z.backward()       # Traverse tape in reverse:
                          # 1. grad_z = ones_like(z)  # ∂z/∂z = 1
                          # 2. grad_y = mul_backward(grad_z, y, 3)  # Chain rule
                          # 3. grad_x = add_backward(grad_y, x, 2)  # Chain rule
                          # 4. x.grad = grad_x

Examples:
    # Enable gradient tape recording
    var tape = GradientTape()
    tape.enable()

    # Operations are recorded automatically
    var x = Variable(data, requires_grad=True)
    var y = x * 2
    var z = y + 3
    var loss = z.sum()

    # Compute gradients
    tape.backward(loss)

    # Access gradients
    print(x.grad)  # Chain rule: ∂loss/∂x = ∂loss/∂z * ∂z/∂y * ∂y/∂x
"""

from ..core.extensor import ExTensor


# Operation types supported by the gradient tape
alias OP_ADD = "add"
alias OP_SUBTRACT = "subtract"
alias OP_MULTIPLY = "multiply"
alias OP_DIVIDE = "divide"
alias OP_MATMUL = "matmul"
alias OP_POWER = "power"
alias OP_SUM = "sum"
alias OP_MEAN = "mean"
alias OP_RELU = "relu"
alias OP_SIGMOID = "sigmoid"
alias OP_TANH = "tanh"
alias OP_SOFTMAX = "softmax"


struct TapeNode(Copyable, Movable):
    """Represents a single operation in the computation graph.

    Each node records:
    - The operation type (e.g., "add", "multiply", "matmul")
    - Input tensors that were used
    - Output tensor that was produced
    - Metadata needed for the backward pass

    During backward propagation, nodes are traversed in reverse topological
    order, and each node's backward function is called to compute gradients.

    Attributes:
        op_type: String identifier for the operation (e.g., "add", "matmul")
        input_ids: IDs of input Variables (for tracking dependencies)
        output_id: ID of output Variable
        saved_tensors: Tensors saved for backward pass (e.g., input shapes, values)

    Note:
        This is a simplified implementation. A full production system would need:
        - Memory-efficient tensor storage (weak references, tensor recomputation)
        - Support for in-place operations
        - Gradient checkpointing for memory/compute tradeoffs
    """

    var op_type: String
    var input_ids: List[Int]
    var output_id: Int
    # TODO: Add saved_tensors field for storing metadata needed in backward pass
    # For now, we'll rely on the backward functions in shared/core/ which take
    # the necessary inputs directly

    fn __init__(out self, op_type: String, input_ids: List[Int], output_id: Int):
        """Initialize a tape node.

        Args:
            op_type: String identifier for the operation
            input_ids: IDs of input Variables
            output_id: ID of output Variable
        """
        self.op_type = op_type
        self.input_ids = input_ids
        self.output_id = output_id


struct GradientTape:
    """Global gradient tape for recording operations.

    The tape maintains a chronological list of all operations performed on
    Variables with requires_grad=True. During backward(), the tape is traversed
    in reverse to compute gradients via the chain rule.

    Attributes:
        nodes: Chronological list of recorded operations
        enabled: Whether the tape is currently recording

    Design Note:
        This uses a global tape approach (like TensorFlow's GradientTape) rather
        than per-tensor graph tracking (like PyTorch). The global tape is simpler
        to implement but requires explicit enable/disable calls.

    Examples:
        var tape = GradientTape()
        tape.enable()

        var x = Variable(data, requires_grad=True)
        var y = x * 2  # Recorded
        var z = y + 3  # Recorded

        tape.backward(z)  # Compute all gradients
        tape.disable()
    """

    var nodes: List[TapeNode]
    var enabled: Bool

    fn __init__(mut self):
        """Initialize an empty gradient tape."""
        self.nodes = List[TapeNode]()
        self.enabled = False

    fn enable(mut self):
        """Enable recording of operations.

        After calling enable(), all operations on Variables with requires_grad=True
        will be recorded in the tape.

        Examples:
            var tape = GradientTape()
            tape.enable()
            var y = x + 1  # Recorded
        """
        self.enabled = True

    fn disable(mut self):
        """Disable recording of operations.

        After calling disable(), operations will not be recorded. This is useful
        for inference or when you want to break the computation graph.

        Examples:
            tape.disable()
            var y = x + 1  # Not recorded (no gradient tracking)
        """
        self.enabled = False

    fn clear(mut self):
        """Clear all recorded operations.

        Resets the tape to an empty state. Should be called after backward()
        to free memory and prepare for the next forward pass.

        Examples:
            tape.backward(loss)
            tape.clear()  # Free memory
        """
        self.nodes = List[TapeNode]()

    fn record(mut self, op_type: String, input_ids: List[Int], output_id: Int):
        """Record an operation in the tape.

        This method is called internally by Variable operations to register
        themselves in the computation graph.

        Args:
            op_type: String identifier for the operation
            input_ids: IDs of input Variables
            output_id: ID of output Variable

        Note:
            Only records if tape is enabled and at least one input requires gradients.

        Examples:
            # Internal use by Variable operations
            if tape.enabled and (x.requires_grad or y.requires_grad):
                tape.record("add", input_ids, output_id)
        """
        if not self.enabled:
            return

        var node = TapeNode(op_type, input_ids, output_id)
        self.nodes.append(node)

    fn backward(mut self):
        """Compute gradients by traversing tape in reverse.

        Applies the chain rule in reverse topological order:
        1. Initialize gradient of final output to ones (∂L/∂L = 1)
        2. For each node in reverse:
           a. Get upstream gradient (∂L/∂output)
           b. Call backward function to get local gradients (∂output/∂inputs)
           c. Apply chain rule: ∂L/∂input = ∂L/∂output * ∂output/∂input
           d. Accumulate gradients in input Variables

        Note:
            This is a placeholder. Full implementation requires:
            - Mapping from Variable IDs to Variable objects
            - Calling the appropriate backward function for each operation
            - Handling gradient accumulation for Variables used multiple times

        Examples:
            var x = Variable(data, requires_grad=True)
            var loss = compute_loss(x)
            tape.backward()  # Populates x.grad
        """
        # TODO: Implement backward pass
        # For now, this is a placeholder that will be implemented once we have:
        # 1. Variable ID to Variable mapping
        # 2. Dispatching to correct backward functions
        # 3. Gradient accumulation logic
        pass


# Global gradient tape instance
# TODO: Consider thread-local storage for multi-threaded training
# var global_tape = GradientTape()
