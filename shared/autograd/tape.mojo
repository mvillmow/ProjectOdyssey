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
        y = x + 2          # Records: TapeNode(op="add", inputs=[x], output=y)
        z = y * 3          # Records: TapeNode(op="mul", inputs=[y], output=z)

    Backward Pass:
        z.backward()       # Traverse tape in reverse:
                          # 1. grad_z = ones_like(z)  # dz/dz = 1
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
    print(x.grad)  # Chain rule: dLoss/dx
"""

from ..core.extensor import ExTensor, ones_like, zeros_like
from ..core.arithmetic import (
    add_backward,
    subtract_backward,
    multiply_backward,
    divide_backward,
)
from ..core.reduction import (
    sum_backward,
    mean_backward,
)
from ..core.matrix import (
    matmul_backward,
)
from ..core.activation import (
    relu_backward,
    sigmoid_backward,
    tanh_backward,
    softmax_backward,
)


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
alias OP_NEG = "neg"
alias OP_EXP = "exp"
alias OP_LOG = "log"
alias OP_SQRT = "sqrt"


struct SavedTensors(Movable):
    """Container for tensors saved during forward pass for backward computation.

    Different operations need different tensors saved:
    - Binary ops (add, mul): Need both inputs and output
    - Unary ops (relu, exp): Need input and output
    - Reductions (sum, mean): Need input shape for broadcasting back
    """

    var tensors: List[ExTensor]
    var shapes: List[List[Int]]
    var scalars: List[Float64]

    fn __init__(out self):
        """Initialize empty saved tensors."""
        self.tensors = List[ExTensor]()
        self.shapes = List[List[Int]]()
        self.scalars = List[Float64]()

    fn __moveinit__(out self, owned existing: Self):
        """Move constructor."""
        self.tensors = existing.tensors^
        self.shapes = existing.shapes^
        self.scalars = existing.scalars^

    fn add_tensor(mut self, tensor: ExTensor) raises:
        """Save a tensor for backward pass."""
        # Create a copy of the tensor
        var copy = zeros_like(tensor)
        var size = tensor.numel()
        for i in range(size):
            copy._data[i] = tensor._data[i]
        self.tensors.append(copy^)

    fn add_shape(mut self, shape: List[Int]):
        """Save a shape for backward pass."""
        var shape_copy = List[Int]()
        for i in range(len(shape)):
            shape_copy.append(shape[i])
        self.shapes.append(shape_copy^)

    fn add_scalar(mut self, value: Float64):
        """Save a scalar for backward pass."""
        self.scalars.append(value)


struct TapeNode(Movable):
    """Represents a single operation in the computation graph.

    Each node records:
    - The operation type (e.g., "add", "multiply", "matmul")
    - Input variable IDs that were used
    - Output variable ID that was produced
    - Saved tensors needed for the backward pass

    During backward propagation, nodes are traversed in reverse topological
    order, and each node's backward function is called to compute gradients.

    Attributes:
        op_type: String identifier for the operation (e.g., "add", "matmul")
        input_ids: IDs of input Variables (for tracking dependencies)
        output_id: ID of output Variable
        saved_tensors: Tensors saved for backward pass
    """

    var op_type: String
    var input_ids: List[Int]
    var output_id: Int
    var saved: SavedTensors

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
        self.saved = SavedTensors()

    fn __init__(
        out self,
        op_type: String,
        input_ids: List[Int],
        output_id: Int,
        owned saved: SavedTensors,
    ):
        """Initialize a tape node with saved tensors.

        Args:
            op_type: String identifier for the operation
            input_ids: IDs of input Variables
            output_id: ID of output Variable
            saved: Saved tensors for backward pass
        """
        self.op_type = op_type
        self.input_ids = input_ids
        self.output_id = output_id
        self.saved = saved^

    fn __moveinit__(out self, owned existing: Self):
        """Move constructor."""
        self.op_type = existing.op_type
        self.input_ids = existing.input_ids^
        self.output_id = existing.output_id
        self.saved = existing.saved^


struct VariableRegistry:
    """Registry mapping variable IDs to their gradient tensors.

    This allows the backward pass to look up and accumulate gradients
    for variables by their ID.
    """

    var grads: List[ExTensor]
    var has_grad: List[Bool]
    var requires_grad: List[Bool]
    var next_id: Int

    fn __init__(out self):
        """Initialize empty registry."""
        self.grads = List[ExTensor]()
        self.has_grad = List[Bool]()
        self.requires_grad = List[Bool]()
        self.next_id = 0

    fn register(mut self, requires_grad: Bool) -> Int:
        """Register a new variable and return its ID.

        Args:
            requires_grad: Whether this variable requires gradients

        Returns:
            The unique ID assigned to this variable
        """
        var id = self.next_id
        self.next_id += 1

        # Extend lists to accommodate new ID
        # Create a placeholder tensor (will be replaced when gradient is computed)
        var placeholder_shape = List[Int]()
        placeholder_shape.append(1)
        var placeholder = ExTensor(placeholder_shape, DType.float32)
        self.grads.append(placeholder^)
        self.has_grad.append(False)
        self.requires_grad.append(requires_grad)

        return id

    fn set_grad(mut self, id: Int, grad: ExTensor) raises:
        """Set or accumulate gradient for a variable.

        Args:
            id: Variable ID
            grad: Gradient tensor to set/accumulate
        """
        if id >= len(self.grads):
            return

        if self.has_grad[id]:
            # Accumulate gradients
            var existing = self.grads[id]
            var size = grad.numel()
            for i in range(size):
                existing._data[i] = existing._data[i] + grad._data[i]
            self.grads[id] = existing^
        else:
            # First gradient - copy it
            var grad_copy = zeros_like(grad)
            var size = grad.numel()
            for i in range(size):
                grad_copy._data[i] = grad._data[i]
            self.grads[id] = grad_copy^
            self.has_grad[id] = True

    fn get_grad(self, id: Int) -> ExTensor:
        """Get gradient for a variable.

        Args:
            id: Variable ID

        Returns:
            The gradient tensor (or placeholder if not computed)
        """
        if id < len(self.grads):
            return self.grads[id]
        # Return empty placeholder
        var placeholder_shape = List[Int]()
        placeholder_shape.append(1)
        return ExTensor(placeholder_shape, DType.float32)

    fn has_gradient(self, id: Int) -> Bool:
        """Check if a variable has a computed gradient.

        Args:
            id: Variable ID

        Returns:
            True if gradient has been computed
        """
        if id < len(self.has_grad):
            return self.has_grad[id]
        return False

    fn clear(mut self):
        """Clear all gradients but keep variable registrations."""
        for i in range(len(self.has_grad)):
            self.has_grad[i] = False


struct GradientTape:
    """Global gradient tape for recording operations.

    The tape maintains a chronological list of all operations performed on
    Variables with requires_grad=True. During backward(), the tape is traversed
    in reverse to compute gradients via the chain rule.

    Attributes:
        nodes: Chronological list of recorded operations
        enabled: Whether the tape is currently recording
        registry: Variable registry for gradient storage

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
    var registry: VariableRegistry

    fn __init__(out self):
        """Initialize an empty gradient tape."""
        self.nodes = List[TapeNode]()
        self.enabled = False
        self.registry = VariableRegistry()

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
        self.registry.clear()

    fn register_variable(mut self, requires_grad: Bool) -> Int:
        """Register a new variable in the tape's registry.

        Args:
            requires_grad: Whether this variable requires gradients

        Returns:
            Unique ID for the variable
        """
        return self.registry.register(requires_grad)

    fn record(
        mut self,
        op_type: String,
        input_ids: List[Int],
        output_id: Int,
        owned saved: SavedTensors,
    ):
        """Record an operation in the tape.

        This method is called internally by Variable operations to register
        themselves in the computation graph.

        Args:
            op_type: String identifier for the operation
            input_ids: IDs of input Variables
            output_id: ID of output Variable
            saved: Saved tensors for backward pass

        Note:
            Only records if tape is enabled.

        Examples:
            # Internal use by Variable operations
            if tape.enabled:
                tape.record("add", input_ids, output_id, saved)
        """
        if not self.enabled:
            return

        var node = TapeNode(op_type, input_ids, output_id, saved^)
        self.nodes.append(node^)

    fn backward(mut self, output_id: Int, output_grad: ExTensor) raises:
        """Compute gradients by traversing tape in reverse.

        Applies the chain rule in reverse topological order:
        1. Initialize gradient of final output with provided grad
        2. For each node in reverse:
           a. Get upstream gradient (dL/d_output)
           b. Call backward function to get local gradients (d_output/d_inputs)
           c. Apply chain rule: dL/d_input = dL/d_output * d_output/d_input
           d. Accumulate gradients in input Variables

        Args:
            output_id: ID of the output variable to backprop from
            output_grad: Gradient of the loss with respect to output

        Examples:
            var loss = compute_loss(x)
            var ones = ones_like(loss.data)
            tape.backward(loss.id, ones)  # Populates x.grad
        """
        # Set the gradient of the output
        self.registry.set_grad(output_id, output_grad)

        # Traverse nodes in reverse order
        var num_nodes = len(self.nodes)
        for rev_i in range(num_nodes):
            var i = num_nodes - 1 - rev_i
            var node = self.nodes[i]

            # Skip if output doesn't have gradient yet
            if not self.registry.has_gradient(node.output_id):
                continue

            var grad_output = self.registry.get_grad(node.output_id)

            # Dispatch to appropriate backward function based on op_type
            if node.op_type == OP_ADD:
                self._backward_add(node, grad_output)
            elif node.op_type == OP_SUBTRACT:
                self._backward_subtract(node, grad_output)
            elif node.op_type == OP_MULTIPLY:
                self._backward_multiply(node, grad_output)
            elif node.op_type == OP_DIVIDE:
                self._backward_divide(node, grad_output)
            elif node.op_type == OP_SUM:
                self._backward_sum(node, grad_output)
            elif node.op_type == OP_MEAN:
                self._backward_mean(node, grad_output)
            elif node.op_type == OP_MATMUL:
                self._backward_matmul(node, grad_output)
            elif node.op_type == OP_RELU:
                self._backward_relu(node, grad_output)
            elif node.op_type == OP_SIGMOID:
                self._backward_sigmoid(node, grad_output)
            elif node.op_type == OP_TANH:
                self._backward_tanh(node, grad_output)
            elif node.op_type == OP_NEG:
                self._backward_neg(node, grad_output)
            # Add more operations as needed

    fn _backward_add(mut self, node: TapeNode, grad_output: ExTensor) raises:
        """Backward pass for addition: d(a+b)/da = 1, d(a+b)/db = 1."""
        # grad_input = grad_output (gradient flows unchanged)
        # Use add_backward from shared.core
        var result = add_backward(grad_output)

        if len(node.input_ids) >= 1:
            self.registry.set_grad(node.input_ids[0], result.grad_a)
        if len(node.input_ids) >= 2:
            self.registry.set_grad(node.input_ids[1], result.grad_b)

    fn _backward_subtract(mut self, node: TapeNode, grad_output: ExTensor) raises:
        """Backward pass for subtraction: d(a-b)/da = 1, d(a-b)/db = -1."""
        var result = subtract_backward(grad_output)

        if len(node.input_ids) >= 1:
            self.registry.set_grad(node.input_ids[0], result.grad_a)
        if len(node.input_ids) >= 2:
            self.registry.set_grad(node.input_ids[1], result.grad_b)

    fn _backward_multiply(mut self, node: TapeNode, grad_output: ExTensor) raises:
        """Backward pass for multiplication: d(a*b)/da = b, d(a*b)/db = a."""
        if len(node.saved.tensors) < 2:
            return

        var a = node.saved.tensors[0]
        var b = node.saved.tensors[1]
        var result = multiply_backward(grad_output, a, b)

        if len(node.input_ids) >= 1:
            self.registry.set_grad(node.input_ids[0], result.grad_a)
        if len(node.input_ids) >= 2:
            self.registry.set_grad(node.input_ids[1], result.grad_b)

    fn _backward_divide(mut self, node: TapeNode, grad_output: ExTensor) raises:
        """Backward pass for division: d(a/b)/da = 1/b, d(a/b)/db = -a/b^2."""
        if len(node.saved.tensors) < 2:
            return

        var a = node.saved.tensors[0]
        var b = node.saved.tensors[1]
        var result = divide_backward(grad_output, a, b)

        if len(node.input_ids) >= 1:
            self.registry.set_grad(node.input_ids[0], result.grad_a)
        if len(node.input_ids) >= 2:
            self.registry.set_grad(node.input_ids[1], result.grad_b)

    fn _backward_sum(mut self, node: TapeNode, grad_output: ExTensor) raises:
        """Backward pass for sum reduction."""
        if len(node.saved.shapes) < 1:
            return

        var input_shape = node.saved.shapes[0]
        var axis = -1  # Default: full reduction
        if len(node.saved.scalars) >= 1:
            axis = Int(node.saved.scalars[0])

        var grad_input = sum_backward(grad_output, input_shape, axis)

        if len(node.input_ids) >= 1:
            self.registry.set_grad(node.input_ids[0], grad_input)

    fn _backward_mean(mut self, node: TapeNode, grad_output: ExTensor) raises:
        """Backward pass for mean reduction."""
        if len(node.saved.shapes) < 1:
            return

        var input_shape = node.saved.shapes[0]
        var axis = -1  # Default: full reduction
        if len(node.saved.scalars) >= 1:
            axis = Int(node.saved.scalars[0])

        var grad_input = mean_backward(grad_output, input_shape, axis)

        if len(node.input_ids) >= 1:
            self.registry.set_grad(node.input_ids[0], grad_input)

    fn _backward_matmul(mut self, node: TapeNode, grad_output: ExTensor) raises:
        """Backward pass for matrix multiplication."""
        if len(node.saved.tensors) < 2:
            return

        var a = node.saved.tensors[0]
        var b = node.saved.tensors[1]
        var result = matmul_backward(grad_output, a, b)

        if len(node.input_ids) >= 1:
            self.registry.set_grad(node.input_ids[0], result.grad_a)
        if len(node.input_ids) >= 2:
            self.registry.set_grad(node.input_ids[1], result.grad_b)

    fn _backward_relu(mut self, node: TapeNode, grad_output: ExTensor) raises:
        """Backward pass for ReLU activation."""
        if len(node.saved.tensors) < 1:
            return

        var x = node.saved.tensors[0]
        var grad_input = relu_backward(grad_output, x)

        if len(node.input_ids) >= 1:
            self.registry.set_grad(node.input_ids[0], grad_input)

    fn _backward_sigmoid(mut self, node: TapeNode, grad_output: ExTensor) raises:
        """Backward pass for sigmoid activation."""
        if len(node.saved.tensors) < 1:
            return

        # saved tensor is the output of sigmoid
        var output = node.saved.tensors[0]
        var grad_input = sigmoid_backward(grad_output, output)

        if len(node.input_ids) >= 1:
            self.registry.set_grad(node.input_ids[0], grad_input)

    fn _backward_tanh(mut self, node: TapeNode, grad_output: ExTensor) raises:
        """Backward pass for tanh activation."""
        if len(node.saved.tensors) < 1:
            return

        # saved tensor is the output of tanh
        var output = node.saved.tensors[0]
        var grad_input = tanh_backward(grad_output, output)

        if len(node.input_ids) >= 1:
            self.registry.set_grad(node.input_ids[0], grad_input)

    fn _backward_neg(mut self, node: TapeNode, grad_output: ExTensor) raises:
        """Backward pass for negation: d(-x)/dx = -1."""
        # Negate the gradient
        var grad_input = zeros_like(grad_output)
        var size = grad_output.numel()
        for i in range(size):
            grad_input._data[i] = -grad_output._data[i]

        if len(node.input_ids) >= 1:
            self.registry.set_grad(node.input_ids[0], grad_input^)

    fn get_grad(self, var_id: Int) -> ExTensor:
        """Get the computed gradient for a variable.

        Args:
            var_id: The variable ID

        Returns:
            The gradient tensor
        """
        return self.registry.get_grad(var_id)


struct NoGradContext:
    """Context manager equivalent for disabling gradient computation.

    Used for inference or when you want to temporarily disable gradient tracking.
    Unlike Python's context manager, you must manually call exit().

    Examples:
        var ctx = NoGradContext(tape)
        ctx.enter()  # Disable gradient tracking

        # Operations here are not recorded
        var y = model.forward(x)

        ctx.exit()  # Re-enable gradient tracking
    """

    var tape: UnsafePointer[GradientTape]
    var was_enabled: Bool

    fn __init__(out self, mut tape: GradientTape):
        """Initialize no-grad context.

        Args:
            tape: The gradient tape to manage
        """
        self.tape = UnsafePointer[GradientTape].address_of(tape)
        self.was_enabled = tape.enabled

    fn enter(mut self):
        """Enter no-grad mode (disable recording)."""
        self.tape[].disable()

    fn exit(mut self):
        """Exit no-grad mode (restore previous state)."""
        if self.was_enabled:
            self.tape[].enable()
