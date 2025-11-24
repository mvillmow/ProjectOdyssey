"""Example: Custom Layers - Multi-Head Attention

This example implements multi-head self-attention mechanism.

Usage:
    pixi run mojo run examples/custom-layers/attention_layer.mojo

See documentation: docs/advanced/custom-layers.md
"""

from shared.core import Module, Linear, Tensor


struct MultiHeadAttention(Module):
    """Multi-head self-attention mechanism.

    Used in Transformers and other attention-based models.
    """
    var num_heads: Int
    var head_dim: Int
    var embed_dim: Int

    var q_proj: Linear  # Query projection
    var k_proj: Linear  # Key projection
    var v_proj: Linear  # Value projection
    var out_proj: Linear  # Output projection

    fn __init__(mut self, embed_dim: Int, num_heads: Int):
        """Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projection layers
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    fn forward(mut self, x: Tensor) -> Tensor:
        """Compute multi-head attention.

        Args:
            x: Input tensor, shape [batch, seq_len, embed_dim].

        Returns:
            Output tensor, shape [batch, seq_len, embed_dim].
        """
        var batch_size = x.shape()[0]
        var seq_len = x.shape()[1]

        # Project to Q, K, V
        var q = self.q_proj.forward(x)
        var k = self.k_proj.forward(x)
        var v = self.v_proj.forward(x)

        # Reshape for multi-head attention
        # [batch, seq_len, embed_dim] -> [batch, num_heads, seq_len, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        var scores = q @ k.transpose(-2, -1) / sqrt(Float64(self.head_dim))
        var attn_weights = softmax(scores, dim=-1)
        var attn_output = attn_weights @ v

        # Reshape back
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        return self.out_proj.forward(attn_output)

    fn parameters(mut self) -> List[Tensor]:
        var params = List[Tensor]()
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params


fn main() raises:
    """Demonstrate multi-head attention."""

    # Create attention layer
    var attention = MultiHeadAttention(embed_dim=512, num_heads=8)

    # Test with sample input
    var input = Tensor.randn(32, 100, 512)  # [batch, seq_len, embed_dim]
    print("Input shape:", input.shape())

    var output = attention.forward(input)
    print("Output shape:", output.shape())
    print("Expected: same as input [32, 100, 512]")

    # Show parameter count
    var params = attention.parameters()
    print("Number of parameter tensors:", len(params))

    print("\nMulti-head attention example complete!")
