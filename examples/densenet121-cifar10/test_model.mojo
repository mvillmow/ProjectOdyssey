"""Quick integration test for DenseNet-121 model"""

from model import DenseNet121
from shared.core import ExTensor, zeros


fn main() raises:
    print("Testing DenseNet-121 Model")
    print("=" * 60)

    print("Initializing DenseNet-121...")
    var model = DenseNet121(num_classes=10)
    print("✓ Model initialized")

    print(
        "\nCreating dummy input (batch=1, channels=3, height=32, width=32)..."
    )
    print("  (Using batch=1 due to memory constraints)")
    var input = zeros(
        List[Int]().append(1).append(3).append(32).append(32), DType.float32
    )
    var input_data = input._data.bitcast[Float32]()
    for i in range(1 * 3 * 32 * 32):
        input_data[i] = Float32(0.1)
    print("✓ Input created")

    print("\nTesting inference mode...")
    var logits_inf = model.forward(input, training=False)
    print(
        "  Output shape: ("
        + String(logits_inf.shape()[0])
        + ", "
        + String(logits_inf.shape()[1])
        + ")"
    )
    if logits_inf.shape()[0] == 1 and logits_inf.shape()[1] == 10:
        print("✓ Inference mode PASSED")
    else:
        print("✗ Inference mode FAILED")
        return

    print("\nTesting training mode...")
    var logits_train = model.forward(input, training=True)
    print(
        "  Output shape: ("
        + String(logits_train.shape()[0])
        + ", "
        + String(logits_train.shape()[1])
        + ")"
    )
    if logits_train.shape()[0] == 1 and logits_train.shape()[1] == 10:
        print("✓ Training mode PASSED")
    else:
        print("✗ Training mode FAILED")
        return

    print("\n" + "=" * 60)
    print("✓ DenseNet-121 ALL TESTS PASSED!")
    print("=" * 60)
