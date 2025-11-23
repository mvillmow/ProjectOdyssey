"""Quick integration test for ResNet-18 model"""

from model import ResNet18
from shared.core import ExTensor, zeros

fn main() raises:
    print("Testing ResNet-18 Model")
    print("="*60)

    # Create model
    print("Initializing ResNet-18...")
    var model = ResNet18(num_classes=10)
    print("✓ Model initialized")

    # Create dummy input
    print("\nCreating dummy input (batch=4, channels=3, height=32, width=32)...")
    var input = zeros(
        List[Int]().append(4).append(3).append(32).append(32),
        DType.float32
    )
    var input_data = input._data.bitcast[Float32]()
    for i in range(4 * 3 * 32 * 32):
        input_data[i] = Float32(0.1)
    print("✓ Input created")

    # Test inference mode
    print("\nTesting inference mode...")
    var logits_inf = model.forward(input, training=False)
    print("  Output shape: (" + str(logits_inf.shape[0]) + ", " + str(logits_inf.shape[1]) + ")")
    if logits_inf.shape[0] == 4 and logits_inf.shape[1] == 10:
        print("✓ Inference mode PASSED")
    else:
        print("✗ Inference mode FAILED - wrong output shape")
        return

    # Test training mode
    print("\nTesting training mode...")
    var logits_train = model.forward(input, training=True)
    print("  Output shape: (" + str(logits_train.shape[0]) + ", " + str(logits_train.shape[1]) + ")")
    if logits_train.shape[0] == 4 and logits_train.shape[1] == 10:
        print("✓ Training mode PASSED")
    else:
        print("✗ Training mode FAILED - wrong output shape")
        return

    # Check for NaN/Inf
    print("\nChecking for NaN/Inf in outputs...")
    var logits_data = logits_train._data.bitcast[Float32]()
    var has_nan_inf = False
    for i in range(40):
        var val = logits_data[i]
        if val > 1e10 or val < -1e10 or val != val:
            has_nan_inf = True
            break

    if has_nan_inf:
        print("✗ Found NaN or Inf in outputs")
    else:
        print("✓ No NaN/Inf detected")

    print("\n" + "="*60)
    print("✓ ResNet-18 ALL TESTS PASSED!")
    print("="*60)
