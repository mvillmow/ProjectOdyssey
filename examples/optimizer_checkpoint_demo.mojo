"""Demo of optimizer state save/load functionality.

This example demonstrates how to save and restore optimizer state
for checkpoint/resume training workflows.

Note: This is a simplified demo that tests the save/load mechanics
without running full training loops. The key point is demonstrating
that hyperparameters can be saved and restored.
"""

from shared.autograd.optimizers import SGD, Adam, AdaGrad, RMSprop


fn main() raises:
    print("=== Optimizer Checkpoint Demo ===\n")

    # Test SGD with momentum
    print("Testing SGD...")
    var sgd = SGD(learning_rate=0.01, momentum=0.9)
    print("  Original learning rate:", sgd.learning_rate)
    print("  Original momentum:", sgd.momentum)

    # Save state
    print("  Saving SGD state to checkpoints/sgd/...")
    sgd.save_state("checkpoints/sgd")
    print("  ✓ Saved")

    # Create new optimizer with different parameters and load state
    var sgd_loaded = SGD(learning_rate=0.001, momentum=0.0)  # Different params
    print("  Loading SGD state...")
    sgd_loaded.load_state("checkpoints/sgd")
    print("  ✓ Loaded")

    print("  Restored learning rate:", sgd_loaded.learning_rate)
    print("  Restored momentum:", sgd_loaded.momentum)

    # Verify values match
    if (
        sgd_loaded.learning_rate == sgd.learning_rate
        and sgd_loaded.momentum == sgd.momentum
    ):
        print("  ✓ SGD checkpoint test PASSED!\n")
    else:
        print("  ✗ SGD checkpoint test FAILED!\n")
        return

    # Test Adam
    print("Testing Adam...")
    var adam = Adam(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    )
    print("  Original learning rate:", adam.learning_rate)
    print("  Original beta1:", adam.beta1)
    print("  Original beta2:", adam.beta2)
    print("  Original epsilon:", adam.epsilon)
    print("  Original weight_decay:", adam.weight_decay)

    # Save state
    print("  Saving Adam state to checkpoints/adam/...")
    adam.save_state("checkpoints/adam")
    print("  ✓ Saved")

    # Create new optimizer and load state
    var adam_loaded = Adam()  # Use defaults
    print("  Loading Adam state...")
    adam_loaded.load_state("checkpoints/adam")
    print("  ✓ Loaded")

    print("  Restored learning rate:", adam_loaded.learning_rate)
    print("  Restored beta1:", adam_loaded.beta1)
    print("  Restored beta2:", adam_loaded.beta2)

    # Verify values match
    if (
        adam_loaded.learning_rate == adam.learning_rate
        and adam_loaded.beta1 == adam.beta1
    ):
        print("  ✓ Adam checkpoint test PASSED!\n")
    else:
        print("  ✗ Adam checkpoint test FAILED!\n")
        return

    # Test AdaGrad
    print("Testing AdaGrad...")
    var adagrad = AdaGrad(
        learning_rate=0.01, epsilon=1e-10, weight_decay=0.0001
    )
    print("  Original learning rate:", adagrad.learning_rate)
    print("  Original epsilon:", adagrad.epsilon)
    print("  Original weight_decay:", adagrad.weight_decay)

    # Save state
    print("  Saving AdaGrad state to checkpoints/adagrad/...")
    adagrad.save_state("checkpoints/adagrad")
    print("  ✓ Saved")

    # Create new optimizer and load state
    var adagrad_loaded = AdaGrad(learning_rate=0.001)  # Different params
    print("  Loading AdaGrad state...")
    adagrad_loaded.load_state("checkpoints/adagrad")
    print("  ✓ Loaded")

    print("  Restored learning rate:", adagrad_loaded.learning_rate)
    print("  Restored epsilon:", adagrad_loaded.epsilon)

    # Verify values match
    if (
        adagrad_loaded.learning_rate == adagrad.learning_rate
        and adagrad_loaded.epsilon == adagrad.epsilon
    ):
        print("  ✓ AdaGrad checkpoint test PASSED!\n")
    else:
        print("  ✗ AdaGrad checkpoint test FAILED!\n")
        return

    # Test RMSprop
    print("Testing RMSprop...")
    var rmsprop = RMSprop(
        learning_rate=0.01,
        alpha=0.99,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.9,
    )
    print("  Original learning rate:", rmsprop.learning_rate)
    print("  Original alpha:", rmsprop.alpha)
    print("  Original epsilon:", rmsprop.epsilon)
    print("  Original momentum:", rmsprop.momentum)

    # Save state
    print("  Saving RMSprop state to checkpoints/rmsprop/...")
    rmsprop.save_state("checkpoints/rmsprop")
    print("  ✓ Saved")

    # Create new optimizer and load state
    var rmsprop_loaded = RMSprop()  # Use defaults
    print("  Loading RMSprop state...")
    rmsprop_loaded.load_state("checkpoints/rmsprop")
    print("  ✓ Loaded")

    print("  Restored learning rate:", rmsprop_loaded.learning_rate)
    print("  Restored alpha:", rmsprop_loaded.alpha)
    print("  Restored momentum:", rmsprop_loaded.momentum)

    # Verify values match
    if (
        rmsprop_loaded.learning_rate == rmsprop.learning_rate
        and rmsprop_loaded.alpha == rmsprop.alpha
    ):
        print("  ✓ RMSprop checkpoint test PASSED!\n")
    else:
        print("  ✗ RMSprop checkpoint test FAILED!\n")
        return

    print("=== All optimizer checkpoint tests PASSED! ===\n")
    print("Checkpoint directories created:")
    print("  - checkpoints/sgd/")
    print("  - checkpoints/adam/")
    print("  - checkpoints/adagrad/")
    print("  - checkpoints/rmsprop/")
    print("\nEach directory contains:")
    print("  - metadata.txt (hyperparameters)")
    print("  - *.weights files (optimizer state tensors, if any)")
    print("\nYou can inspect the checkpoint files to see the saved state.")
