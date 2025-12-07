"""Example: First Model - Complete Training Script

This example shows a complete training script for a digit classifier on MNIST.

Usage:
    pixi run mojo run examples/getting-started/first_model_train.mojo

See documentation: docs/getting-started/first_model.md

FIXME: This example does not compile. Issues:
1. API Mismatch: Imports Trainer, SGD, CrossEntropyLoss classes that do
   NOT exist in shared.training. The actual library provides:
   - Callback interface and schedulers
   - No high-level Trainer class
   - No optimizer classes (SGD, etc.)
   - No built-in loss function classes

2. Missing callbacks: EarlyStopping and ModelCheckpoint are exported but
   need to be used differently (they exist in shared.training.callbacks)

3. Module dependencies: Imports from 'model' and 'prepare_data' modules
   that don't exist. Also depends on first_model_model.mojo which doesn't
   compile.

4. Data loading issue: BatchLoader has struct inheritance issues in the
   library itself (cascading dependency failure)

This example needs complete redesign to use actual functional API and
would require implementing helper modules for data preparation.
"""

# FIXME: These imports don't exist in shared library
# from shared.training import Trainer, SGD, CrossEntropyLoss
# from shared.training.callbacks import EarlyStopping, ModelCheckpoint
# from shared.data import BatchLoader
# from model import DigitClassifier
# from prepare_data import prepare_mnist


fn main() raises:
    """Train the digit classifier."""

    print("=" * 50)
    print("Training Digit Classifier")
    print("=" * 50)

    # Step 1: Load data
    var train_data, test_data = prepare_mnist()

    # Step 2: Create data loaders
    var train_loader = BatchLoader(
        train_data, batch_size=32, shuffle=True, drop_last=True
    )

    var test_loader = BatchLoader(test_data, batch_size=32, shuffle=False)

    # Step 3: Create model
    var model = DigitClassifier()
    print("\nModel architecture:")
    print(model.model.summary())

    # Step 4: Configure optimizer
    var optimizer = SGD(learning_rate=0.01, momentum=0.9)

    # Step 5: Configure loss function
    var loss_fn = CrossEntropyLoss()

    # Step 6: Create trainer
    var trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)

    # Step 7: Add callbacks
    trainer.add_callback(EarlyStopping(patience=3, min_delta=0.001))
    trainer.add_callback(
        ModelCheckpoint(filepath="best_model.mojo", save_best_only=True)
    )

    # Step 8: Train the model
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=10,
        verbose=True,
    )

    print("\nTraining complete!")
