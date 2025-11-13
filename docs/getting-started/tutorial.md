# Tutorial

## Example 1: Basic Usage

Here's how to create a simple component:

```python
# Create a new component instance
component = MyComponent(name="example")

# Configure the component
component.set_option("verbose", True)

# Process some data
result = component.process(data)
```

This example demonstrates the basic workflow: create, configure, and process.

## Example 2: Advanced Pattern

For more complex scenarios:

```python
# Advanced usage with context manager
with MyComponent(name="advanced") as comp:
    comp.configure(options)
    result = comp.process_batch(items)
```

The context manager ensures proper cleanup.
