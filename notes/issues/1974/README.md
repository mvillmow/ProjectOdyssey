# Issue #1974: Add Pipeline type alias for transforms

## Objective

Add Pipeline type alias that points to Compose[T] in the transforms module and export it from the data package.

## Deliverables

- Added `alias Pipeline[T: Transform & Copyable & Movable] = Compose[T]` in `shared/data/transforms.mojo` (line 133)
- Added Pipeline to exports in `shared/data/__init__.mojo` (line 72)

## Success Criteria

- [x] Pipeline type alias defined in transforms.mojo after Compose struct
- [x] Pipeline exported from shared.data package
- [x] Tests can import Pipeline from shared.data.transforms
- [x] Type alias properly parameterized as Pipeline[T]

## References

- Compose struct definition: `shared/data/transforms.mojo:86-130`
- Data package exports: `shared/data/__init__.mojo:68-81`

## Implementation Notes

The tests imported Pipeline from shared.data.transforms but the type alias didn't exist. Only the Compose[T] generic struct was defined. This alias provides a semantic convenience name for users who prefer "Pipeline" terminology over "Compose".

The type alias uses the same generic constraints as Compose to ensure proper type safety and trait implementation.
