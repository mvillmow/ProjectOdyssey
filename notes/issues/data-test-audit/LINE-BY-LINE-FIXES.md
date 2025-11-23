# Line-by-Line Fixes for Data Test Failures

Quick reference with exact line numbers and code snippets for all 15 implementation files.

## 1. shared/data/transforms.mojo

### Error 1.1: @value Decorator (Line 405)

```
Line 405: @value
Line 406: struct Normalize:

FIX: Change line 405 to:
@fieldwise_init
```

### Error 1.2: @value Decorator (Line 501)

```
Line 501: @value
Line 502: struct Denormalize:

FIX: Change line 501 to:
@fieldwise_init
```

### Error 1.3: @value Decorator (Line 629)

```
Line 629: @value
Line 630: struct RandomCrop:

FIX: Change line 629 to:
@fieldwise_init
```

### Error 1.4: @value Decorator (Line 728)

```
Line 728: @value
Line 729: struct RandomRotation:

FIX: Change line 728 to:
@fieldwise_init
```

### Error 1.5: Missing (Copyable, Movable) (Line 54)

```
Line 54: trait Transform:

FIX: Change to:
trait Transform(Copyable, Movable):
```

### Error 1.6: ExTensor.num_elements() (Line 51)

```
Line 51:     var total_elements = data.num_elements()

FIX: Add to ExTensor class OR change to:
    var total_elements = 1
    for dim in data.shape:
        total_elements *= dim
```

### Error 1.7: ExTensor initialization missing dtype (Line 498)

```
Line 498:         return ExTensor(cropped^)

FIX: Change to:
         return ExTensor(cropped^, DType.float32)
```

### Error 1.8: ExTensor initialization missing dtype (Line 402)

```
Line 402:         return ExTensor(cropped^)

FIX: Change to:
         return ExTensor(cropped^, DType.float32)
```

### Error 1.9: ExTensor return missing move semantics (Line 539)

```
Line 539:             return data

FIX: Change to:
             return data^
```

### Error 1.10: ExTensor.num_elements() (Line 681)

```
Line 681:         var total_elements = data.num_elements()

FIX: Implement as in Error 1.6
```

### Error 1.11: ExTensor return missing move semantics (Line 603)

```
Line 603:             return data

FIX: Change to:
             return data^
```

### Error 1.12: ExTensor.num_elements() (Line 546)

```
Line 546:         var total_elements = data.num_elements()

FIX: Implement as in Error 1.6
```

### Error 1.13: Optional value access (Line 452)

```
Line 452:             var pad = self.padding.value()[]

FIX: Change to:
             var pad = self.padding.value()
```

### Error 1.14: Optional value access (Line 473)

```
Line 473:             var pad = self.padding.value()[]

FIX: Change to:
             var pad = self.padding.value()
```

### Error 1.15: ExTensor.num_elements() (Line 795)

```
Line 795:         var total_elements = data.num_elements()

FIX: Implement as in Error 1.6
```

### Error 1.16: ExTensor return missing move semantics (Line 788)

```
Line 788:             return data  # Don't erase

FIX: Change to:
             return data^  # Don't erase
```

### Error 1.17: ExTensor.num_elements() (Line 833)

```
Line 833:         var total_elements = data.num_elements()

FIX: Implement as in Error 1.6
```

### Error 1.18: ExTensor return missing move semantics (Line 825)

```
Line 825:         return data

FIX: Change to:
         return data^
```

### Error 1.19: ExTensor return missing move semantics (Line 833)

```
Line 833:         return data

FIX: Change to:
         return data^
```

### Error 1.20: ExTensor.num_elements() (Line 610)

```
Line 610:         var total_elements = data.num_elements()

FIX: Implement as in Error 1.6
```

---

## 2. shared/data/samplers.mojo

### Error 2.1: @value Decorator (Line 38)

```
Line 38: @value
Line 39: struct SequentialSampler(Sampler):

FIX: Change line 38 to:
@fieldwise_init
```

### Error 2.2: List return without move semantics (Line 83)

```
Line 83:         return indices

FIX: Change to:
         return indices^
```

### Error 2.3: @value Decorator (Line 91)

```
Line 91: @value
Line 92: struct RandomSampler(Sampler):

FIX: Change line 91 to:
@fieldwise_init
```

### Error 2.4: List return without move semantics (Line 164)

```
Line 164:         return indices

FIX: Change to:
         return indices^
```

### Error 2.5: @value Decorator (Line 172)

```
Line 172: @value
Line 173: struct WeightedSampler(Sampler):

FIX: Change line 172 to:
@fieldwise_init
```

### Error 2.6: owned keyword deprecation (Line 186)

```
Line 186:         owned weights: List[Float64],

FIX: Change to:
         var weights: List[Float64],

And in __init__ body, add:
         self.weights = weights^
```

### Error 2.7: Pointer syntax error (Line 205)

```
Line 205:             if w[] < 0:

FIX: Change to (w is already dereferenced):
             if w < 0:
```

### Error 2.8: Pointer syntax error (Line 241)

```
Line 241:             total += w[]

FIX: Change to:
             total += w
```

### Error 2.9: Type inference (Float64 vs Int64) (Line 251)

```
Line 251:             if r < cumsum[i]:

Current: r is Float64, cumsum[i] is Int64
FIX: Ensure both are same type (Float64):
         var cumsum = List[Float64](...)  # Line 248 area
         var r = Float64(random_si64(...))
```

---

## 3. shared/data/text_transforms.mojo

### Error 3.1: @value Decorator (Line 113)

```
Line 113: @value
Line 114: struct RandomSwapWords:

FIX: Change line 113 to:
@fieldwise_init
```

### Error 3.2: @value Decorator (Line 178)

```
Line 178: @value
Line 179: struct RandomDeleteWords:

FIX: Change line 178 to:
@fieldwise_init
```

### Error 3.3: @value Decorator (Line 242)

```
Line 242: @value
Line 243: struct RandomDuplicateWords:

FIX: Change line 242 to:
@fieldwise_init
```

### Error 3.4: @value Decorator (Line 319)

```
Line 319: @value
Line 320: struct SynonymReplacement:

FIX: Change line 319 to:
@fieldwise_init
```

### Error 3.5: Missing (Copyable, Movable) on trait (Line 65)

```
Line 65: trait TextTransform:

FIX: Change to:
trait TextTransform(Copyable, Movable):
```

### Error 3.6: owned keyword + parameter ordering (Line 257)

```
Line 257:     fn __init__(out self, p: Float64 = 0.1, n: Int = 1, owned vocabulary: List[String]):

FIX: Change to:
     fn __init__(out self, vocabulary: List[String], p: Float64 = 0.1, n: Int = 1):
         self.vocabulary = vocabulary^
         self.p = p
         self.n = n
```

### Error 3.7: StringSlice conversion (Line 84)

```
Line 84:             words.append(parts[i])

FIX: Change to:
             words.append(String(parts[i]))
```

### Error 3.8: List return without move semantics (Line 86)

```
Line 86:     return words

FIX: Change to:
     return words^
```

### Error 3.9: owned keyword + parameter ordering (Line 333)

```
Line 333:     fn __init__(out self, p: Float64 = 0.2, owned synonyms: Dict[String, List[String]]):

FIX: Change to:
     fn __init__(out self, synonyms: Dict[String, List[String]], p: Float64 = 0.2):
         self.synonyms = synonyms^
         self.p = p
```

### Error 3.10: List copy without move semantics (Line 314)

```
Line 314:             words = new_words

FIX: Change to:
             words = new_words^
```

### Error 3.11: List copy without move semantics (Line 372)

```
Line 372:                 var syns = self.synonyms[word]

FIX: If this is causing copy error, change to:
             var syns = self.synonyms[word]^
```

---

## 4. shared/data/generic_transforms.mojo

### Error 4.1: @value Decorator (Line 38)

```
Line 38: @value
Line 39: struct IdentityTransform:

FIX: Change line 38 to:
@fieldwise_init
```

### Error 4.2: @value Decorator (Line 70)

```
Line 70: @value
Line 71: struct Compose:

FIX: Change line 70 to:
@fieldwise_init
```

### Error 4.3: @value Decorator (Line 178)

```
Line 178: @value
Line 179: struct Clamp:

FIX: Change line 178 to:
@fieldwise_init
```

### Error 4.4: @value Decorator (Line 242)

```
Line 242: @value
Line 243: struct Normalize:

FIX: Change line 242 to:
@fieldwise_init
```

### Error 4.5: @value Decorator (Line 411)

```
Line 411: @value
Line 412: struct ToFloat32:

FIX: Change line 411 to:
@fieldwise_init
```

### Error 4.6: @value Decorator (Line 445)

```
Line 445: @value
Line 446: struct ToInt32:

FIX: Change line 445 to:
@fieldwise_init
```

### Error 4.7: raise in non-raises context (Line 207)

```
Line 207:             raise Error("min_val must be <= max_val")

FIX: Add 'raises' to function signature:
fn __init__(out self, min_val: Float32, max_val: Float32) raises:
```

### Error 4.8: ExTensor return missing move semantics (Line 62)

```
Line 62:         return data

FIX: Change to:
         return data^
```

### Error 4.9: ExTensor.num_elements() (Line 108)

```
Line 108:         var result_values = List[Float32](capacity=data.num_elements())

FIX: Use calculated value:
         var num_elems = 1
         for dim in data.shape:
             num_elems *= dim
         var result_values = List[Float32](capacity=num_elems)
```

### Error 4.10: ExTensor.num_elements() (Line 110)

```
Line 110:         for i in range(data.num_elements()):

FIX: Change to:
         var num_elems = 1
         for dim in data.shape:
             num_elems *= dim
         for i in range(num_elems):
```

### Error 4.11: ExTensor.num_elements() (Line 221)

```
Line 221:         var result_values = List[Float32](capacity=data.num_elements())

FIX: As in Error 4.9
```

### Error 4.12: ExTensor.num_elements() (Line 223)

```
Line 223:         for i in range(data.num_elements()):

FIX: As in Error 4.10
```

### Error 4.13: ExTensor.num_elements() (Line 278)

```
Line 278:         print("  Elements:", data.num_elements())

FIX: As in Error 4.10
```

### Error 4.14: ExTensor.num_elements() (Line 281)

```
Line 281:         if data.num_elements() > 0:

FIX: As in Error 4.10
```

### Error 4.15: ExTensor.num_elements() (Line 437)

```
Line 437:         var result_values = List[Float32](capacity=data.num_elements())

FIX: As in Error 4.9
```

### Error 4.16: ExTensor.num_elements() (Line 439)

```
Line 439:         for i in range(data.num_elements()):

FIX: As in Error 4.10
```

### Error 4.17: ExTensor.num_elements() (Line 473)

```
Line 473:         var result_values = List[Float32](capacity=data.num_elements())

FIX: As in Error 4.9
```

### Error 4.18: ExTensor.num_elements() (Line 475)

```
Line 475:         for i in range(data.num_elements()):

FIX: As in Error 4.10
```

---

## 5. shared/data/loaders.mojo

### Error 5.1: Dynamic trait (Line 60)

```
Line 60:     var dataset: Dataset

FIX: Change struct signature to use generic:
struct BaseLoader[DatasetType: Dataset]:
    var dataset: DatasetType
```

### Error 5.2: owned keyword deprecation (Line 40)

```
Line 40:     fn __moveinit__(out self, owned existing: Self):

FIX: Change to:
     fn __moveinit__(out self, deinit existing: Self):
```

### Error 5.3: Struct inheritance (Line 106)

```
Line 106: struct BatchLoader(BaseLoader):

FIX: Change to use composition:
struct BatchLoader:
    var base: BaseLoader
    var shuffle: Bool
```

---

## 6. shared/data/datasets.mojo

### Error 6.1: Dict type constraint (Line 129)

```
Line 129:     var _cache: Dict[Int, Tuple[ExTensor, ExTensor]]

FIX: ExTensor needs to be Copyable/Movable. Either:
1. Make ExTensor Copyable, Movable
2. Use List instead of Dict for caching
3. Use custom cache structure
```

---

## 7. tests/shared/data/test_datasets.mojo

### Error 7.1: Missing main function

```
End of file: No main() function

FIX: Add at end:
fn main():
    print("Running dataset tests...")
    test_tensor_dataset()
    test_file_dataset()
    print("All tests passed!")
```

---

## 8. tests/shared/data/test_loaders.mojo

### Error 8.1: Missing main function

```
End of file: No main() function

FIX: Add at end:
fn main():
    print("Running loader tests...")
    test_batch_loader()
    test_parallel_loader()
    print("All tests passed!")
```

---

## 9. tests/shared/data/test_transforms.mojo

### Error 9.1: Missing main function

```
End of file: No main() function

FIX: Add at end:
fn main():
    print("Running transform tests...")
    test_pipeline()
    test_augmentations()
    print("All tests passed!")
```

---

## 10. tests/shared/data/datasets/test_tensor_dataset.mojo

### Error 10.1: Missing Tensor module (Line 14)

```
Line 14: from tensor import Tensor

FIX: Change to:
# Removed - use ExTensor instead
```

### Error 10.2: TensorDataset not exported (Line 13)

```
Line 13: from shared.data.datasets import TensorDataset

FIX: Check if TensorDataset exists. If not, either:
1. Implement it in datasets.mojo
2. Use ExTensorDataset instead
```

---

## 11. tests/shared/data/datasets/test_file_dataset.mojo

### Error 11.1: str() function missing (Line 104)

```
Line 104:         file_paths.append("/path/to/image_" + str(i) + ".jpg")

FIX: Add helper function and use:
         file_paths.append("/path/to/image_" + int_to_string(i) + ".jpg")

fn int_to_string(value: Int) -> String:
    # Implementation (see FIXME-GUIDE.md)
```

### Error 11.2: str() function missing (Line 169)

```
Line 169:         file_paths.append("/images/img" + str(i) + ".jpg")

FIX: Same as Error 11.1
```

---

## 12. tests/shared/data/loaders/test_batch_loader.mojo

### Error 12.1: Struct inheritance (Line 106 in loaders.mojo)

```
Caused by: struct BatchLoader(BaseLoader) in loaders.mojo

FIX: See Error 5.3 above
```

### Error 12.2: TensorDataset missing (Line 13)

```
Line 13: from shared.data.datasets import TensorDataset

FIX: See Error 10.2 above
```

---

## 13. tests/shared/data/transforms/test_augmentations.mojo

### Error 13.1: Missing Tensor module (Line 19)

```
Line 19: from tensor import Tensor

FIX: Change to:
# Removed - use ExTensor instead
```

---

## 14. tests/shared/data/transforms/test_text_augmentations.mojo

### Error 14.1: str() function missing (Line 328, 353, 369)

```
Line 328: synonyms["quick"] = quick_syns

FIX: If copy error, add move semantics:
     synonyms["quick"] = quick_syns^
```

---

## 15. tests/shared/data/transforms/test_generic_transforms.mojo

### Error 15.1: str() function missing (Line 387, 388, 389, etc.)

```
Line 387:     assert_equal(int(result[0]), 1)

FIX: Change to:
     assert_equal(result[0] as Int, 1)

Or use custom conversion:
fn float_to_int(value: Float32) -> Int:
    return Int(value)
```

---

## Summary Table: Quick Fixes

| File | Line | Error | Fix |
|------|------|-------|-----|
| transforms.mojo | 54 | trait | Add (Copyable, Movable) |
| transforms.mojo | 405 | @value | Replace with @fieldwise_init |
| transforms.mojo | 501 | @value | Replace with @fieldwise_init |
| transforms.mojo | 629 | @value | Replace with @fieldwise_init |
| transforms.mojo | 728 | @value | Replace with @fieldwise_init |
| transforms.mojo | 402, 498 | dtype | Add DType.float32 parameter |
| transforms.mojo | 51, 546, 681, 795, 610 | num_elements | Add method to ExTensor |
| transforms.mojo | 539, 603, 788, 825, 833 | move semantics | Add ^ to returns |
| samplers.mojo | 38, 91, 172 | @value | Replace with @fieldwise_init |
| samplers.mojo | 83, 164 | move semantics | Add ^ to returns |
| samplers.mojo | 186 | owned | Replace with var |
| text_transforms.mojo | 65 | trait | Add (Copyable, Movable) |
| text_transforms.mojo | 113, 178, 242, 319 | @value | Replace with @fieldwise_init |
| text_transforms.mojo | 257, 333 | owned+ordering | Reorder params, use var |
| generic_transforms.mojo | 38, 70, 178, 242, 411, 445 | @value | Replace with @fieldwise_init |
| loaders.mojo | 40 | owned | Replace with deinit |
| loaders.mojo | 60 | dynamic trait | Use generic type param |
| loaders.mojo | 106 | inheritance | Use composition |
| datasets.mojo | 129 | Dict constraint | Use Copyable/Movable |

---

**Total Changes Required**: ~80 distinct fixes across 7 implementation files and 8 test files
**Estimated Implementation Time**: 4-6 hours
**Priority**: P1 - Blocks comprehensive test suite (38 tests)

Last updated: 2025-11-22
