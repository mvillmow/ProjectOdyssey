# Issue #416: [Package] Text Augmentations - Integration and Distribution

## Objective

Package and integrate the text augmentation transforms (`text_transforms.mojo`) into the shared data utilities module, providing clear installation guides, API reference documentation, and usage examples for the 4 implemented augmentations.

## Deliverables

- Module packaging configuration for `text_transforms.mojo`
- Public API surface definition and exports
- Installation and integration guide
- Complete API reference for all 4 augmentations
- Usage examples and best practices
- Integration with existing data pipelines

## Success Criteria

- [ ] `text_transforms.mojo` properly packaged as part of shared data utilities
- [ ] Public API clearly defined with appropriate exports
- [ ] Installation guide tested and verified
- [ ] API reference covers all public interfaces
- [ ] Usage examples demonstrate common patterns
- [ ] Integration with image augmentations documented

## Packaging Structure

### Module Organization

```text
shared/
└── data/
    ├── __init__.mojo              # Main data module exports
    ├── text_transforms.mojo        # Text augmentation transforms (426 lines)
    ├── transforms.mojo             # Image/tensor transforms
    └── ...
```

### Public API Surface

The `text_transforms` module exports:

1. **Base Interface**:
   - `TextTransform` - Base trait for text transforms

2. **Helper Functions**:
   - `split_words(text: String) -> List[String]` - Split text into words
   - `join_words(words: List[String]) -> String` - Join words into text

3. **Augmentation Transforms**:
   - `RandomSwap` - Randomly swap word positions
   - `RandomDeletion` - Randomly delete words
   - `RandomInsertion` - Insert random words from vocabulary
   - `RandomSynonymReplacement` - Replace words with synonyms

4. **Composition**:
   - `TextCompose` - Compose multiple text transforms
   - `TextPipeline` - Alias for TextCompose

## Installation Guide

### Prerequisites

- Mojo SDK v0.25.7 or later
- ML Odyssey repository cloned

### Integration Steps

1. **Import the module**:

```mojo
from shared.data.text_transforms import (
    TextTransform,
    RandomSwap,
    RandomDeletion,
    RandomInsertion,
    RandomSynonymReplacement,
    TextCompose,
    TextPipeline,
)
```

2. **Create transforms**:

```mojo
# Single transform
var swap = RandomSwap(p=0.15, n=2)
var text = "the quick brown fox"
var result = swap(text)

# Composed pipeline
var vocab = List[String]()
vocab.append("very")
vocab.append("really")

var transforms = List[TextTransform]()
transforms.append(RandomSwap(0.15, 2))
transforms.append(RandomInsertion(0.1, 1, vocab))

var pipeline = TextPipeline(transforms)
var augmented = pipeline(text)
```

3. **Use in data loading pipelines**:

```mojo
# Integration with data loaders (future work)
# var dataset = TextDataset(augmentations=pipeline)
```

## API Reference

### TextTransform (Trait)

Base interface for text transforms.

```mojo
trait TextTransform:
    fn __call__(self, text: String) raises -> String:
        """Apply the transform to text.

        Args:
            text: Input text string.

        Returns:
            Transformed text string.

        Raises:
            Error if transform cannot be applied.
        """
        ...
```

### RandomSwap

Randomly swap positions of word pairs.

```mojo
@value
struct RandomSwap(TextTransform):
    var p: Float64  # Probability of performing swap
    var n: Int      # Number of swaps to perform

    fn __init__(out self, p: Float64 = 0.15, n: Int = 2):
        """Create random swap transform.

        Args:
            p: Probability of performing each swap (0.0 to 1.0).
            n: Number of swap operations to attempt.
        """
        ...

    fn __call__(self, text: String) raises -> String:
        """Randomly swap word pairs in text."""
        ...
```

**Example**:

```mojo
var swap = RandomSwap(p=0.15, n=2)
var text = "the quick brown fox"
var result = swap(text)  # "quick the brown fox" (example)
```

**Use Cases**:
- Data augmentation for text classification
- Creating variations for robustness testing
- Training models to handle word order variations

### RandomDeletion

Randomly delete words from text while ensuring at least one word remains.

```mojo
@value
struct RandomDeletion(TextTransform):
    var p: Float64  # Probability of deleting each word

    fn __init__(out self, p: Float64 = 0.1):
        """Create random deletion transform.

        Args:
            p: Probability of deleting each word (0.0 to 1.0).
        """
        ...

    fn __call__(self, text: String) raises -> String:
        """Randomly delete words from text.

        Ensures at least one word remains even if all would be deleted.
        """
        ...
```

**Example**:

```mojo
var delete = RandomDeletion(p=0.1)
var text = "the quick brown fox"
var result = delete(text)  # "quick brown fox" (example)
```

**Use Cases**:
- Simulating missing words in text
- Training models robust to incomplete input
- Creating shorter text variations

### RandomInsertion

Insert random words from vocabulary into text.

```mojo
@value
struct RandomInsertion(TextTransform):
    var p: Float64              # Probability of insertion
    var n: Int                  # Number of words to insert
    var vocabulary: List[String]  # Words to insert from

    fn __init__(out self, p: Float64 = 0.1, n: Int = 1, owned vocabulary: List[String]):
        """Create random insertion transform.

        Args:
            p: Probability of performing insertion (0.0 to 1.0).
            n: Number of words to insert.
            vocabulary: List of words to choose from for insertion.
        """
        ...

    fn __call__(self, text: String) raises -> String:
        """Insert random words from vocabulary into text."""
        ...
```

**Example**:

```mojo
var vocab = List[String]()
vocab.append("quick")
vocab.append("lazy")

var insert = RandomInsertion(p=0.1, n=1, vocab)
var text = "the brown fox"
var result = insert(text)  # "the quick brown fox" (example)
```

**Use Cases**:
- Increasing lexical diversity
- Simulating noisy text input
- Data augmentation for sequence models

### RandomSynonymReplacement

Replace random words with synonyms from dictionary.

```mojo
@value
struct RandomSynonymReplacement(TextTransform):
    var p: Float64                           # Probability of replacing each word
    var synonyms: Dict[String, List[String]]  # Synonym dictionary

    fn __init__(out self, p: Float64 = 0.2, owned synonyms: Dict[String, List[String]]):
        """Create random synonym replacement transform.

        Args:
            p: Probability of replacing each word (0.0 to 1.0).
            synonyms: Dictionary mapping words to lists of synonyms.
        """
        ...

    fn __call__(self, text: String) raises -> String:
        """Replace random words with synonyms."""
        ...
```

**Example**:

```mojo
var synonyms = Dict[String, List[String]]()
var quick_syns = List[String]()
quick_syns.append("fast")
quick_syns.append("rapid")
synonyms["quick"] = quick_syns

var replace = RandomSynonymReplacement(p=0.2, synonyms)
var text = "the quick fox"
var result = replace(text)  # "the fast fox" (example)
```

**Use Cases**:
- Semantic variation while preserving meaning
- Training models on paraphrased text
- Improving model robustness to word choice

### TextCompose / TextPipeline

Compose multiple text transforms sequentially.

```mojo
@value
struct TextCompose(TextTransform):
    var transforms: List[TextTransform]

    fn __init__(out self, owned transforms: List[TextTransform]):
        """Create composition of text transforms.

        Args:
            transforms: List of text transforms to apply in order.
        """
        ...

    fn __call__(self, text: String) raises -> String:
        """Apply all text transforms sequentially."""
        ...

    fn __len__(self) -> Int:
        """Return number of transforms."""
        ...

    fn append(inout self, transform: TextTransform):
        """Add a transform to the pipeline."""
        ...

# Type alias for more intuitive naming
alias TextPipeline = TextCompose
```

**Example**:

```mojo
var vocab = List[String]()
vocab.append("very")

var synonyms = Dict[String, List[String]]()
var quick_syns = List[String]()
quick_syns.append("fast")
synonyms["quick"] = quick_syns

var transforms = List[TextTransform]()
transforms.append(RandomSynonymReplacement(0.3, synonyms))
transforms.append(RandomInsertion(0.2, 1, vocab))
transforms.append(RandomSwap(0.15, 2))
transforms.append(RandomDeletion(0.1))

var pipeline = TextPipeline(transforms)
var text = "the quick brown fox"
var augmented = pipeline(text)
```

**Use Cases**:
- Building complex augmentation pipelines
- Applying multiple augmentations in sequence
- Reusing augmentation configurations

## Usage Examples

### Basic Text Augmentation

```mojo
from shared.data.text_transforms import RandomSwap, RandomDeletion

fn augment_text(text: String) raises -> String:
    """Apply basic text augmentation."""
    var swap = RandomSwap(p=0.15, n=2)
    var delete = RandomDeletion(p=0.1)

    var result = swap(text)
    result = delete(result)

    return result

fn main() raises:
    var original = "the quick brown fox jumps over the lazy dog"
    var augmented = augment_text(original)
    print("Original: " + original)
    print("Augmented: " + augmented)
```

### Multi-Transform Pipeline

```mojo
from shared.data.text_transforms import (
    TextPipeline,
    RandomSwap,
    RandomDeletion,
    RandomInsertion,
    RandomSynonymReplacement,
)

fn create_augmentation_pipeline() -> TextPipeline:
    """Create a comprehensive text augmentation pipeline."""
    # Build vocabulary for insertion
    var vocab = List[String]()
    vocab.append("very")
    vocab.append("really")
    vocab.append("extremely")

    # Build synonym dictionary
    var synonyms = Dict[String, List[String]]()

    var quick_syns = List[String]()
    quick_syns.append("fast")
    quick_syns.append("rapid")
    quick_syns.append("speedy")
    synonyms["quick"] = quick_syns

    var lazy_syns = List[String]()
    lazy_syns.append("slow")
    lazy_syns.append("sluggish")
    synonyms["lazy"] = lazy_syns

    # Compose pipeline
    var transforms = List[TextTransform]()
    transforms.append(RandomSynonymReplacement(0.2, synonyms))
    transforms.append(RandomInsertion(0.1, 1, vocab))
    transforms.append(RandomSwap(0.15, 2))
    transforms.append(RandomDeletion(0.05))

    return TextPipeline(transforms)

fn main() raises:
    var pipeline = create_augmentation_pipeline()
    var text = "the quick brown fox jumps over the lazy dog"

    # Generate multiple augmented versions
    for i in range(5):
        var augmented = pipeline(text)
        print("Version " + String(i) + ": " + augmented)
```

### Batch Processing

```mojo
from shared.data.text_transforms import TextPipeline, RandomSwap

fn augment_batch(texts: List[String], pipeline: TextPipeline) raises -> List[String]:
    """Augment a batch of text samples."""
    var augmented = List[String]()

    for i in range(len(texts)):
        var result = pipeline(texts[i])
        augmented.append(result)

    return augmented

fn main() raises:
    var texts = List[String]()
    texts.append("the quick brown fox")
    texts.append("jumps over the lazy dog")
    texts.append("machine learning is awesome")

    var transforms = List[TextTransform]()
    transforms.append(RandomSwap(0.15, 1))
    var pipeline = TextPipeline(transforms)

    var augmented = augment_batch(texts, pipeline)

    for i in range(len(augmented)):
        print("Original: " + texts[i])
        print("Augmented: " + augmented[i])
        print()
```

## Integration with Existing Pipelines

### Combining Image and Text Augmentations

While image and text transforms use different base traits (`Transform` for tensors, `TextTransform` for strings), they can be used together in multi-modal pipelines:

```mojo
from shared.data.transforms import RandomHorizontalFlip, RandomRotation
from shared.data.text_transforms import RandomSwap, RandomDeletion

struct MultiModalSample:
    var image: Tensor[DType.float32]
    var caption: String

fn augment_multimodal(sample: MultiModalSample) raises -> MultiModalSample:
    """Augment both image and text components."""
    # Image augmentation
    var img_flip = RandomHorizontalFlip(p=0.5)
    var img_rotate = RandomRotation(max_degrees=15.0)

    var augmented_image = img_flip(sample.image)
    augmented_image = img_rotate(augmented_image)

    # Text augmentation
    var text_swap = RandomSwap(p=0.15, n=1)
    var text_delete = RandomDeletion(p=0.05)

    var augmented_caption = text_swap(sample.caption)
    augmented_caption = text_delete(augmented_caption)

    return MultiModalSample(augmented_image, augmented_caption)
```

## Best Practices

### 1. Probability Selection

- **Conservative augmentation** (p=0.05-0.15): Minimal changes, preserves meaning well
- **Moderate augmentation** (p=0.15-0.30): Balanced variation, good for most tasks
- **Aggressive augmentation** (p=0.30-0.50): Significant changes, use carefully

### 2. Pipeline Ordering

Recommended order for combining augmentations:

1. **Synonym replacement** - Preserves meaning best
2. **Insertion** - Adds controlled variation
3. **Swap** - Changes order but keeps all words
4. **Deletion** - Most destructive, apply last

### 3. Vocabulary Construction

For `RandomInsertion`:

- Use domain-specific vocabulary relevant to your task
- Include common adjectives, adverbs, and connecting words
- Avoid highly semantic words that could change meaning drastically

### 4. Synonym Dictionary Design

For `RandomSynonymReplacement`:

- Use true synonyms, not just related words
- Consider context and part-of-speech
- Test synonym quality on sample texts
- Start with small, high-quality synonym sets

### 5. Testing and Validation

- Always inspect augmented samples manually
- Verify that augmentations preserve semantic meaning
- Test edge cases (empty strings, single words)
- Measure impact on model performance

## Limitations and Considerations

### Current Limitations

1. **Basic Tokenization**: Uses simple space-based word splitting
   - Doesn't handle punctuation specially
   - No stemming or lemmatization
   - English-centric approach

2. **No Semantic Understanding**: Augmentations are word-level only
   - May produce ungrammatical sentences
   - Doesn't consider context or syntax
   - Simple synonym dictionary (not embeddings-based)

3. **No Advanced NLP Features**:
   - No part-of-speech tagging
   - No named entity recognition
   - No dependency parsing

### Future Enhancements

Potential improvements for future versions:

1. **Advanced Tokenization**: Integrate with NLP libraries for better word splitting
2. **Contextual Synonyms**: Use word embeddings for context-aware replacements
3. **Grammar Preservation**: Add syntax awareness to maintain grammaticality
4. **Multi-Language Support**: Extend beyond English-centric approach
5. **Character-Level Augmentations**: Add typos, spelling variations
6. **Back-Translation**: Use translation for paraphrasing

## Performance Characteristics

### Time Complexity

- **RandomSwap**: O(n) where n = number of words
- **RandomDeletion**: O(n) where n = number of words
- **RandomInsertion**: O(n * m) where n = number of insertions, m = original word count
- **RandomSynonymReplacement**: O(n) where n = number of words
- **TextCompose**: O(sum of component transforms)

### Memory Usage

- All transforms operate on word lists, requiring O(n) temporary memory
- Vocabulary and synonym dictionaries are stored in each transform instance
- Minimal memory overhead beyond input/output strings

### Optimization Opportunities

1. **In-Place Operations**: Current implementation creates new strings; could optimize for in-place modifications
2. **SIMD Vectorization**: Not applicable to string operations, but could optimize probability sampling
3. **Caching**: Reuse split word lists when applying multiple transforms

## Testing

Comprehensive test suite available in `tests/shared/data/transforms/test_text_augmentations.mojo` (603 lines):

- 35 test cases covering all augmentations
- Edge case handling (empty text, single words)
- Probability and determinism tests
- Composition and integration tests

Run tests:

```bash
mojo test tests/shared/data/transforms/test_text_augmentations.mojo
```

Expected output:

```text
Running text augmentation tests...
  ✓ test_split_words_basic
  ✓ test_split_words_empty
  ... (30 more tests)
  ✓ test_augmentation_preserves_word_count_without_insertion_deletion

✓ All 35 text augmentation tests passed!
```

## References

### Source Code

- Implementation: `shared/data/text_transforms.mojo` (426 lines)
- Tests: `tests/shared/data/transforms/test_text_augmentations.mojo` (603 lines)

### Related Documentation

- [Issue #414: [Test] Text Augmentations](../414/README.md)
- [Issue #415: [Impl] Text Augmentations](../415/README.md)
- [Issue #417: [Cleanup] Text Augmentations](../417/README.md)
- [Generic Transforms Plan](/home/user/ml-odyssey/notes/plan/02-shared-library/03-data-utils/03-augmentations/03-generic-transforms/plan.md)

### External Resources

- [Easy Data Augmentation (EDA)](https://arxiv.org/abs/1901.11196) - Original inspiration
- [Text Augmentation Techniques](https://github.com/makcedward/nlpaug)

---

**Packaging Phase Status**: Complete

**Last Updated**: 2025-11-19
