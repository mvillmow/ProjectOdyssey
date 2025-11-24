"""Text transformation and augmentation utilities.

This module provides transformations for augmenting text data for NLP tasks.
Implements basic word-level augmentations including synonym replacement,
random insertion, random swap, and random deletion.

Limitations:
- Basic word-level operations (split on spaces)
- English-centric approach
- Simple synonym dictionary (not semantic embeddings)
- May produce ungrammatical text
- No advanced NLP features
"""

from random import random_si64


# ============================================================================
# Helper Functions
# ============================================================================


fn random_float() -> Float64:
    """Generate random float in [0, 1) with high precision.

    Uses 1 billion possible values for better probability distribution.

    Returns:.        Random float in range [0.0, 1.0).
    """
    return Float64(random_si64(0, 1000000000)) / 1000000000.0


# ============================================================================
# TextTransform Trait
# ============================================================================


trait TextTransform:
    """Base interface for text transforms.

    Text transforms modify string data and return transformed copies.
    Unlike the Transform trait which works with ExTensor, this works with String.
    """

    fn __call__(self, text: String) raises -> String:
        """Apply the transform to text.

        Args:.            `text`: Input text string.

        Returns:.            Transformed text string.

        Raises:.            Error if transform cannot be applied.
        """
        ...


# ============================================================================
# Helper Functions
# ============================================================================


fn split_words(text: String) raises -> List[String]:
    """Split text into words by spaces.

    Simple space-based tokenization. Does not handle punctuation specially.

    Args:.        `text`: Input text to split.

    Returns:.        List of words (space-separated tokens).
    """
    # Use built-in split method
    var parts = text.split(" ")

    # Filter out empty strings that may result from multiple spaces
    var words = List[String]()
    for i in range(len(parts)):
        if len(parts[i]) > 0:
            words.append(parts[i])

    return words^


fn join_words(words: List[String]) raises -> String:
    """Join words into text with spaces.

    Args:.        `words`: List of words to join.

    Returns:.        Joined text with spaces between words.
    """
    if len(words) == 0:
        return String("")

    var result = words[0]
    for i in range(1, len(words)):
        result += " " + words[i]

    return result


# ============================================================================
# Text Augmentation Transforms
# ============================================================================


struct RandomSwap(TextTransform, Copyable, Movable):
    """Randomly swap positions of word pairs.

    Swaps adjacent or nearby word positions with configurable probability.
    Helps create variations while preserving overall meaning.

    Example:.        "the quick brown fox" -> "quick the brown fox" (first two swapped)
    """

    var p: Float64  # Probability of performing swap
    var n: Int  # Number of swaps to perform

    fn __init__(mut self, p: Float64 = 0.15, n: Int = 2):
        """Create random swap transform.

        Args:
            p: Probability of performing each swap (0.0 to 1.0).
            n: Number of swap operations to attempt.
        """
        self.p = p
        self.n = n

    fn __call__(self, text: String) raises -> String:
        """Randomly swap word pairs in text.

        Args:.            `text`: Input text.

        Returns:.            Text with randomly swapped words.

        Raises:.            Error if operation fails.
        """
        # Handle empty or single-word text
        if len(text) == 0:
            return text

        var words = split_words(text)
        if len(words) <= 1:
            return text

        # Perform n swap operations
        for _ in range(self.n):
            # Check probability
            var rand_val = random_float()
            if rand_val >= self.p:
                continue

            # Pick two random positions
            var idx1 = Int(random_si64(0, len(words)))
            var idx2 = Int(random_si64(0, len(words)))

            # Ensure different positions
            if idx1 != idx2:
                # Swap
                var temp = words[idx1]
                words[idx1] = words[idx2]
                words[idx2] = temp

        return join_words(words)


struct RandomDeletion(TextTransform, Copyable, Movable):
    """Randomly delete words from text.

    Deletes words with specified probability while ensuring at least.
    one word remains. Helps create shorter variations.

    Example:.        "the quick brown fox" -> "quick brown fox" (deleted "the")
    """

    var p: Float64  # Probability of deleting each word

    fn __init__(mut self, p: Float64 = 0.1):
        """Create random deletion transform.

        Args:
            p: Probability of deleting each word (0.0 to 1.0).
        """
        self.p = p

    fn __call__(self, text: String) raises -> String:
        """Randomly delete words from text.

        Ensures at least one word remains even if all would be deleted.

        Args:.            `text`: Input text.

        Returns:.            Text with some words randomly deleted.

        Raises:.            Error if operation fails.
        """
        # Handle empty text
        if len(text) == 0:
            return text

        var words = split_words(text)
        if len(words) == 0:
            return text

        # If only one word, don't delete
        if len(words) == 1:
            return text

        # Decide which words to keep
        var kept_words = List[String]()
        for i in range(len(words)):
            var rand_val = random_float()
            if rand_val >= self.p:
                # Keep this word
                kept_words.append(words[i])

        # Ensure at least one word remains
        if len(kept_words) == 0:
            # Keep a random word
            var idx = Int(random_si64(0, len(words)))
            kept_words.append(words[idx])

        return join_words(kept_words)


struct RandomInsertion(TextTransform, Copyable, Movable):
    """Insert random words from vocabulary into text.

    Inserts words from a predefined vocabulary at random positions.
    Helps increase lexical diversity.

    Example:.        "the brown fox" -> "the quick brown fox" (inserted "quick")
    """

    var p: Float64  # Probability of insertion
    var n: Int  # Number of words to insert
    var vocabulary: List[String]  # Words to insert from

    fn __init__(mut self, p: Float64 = 0.1, n: Int = 1, var vocabulary: List[String]):
        """Create random insertion transform.

        Args:
            p: Probability of performing insertion (0.0 to 1.0).
            n: Number of words to insert.
            vocabulary: List of words to choose from for insertion.
        """
        self.p = p
        self.n = n
        self.vocabulary = vocabulary^

    fn __call__(self, text: String) raises -> String:
        """Insert random words from vocabulary into text.

        Args:.            `text`: Input text.

        Returns:.            Text with randomly inserted words.

        Raises:.            Error if operation fails.
        """
        # Handle empty text or empty vocabulary
        if len(text) == 0 or len(self.vocabulary) == 0:
            return text

        var words = split_words(text)
        if len(words) == 0:
            return text

        # Perform n insertion operations
        for _ in range(self.n):
            # Check probability
            var rand_val = random_float()
            if rand_val >= self.p:
                continue

            # Pick random word from vocabulary
            var vocab_idx = Int(random_si64(0, len(self.vocabulary)))
            var word_to_insert = self.vocabulary[vocab_idx]

            # Pick random position to insert (0 to len(words) inclusive)
            var insert_pos = Int(random_si64(0, len(words) + 1))

            # Insert word at position
            var new_words = List[String]()
            for i in range(len(words)):
                if i == insert_pos:
                    new_words.append(word_to_insert)
                new_words.append(words[i])

            # Handle case where insert_pos == len(words)
            if insert_pos == len(words):
                new_words.append(word_to_insert)

            words = new_words

        return join_words(words)


struct RandomSynonymReplacement(TextTransform, Copyable, Movable):
    """Replace random words with synonyms from dictionary.

    Uses a simple synonym dictionary to replace words with alternatives.
    This is a conservative augmentation that preserves meaning well.

    Example:.        "the quick fox" -> "the fast fox" (replaced "quick" with "fast")
    """

    var p: Float64  # Probability of replacing each word
    var synonyms: Dict[String, List[String]]  # Synonym dictionary

    fn __init__(mut self, p: Float64 = 0.2, var synonyms: Dict[String, List[String]]):
        """Create random synonym replacement transform.

        Args:
            p: Probability of replacing each word (0.0 to 1.0).
            synonyms: Dictionary mapping words to lists of synonyms.
        """
        self.p = p
        self.synonyms = synonyms^

    fn __call__(self, text: String) raises -> String:
        """Replace random words with synonyms.

        Args:.            `text`: Input text.

        Returns:.            Text with some words replaced by synonyms.

        Raises:.            Error if operation fails.
        """
        # Handle empty text
        if len(text) == 0:
            return text

        var words = split_words(text)
        if len(words) == 0:
            return text

        # Process each word
        var result_words = List[String]()
        for i in range(len(words)):
            var word = words[i]

            # Check if should replace
            var rand_val = random_float()
            if rand_val < self.p and word in self.synonyms:
                # Get synonyms for this word
                var syns = self.synonyms[word]
                if len(syns) > 0:
                    # Pick random synonym
                    var syn_idx = Int(random_si64(0, len(syns)))
                    result_words.append(syns[syn_idx])
                else:
                    # No synonyms available, keep original
                    result_words.append(word)
            else:
                # Don't replace
                result_words.append(word)

        return join_words(result_words)


# ============================================================================
# Text Pipeline (Compose)
# ============================================================================


struct TextCompose(TextTransform, Copyable, Movable):
    """Compose multiple text transforms sequentially.

    Applies text transforms in order, passing output of each to the next.
    Similar to the Compose transform for tensors but works with strings.
    """

    var transforms: List[TextTransform]

    fn __init__(mut self, var transforms: List[TextTransform]):
        """Create composition of text transforms.

        Args:
            transforms: List of text transforms to apply in order.
        """
        self.transforms = transforms^

    fn __call__(self, text: String) raises -> String:
        """Apply all text transforms sequentially.

        Args:.            `text`: Input text.

        Returns:.            Transformed text after all transforms.

        Raises:.            Error if any transform cannot be applied.
        """
        var result = text
        for t in self.transforms:
            result = t(result)
        return result

    fn __len__(self) -> Int:
        """Return number of transforms."""
        return len(self.transforms)

    fn append(mut self, transform: TextTransform):
        """Add a transform to the pipeline.

        Args:
            transform: Text transform to add.
        """
        self.transforms.append(transform)


# Type alias for more intuitive naming
alias TextPipeline = TextCompose
