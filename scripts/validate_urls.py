#!/usr/bin/env python3
"""
URL validation script for pre-commit hook

Validates URLs in Python files to catch broken links before commit.
Focuses on HTTP/HTTPS URLs in comments, docstrings, and string literals.

Exit codes:
    0: All URLs valid or reachable
    1: One or more URLs are broken

Usage:
    python scripts/validate_urls.py [files...]
"""

import re
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Tuple, Set


# URLs to skip validation (known to have issues or require authentication)
SKIP_URLS = {
    "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST",  # NIST has server issues
    "https://www.nist.gov/itl/products-and-services/emnist-dataset",  # NIST has server issues
    # Test and example URLs
    "https://example.com",  # Used in test files
    "http://example.com",  # Used in test files
    "https://arxiv.org/abs/1234.5678",  # Example URL in documentation
    "https://github.com/user/repo.git",  # Example in comments
    # Template URLs (contain placeholders)
    "https://github.com/modularml/mojo/issues/",  # Template in batch_planning_docs.py
    # External documentation (timeout issues in CI)
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview",  # CI timeout
}

# URL pattern to match HTTP/HTTPS URLs
URL_PATTERN = re.compile(
    r"https?://[a-zA-Z0-9][-a-zA-Z0-9@:%._\+~#=]{0,256}"
    r"\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&/=]*"
)


def extract_urls(file_path: Path) -> Set[str]:
    """Extract all HTTP/HTTPS URLs from a file."""
    urls = set()

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        raw_urls = URL_PATTERN.findall(content)
        # Strip trailing ) from URLs (common in markdown links)
        for url in raw_urls:
            # Remove trailing ) that may have been captured from markdown [text](url)
            cleaned_url = url.rstrip(")")
            urls.add(cleaned_url)
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)

    return urls


def check_url(url: str) -> Tuple[bool, str]:
    """
    Check if a URL is reachable.

    Returns:
        (is_valid, error_message).
    """
    if url in SKIP_URLS:
        return True, "Skipped (known issue)"

    try:
        # Set timeout and user agent
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (pre-commit hook)"})

        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.getcode()
            if 200 <= status < 400:
                return True, ""
            else:
                return False, f"HTTP {status}"

    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"URL Error: {e.reason}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def validate_files(file_paths: List[str]) -> int:
    """
    Validate URLs in the given files.

    Returns:
        Exit code (0 = success, 1 = failures).
    """
    all_urls = set()

    # Collect all unique URLs
    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        if file_path.suffix == ".py":
            urls = extract_urls(file_path)
            all_urls.update(urls)

    if not all_urls:
        return 0

    print(f"Checking {len(all_urls)} unique URLs...", file=sys.stderr)

    failed_urls = []

    for url in sorted(all_urls):
        is_valid, error = check_url(url)

        if not is_valid:
            failed_urls.append((url, error))
            print(f"✗ {url}: {error}", file=sys.stderr)
        else:
            if error:  # Skipped URL
                print(f"⊘ {url}: {error}", file=sys.stderr)

    if failed_urls:
        print(f"\n{len(failed_urls)} URL(s) failed validation", file=sys.stderr)
        return 1

    print("All URLs validated successfully", file=sys.stderr)
    return 0


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: validate_urls.py [files...]", file=sys.stderr)
        return 0

    return validate_files(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
