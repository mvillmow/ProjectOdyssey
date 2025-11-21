"""
ML Odyssey Version Module

Provides centralized version information for all Mojo modules.

Usage:
    from shared.version import VERSION, get_version

    fn main():
        print("ML Odyssey version:", VERSION)
        let v = get_version()
        print("Version string:", v)
"""

# Version constants (updated by scripts/update_version.py)
alias VERSION = "0.1.0"
alias VERSION_MAJOR = 0
alias VERSION_MINOR = 1
alias VERSION_PATCH = 0


fn get_version() -> String:
    """
    Get the version string.

    Returns:
        Version string in format "MAJOR.MINOR.PATCH"
    """
    return VERSION


fn get_version_tuple() -> (Int, Int, Int):
    """
    Get the version as a tuple of integers.

    Returns:
        Tuple of (major, minor, patch)
    """
    return (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)


fn version_info() -> String:
    """
    Get detailed version information.

    Returns:
        Formatted string with version details
    """
    return "ML Odyssey v" + VERSION + " (Mojo-based AI Research Platform)"
