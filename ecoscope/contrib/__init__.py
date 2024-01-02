"""
External code included for simplified distribution.
"""
__version__ = "0.30.0"


def in_colab_shell():
    """Tests if the code is being executed within Google Colab."""
    import sys

    if "google.colab" in sys.modules:
        return True
    else:
        return False
