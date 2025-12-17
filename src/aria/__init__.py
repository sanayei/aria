"""ARIA - AI Research & Intelligence Assistant.

A local-first agentic AI personal assistant with privacy-preserving design.
"""

__version__ = "0.1.0"

from aria.config import Settings, get_settings, reload_settings

__all__ = ["Settings", "get_settings", "reload_settings", "__version__"]
