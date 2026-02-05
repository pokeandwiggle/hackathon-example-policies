"""Compatibility module for lerobot API changes.

This module provides compatibility shims for APIs that changed between lerobot versions.
"""

from .normalize import Normalize, Unnormalize

__all__ = ["Normalize", "Unnormalize"]
