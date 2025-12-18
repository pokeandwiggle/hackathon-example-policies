"""
Saliency Analysis for Robot Learning Policies.

This package provides tools to compute and visualize saliency maps showing
which image regions and state dimensions most influence a policy's action predictions.
"""

from .core import compute_saliency
from .visualization import visualize_saliency

__all__ = ["compute_saliency", "visualize_saliency"]
