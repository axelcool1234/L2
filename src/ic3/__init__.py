# src/ic3/__init__.py

"""
IC3: an implementation of IC3 in Python.
https://theory.stanford.edu/~arbrad/papers/IC3.pdf
"""

from .ic3 import IC3Prover
from .extractor import TransitionExtractor

__all__ = ["IC3Prover", "TransitionExtractor"]
