# src/l2/__init__.py

"""
LoopLang: a simple loop language used for loop invariant inference experimentation.
"""

from .core import (
    IRGenCompiler,
    IRGenInterpreter,
    grammar,
    precedence,
)

__all__ = [
    "IRGenCompiler",
    "IRGenInterpreter",
    "grammar",
    "precedence",
]
