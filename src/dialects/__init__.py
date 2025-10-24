# src/dialects/__init__.py

"""
Contains custom dialects like bignum
"""

from .bignum_to_llvm import LowerBigNumToLLVM

__all__ = [
    "LowerBigNumToLLVM",
]
