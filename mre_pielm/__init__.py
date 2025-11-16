"""
MRE-PIELM: Physics-Informed Extreme Learning Machine for MRE Forward Problem

This package implements a fast forward solver for Magnetic Resonance Elastography
using Bernstein polynomial basis functions and extreme learning machines.

Main components:
- BernsteinBasis3D: 3D tensor product Bernstein polynomial basis
- MREPIELM: Model class for MRE using Bernstein basis
- MREPIELMForwardSolver: High-level solver interface
"""

__version__ = "0.1.0"
__author__ = "MRE-PINN Team"

# Core imports
from .model import MREPIELM
from . import core
from . import utils

__all__ = [
    'MREPIELM',
    'core',
    'utils',
]
