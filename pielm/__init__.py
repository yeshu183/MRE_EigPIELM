"""
PIELM (Physics-Informed Extreme Learning Machine) for MRE.

This module implements PIELM as an alternative to PINN for solving MRE inverse problems.
Key advantages:
- One-shot training (no iterations)
- Analytical derivatives via random features
- Fast inference

Modules:
- features: Random feature generation and derivative computation
- model: MREPIELM class (analog of MREPINN)
- training: MREPIELMModel class (one-shot solver)
- equations: PDE constraint matrix construction
- solver: Linear system solver
"""

from .features import RandomFeatures
from .model import MREPIELM
from .solver import solve_linear_system

# Import training only if dependencies available
try:
    from .training import MREPIELMModel
    _has_training = True
except ImportError:
    _has_training = False
    MREPIELMModel = None

__all__ = [
    'RandomFeatures',
    'MREPIELM',
    'solve_linear_system',
]

if _has_training:
    __all__.append('MREPIELMModel')
