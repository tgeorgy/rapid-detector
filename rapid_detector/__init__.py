"""
Rapid Detector - Fast Object Detection Configuration System

A Python package for creating custom object detectors without training,
using foundation models with visual examples.
"""

from .detector import RapidDetector
from .storage import DetectorStorage

__version__ = "0.1.0"
__all__ = ["RapidDetector", "DetectorStorage"]