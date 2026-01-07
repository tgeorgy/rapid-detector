"""Utility functions for Rapid Detector."""

import re


def normalize_detector_id(class_name: str) -> str:
    """
    Normalize class name input to URL-safe detector ID format.
    
    Args:
        class_name: Raw user input class name
        
    Returns:
        Normalized detector ID (a-z, A-Z, 0-9, and dashes only)
        
    Examples:
        normalize_detector_id("Red Car") -> "Red-Car"
        normalize_detector_id("iPhone 14 Pro") -> "iPhone-14-Pro" 
        normalize_detector_id("  Person walking  ") -> "Person-walking"
        normalize_detector_id("Model-X123 @#$%") -> "Model-X123"
        normalize_detector_id("COOKIE") -> "COOKIE"
    """
    if not class_name or not isinstance(class_name, str):
        raise ValueError("Class name must be a non-empty string")
    
    # Strip whitespace but preserve case
    normalized = class_name.strip()
    
    # Replace spaces and common separators with dashes
    normalized = re.sub(r'[-_\s]+', '-', normalized)
    
    # Keep only letters (both cases), numbers, and dashes
    normalized = re.sub(r'[^a-zA-Z0-9-]', '', normalized)
    
    # Remove multiple consecutive dashes
    normalized = re.sub(r'-+', '-', normalized)
    
    # Remove leading/trailing dashes
    normalized = normalized.strip('-')
    
    # Ensure it's not empty after normalization
    if not normalized:
        raise ValueError("Class name contains no valid characters")
    
    return normalized


def get_prompt_text(class_name: str, is_semantic: bool = True) -> str:
    """
    Get the prompt text to use for SAM3 based on class name and semantic flag.
    
    Args:
        class_name: Original user-provided class name
        is_semantic: Whether this is a semantic name or visual-only detector
        
    Returns:
        Prompt text for SAM3 model
    """
    if is_semantic:
        return class_name.strip()
    else:
        return "visual"