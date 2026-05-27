"""Profiling module for user adaptation and intent filtering.

This module provides profile-based response adaptation through
system prompt selection based on user expertise level.
"""

from profiling.profiler import SYSTEM_PROMPTS, build_system_prompt

__all__ = ["SYSTEM_PROMPTS", "build_system_prompt"]
