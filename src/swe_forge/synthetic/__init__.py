"""Synthetic task generation utilities."""

from swe_forge.synthetic.feature_deletion import (
    FeatureDeletionError,
    PythonFeatureDeletion,
    build_python_function_deletion,
)
from swe_forge.synthetic.pipeline import create_feature_deletion_task

__all__ = [
    "FeatureDeletionError",
    "PythonFeatureDeletion",
    "build_python_function_deletion",
    "create_feature_deletion_task",
]
