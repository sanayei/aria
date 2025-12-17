"""Approval system for user confirmation of risky tool executions.

This module provides risk classification and approval handling for tools
based on their potential impact.
"""

from aria.approval.classifier import ActionClassifier
from aria.approval.handler import ApprovalResult, ApprovalHandler

__all__ = [
    "ActionClassifier",
    "ApprovalResult",
    "ApprovalHandler",
]
