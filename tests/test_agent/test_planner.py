"""Tests for task planner."""

import pytest

from aria.agent.planner import TaskPlanner, TaskAnalysis


class TestTaskPlanner:
    """Test TaskPlanner class."""

    def test_analyze_simple_question(self):
        """Test analyzing a simple question."""
        planner = TaskPlanner()

        analysis = planner.analyze_request("What is Python?")

        assert isinstance(analysis, TaskAnalysis)
        assert analysis.requires_tools is False
        assert analysis.complexity == "simple"

    def test_analyze_time_request(self):
        """Test analyzing a time request."""
        planner = TaskPlanner()

        analysis = planner.analyze_request("What time is it?")

        assert isinstance(analysis, TaskAnalysis)
        assert analysis.requires_tools is True
        assert analysis.suggested_approach is not None

    def test_analyze_system_info_request(self):
        """Test analyzing a system info request."""
        planner = TaskPlanner()

        analysis = planner.analyze_request("What's my platform?")

        assert isinstance(analysis, TaskAnalysis)
        assert analysis.requires_tools is True

    def test_analyze_complex_task(self):
        """Test analyzing a complex multi-step task."""
        planner = TaskPlanner()

        analysis = planner.analyze_request(
            "First, send an email, and then create a file, after that search for info"
        )

        assert isinstance(analysis, TaskAnalysis)
        assert analysis.complexity in ["moderate", "complex"]

    def test_confidence_score(self):
        """Test that confidence score is in valid range."""
        planner = TaskPlanner()

        analysis = planner.analyze_request("What time is it?")

        assert 0.0 <= analysis.confidence <= 1.0
