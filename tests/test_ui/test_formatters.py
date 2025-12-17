"""Tests for UI formatters."""

import pytest
from datetime import datetime, timedelta

from aria.ui.formatters import (
    truncate_text,
    truncate_with_preview,
    format_timestamp,
    format_duration,
    format_size,
    format_list,
    format_dict_table,
    format_error_trace,
    wrap_text,
    format_percentage,
    strip_ansi,
)


class TestTruncateText:
    """Test text truncation."""

    def test_short_text(self):
        """Test that short text is not truncated."""
        text = "Short text"
        result = truncate_text(text, max_length=100)
        assert result == text

    def test_long_text(self):
        """Test that long text is truncated."""
        text = "a" * 1000
        result = truncate_text(text, max_length=100)
        assert len(result) <= 100
        assert result.endswith("... (truncated)")

    def test_custom_suffix(self):
        """Test truncation with custom suffix."""
        text = "a" * 1000
        result = truncate_text(text, max_length=100, suffix="...")
        assert result.endswith("...")
        assert len(result) == 100


class TestTruncateWithPreview:
    """Test truncation with preview."""

    def test_short_text(self):
        """Test that short text is not truncated."""
        text = "Short text"
        result = truncate_with_preview(text, max_length=100)
        assert result == text

    def test_long_text(self):
        """Test that long text includes preview message."""
        text = "a" * 1000
        result = truncate_with_preview(text, max_length=100)
        assert len(result) > 100  # Includes preview text
        assert "verbose" in result.lower()


class TestFormatTimestamp:
    """Test timestamp formatting."""

    def test_default_format(self):
        """Test default timestamp format."""
        dt = datetime(2024, 1, 15, 14, 30, 45)
        result = format_timestamp(dt)
        assert result == "2024-01-15 14:30:45"

    def test_custom_format(self):
        """Test custom timestamp format."""
        dt = datetime(2024, 1, 15, 14, 30, 45)
        result = format_timestamp(dt, format="%Y-%m-%d")
        assert result == "2024-01-15"

    def test_current_time(self):
        """Test formatting current time."""
        result = format_timestamp()
        assert isinstance(result, str)
        assert len(result) > 0


class TestFormatDuration:
    """Test duration formatting."""

    def test_milliseconds(self):
        """Test formatting milliseconds."""
        result = format_duration(0.5)
        assert result == "500ms"

    def test_seconds(self):
        """Test formatting seconds."""
        result = format_duration(2.5)
        assert result == "2.5s"

    def test_minutes(self):
        """Test formatting minutes."""
        result = format_duration(90)
        assert result == "1m 30s"

    def test_hours(self):
        """Test formatting hours."""
        result = format_duration(3665)
        assert result == "1h 1m"


class TestFormatSize:
    """Test file size formatting."""

    def test_bytes(self):
        """Test formatting bytes."""
        result = format_size(512)
        assert "B" in result

    def test_kilobytes(self):
        """Test formatting kilobytes."""
        result = format_size(1024 * 5)
        assert "KB" in result

    def test_megabytes(self):
        """Test formatting megabytes."""
        result = format_size(1024 * 1024 * 10)
        assert "MB" in result

    def test_gigabytes(self):
        """Test formatting gigabytes."""
        result = format_size(1024 * 1024 * 1024 * 2)
        assert "GB" in result


class TestFormatList:
    """Test list formatting."""

    def test_bullet_list(self):
        """Test bullet list formatting."""
        items = ["Item 1", "Item 2", "Item 3"]
        result = format_list(items, style="bullet")
        assert "•" in result
        assert "Item 1" in result

    def test_numbered_list(self):
        """Test numbered list formatting."""
        items = ["Item 1", "Item 2", "Item 3"]
        result = format_list(items, style="numbered")
        assert "1." in result
        assert "2." in result
        assert "3." in result

    def test_dash_list(self):
        """Test dash list formatting."""
        items = ["Item 1", "Item 2"]
        result = format_list(items, style="dash")
        assert "-" in result


class TestFormatDictTable:
    """Test dictionary table formatting."""

    def test_basic_dict(self):
        """Test formatting a basic dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        result = format_dict_table(data)
        assert "key1" in result
        assert "value1" in result
        assert "key2" in result
        assert "value2" in result

    def test_empty_dict(self):
        """Test formatting an empty dictionary."""
        data = {}
        result = format_dict_table(data)
        assert result == ""


class TestFormatErrorTrace:
    """Test error trace formatting."""

    def test_with_type(self):
        """Test error formatting with type."""
        error = ValueError("Something went wrong")
        result = format_error_trace(error, include_type=True)
        assert "ValueError" in result
        assert "Something went wrong" in result

    def test_without_type(self):
        """Test error formatting without type."""
        error = ValueError("Something went wrong")
        result = format_error_trace(error, include_type=False)
        assert "ValueError" not in result
        assert "Something went wrong" in result


class TestWrapText:
    """Test text wrapping."""

    def test_short_text(self):
        """Test that short text is not wrapped."""
        text = "Short text"
        result = wrap_text(text, width=80)
        assert result == text

    def test_long_text(self):
        """Test that long text is wrapped."""
        text = "a " * 100
        result = wrap_text(text, width=40)
        assert "\n" in result

    def test_with_indent(self):
        """Test wrapping with indentation."""
        text = "Short text"
        result = wrap_text(text, width=80, indent=4)
        assert result.startswith("    ")


class TestFormatPercentage:
    """Test percentage formatting."""

    def test_basic_percentage(self):
        """Test basic percentage formatting."""
        result = format_percentage(0.755)
        assert result == "75.5%"

    def test_zero_decimal_places(self):
        """Test percentage with no decimal places."""
        result = format_percentage(0.755, decimal_places=0)
        assert result == "76%"

    def test_multiple_decimal_places(self):
        """Test percentage with multiple decimal places."""
        result = format_percentage(0.12345, decimal_places=3)
        assert result == "12.345%"


class TestStripAnsi:
    """Test ANSI code stripping."""

    def test_plain_text(self):
        """Test that plain text is unchanged."""
        text = "Plain text"
        result = strip_ansi(text)
        assert result == text

    def test_with_ansi_codes(self):
        """Test stripping ANSI codes."""
        text = "\x1b[31mRed text\x1b[0m"
        result = strip_ansi(text)
        assert result == "Red text"
        assert "\x1b" not in result
