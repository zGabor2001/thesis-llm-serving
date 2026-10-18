"""Tests for the residency tracker (cold/switch classification)."""

from gateway.dispatcher.residency import ResidencyTracker


def test_residency_tracker_marks_cold_and_switch():
    tracker = ResidencyTracker()

    assert tracker.mark("tenant-a") == {"cold": True, "switch": False}
    assert tracker.mark("tenant-a") == {"cold": False, "switch": False}
    assert tracker.mark("tenant-b") == {"cold": True, "switch": True}
    assert tracker.mark("tenant-a") == {"cold": False, "switch": True}
    assert tracker.mark("tenant-a") == {"cold": False, "switch": False}
