"""
Pytest configuration for Sovereign On-Premise Agentic AI Workbench.
Enables fast test mode for unit and integration testing.
"""
import os
import pytest

# Ensure all tests run with test mode flags
os.environ["WORKBENCH_TEST_MODE"] = "1"
