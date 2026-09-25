"""The whole pipeline must run with the network disabled."""

import socket

import pytest

import solution


@pytest.fixture
def no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise OSError("network disabled in test")

    monkeypatch.setattr(socket, "socket", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("YOLO_OFFLINE", "1")
    yield


def test_pipeline_runs_offline(no_network, clip):
    from roadsight.perception import detector

    detector._MODELS.clear()  # force a fresh weight load under the network guard
    solution._pipeline = None
    events = solution.detect_events(str(clip))
    assert isinstance(events, list)
    assert solution._pipeline is not None, "pipeline failed to initialise offline"
