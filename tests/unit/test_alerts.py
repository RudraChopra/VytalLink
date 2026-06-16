"""Tests for alert providers and the dispatcher (delivery + isolation)."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from vytallink.alerts.base import AlertEvent, AlertProvider, AlertResult
from vytallink.alerts.console import ConsoleAlertProvider
from vytallink.alerts.dispatcher import AlertDispatcher
from vytallink.alerts.webhook import WebhookAlertProvider
from vytallink.common.clock import ManualClock


def _alert() -> AlertEvent:
    return AlertEvent(
        event_uid="evt-1",
        timestamp=ManualClock().now(),
        confidence=0.93,
        source_device="camera-1",
        state="confirmed_fall",
        detection_count=4,
        simulated=True,
    )


@pytest.mark.asyncio
async def test_console_provider_delivers():
    p = ConsoleAlertProvider(clock=ManualClock())
    result = await p.send(_alert())
    assert result.success is True
    assert result.provider == "console"
    assert "message" in result.response_metadata


@pytest.mark.asyncio
async def test_webhook_empty_url_fails_cleanly():
    p = WebhookAlertProvider(url="", clock=ManualClock())
    result = await p.send(_alert())
    assert result.success is False
    assert "not configured" in result.failure_message


def test_webhook_signature():
    p = WebhookAlertProvider(url="https://x", secret="topsecret")
    body = b'{"a":1}'
    sig = p._sign(body)
    expected = "sha256=" + hmac.new(b"topsecret", body, hashlib.sha256).hexdigest()
    assert sig == expected
    # No secret -> no signature.
    assert WebhookAlertProvider(url="https://x")._sign(body) is None


@pytest.mark.asyncio
async def test_webhook_success_with_signature(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 202

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, content, headers):
            captured["url"] = url
            captured["content"] = content
            captured["headers"] = headers
            return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    p = WebhookAlertProvider(url="https://hooks.example/x", secret="s3cret", clock=ManualClock())
    result = await p.send(_alert())
    assert result.success is True
    assert result.response_metadata["status_code"] == 202
    # Body is valid JSON and signature header present.
    payload = json.loads(captured["content"])
    assert payload["event_uid"] == "evt-1"
    assert payload["simulated"] is True
    assert captured["headers"]["X-VytalLink-Signature"].startswith("sha256=")


@pytest.mark.asyncio
async def test_webhook_network_error_recorded_not_raised(monkeypatch):
    class BoomClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            raise RuntimeError("connection refused")

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", BoomClient)
    p = WebhookAlertProvider(url="https://hooks.example/x", clock=ManualClock())
    result = await p.send(_alert())
    assert result.success is False
    assert "connection refused" in result.failure_message


class _RaisingProvider(AlertProvider):
    name = "boom"

    async def send(self, alert):
        raise RuntimeError("kaboom")


@pytest.mark.asyncio
async def test_dispatcher_records_and_isolates(repos, manual_clock):
    dispatcher = AlertDispatcher(
        [ConsoleAlertProvider(clock=manual_clock), _RaisingProvider()],
        repos,
        clock=manual_clock,
    )
    results = await dispatcher.dispatch(_alert())
    # One success (console) + one isolated failure (boom), no exception raised.
    assert len(results) == 2
    assert any(r.success for r in results)
    assert any(not r.success and "boom" in r.provider for r in results)
    # Both attempts recorded in the DB.
    assert repos.alerts.count(event_uid="evt-1") == 2
    assert repos.alerts.count(success=True) == 1
    assert repos.alerts.count(success=False) == 1


@pytest.mark.asyncio
async def test_dispatcher_provider_names(manual_clock):
    d = AlertDispatcher([ConsoleAlertProvider(clock=manual_clock)], None, clock=manual_clock)
    assert d.provider_names == ["console"]
