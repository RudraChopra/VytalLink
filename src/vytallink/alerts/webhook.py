"""Webhook alert provider — posts a signed JSON payload to a configured URL.

* The request is signed with HMAC-SHA256 over the raw body using
  ``WEBHOOK_SECRET`` (header ``X-VytalLink-Signature: sha256=<hex>``) so the
  receiver can verify authenticity. The secret itself is never sent or logged.
* The destination URL is sanitized before logging.
* ``send`` never raises — any error becomes ``AlertResult(success=False, …)``
  so the dispatcher records it without crashing the application.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from vytallink.alerts.base import AlertEvent, AlertProvider, AlertResult
from vytallink.common.clock import Clock, SystemClock, isoformat
from vytallink.common.logging_setup import get_logger
from vytallink.common.sanitize import sanitize_url
from vytallink.common.types import HealthStatus

log = get_logger("alerts.webhook")


class WebhookAlertProvider(AlertProvider):
    name = "webhook"

    def __init__(
        self,
        url: str,
        secret: str = "",
        *,
        timeout: float = 5.0,
        clock: Clock | None = None,
    ) -> None:
        self.url = url
        self._secret = secret
        self.timeout = timeout
        self.clock: Clock = clock or SystemClock()

    @property
    def safe_url(self) -> str:
        return sanitize_url(self.url)

    def _build_payload(self, alert: AlertEvent) -> dict[str, Any]:
        return {
            "type": "fall_event",
            "event_uid": alert.event_uid,
            "timestamp": isoformat(alert.timestamp),
            "confidence": round(alert.confidence, 4),
            "source_device": alert.source_device,
            "state": alert.state,
            "detection_count": alert.detection_count,
            "simulated": alert.simulated,
            "message": alert.message or alert.default_message(),
        }

    def _sign(self, body: bytes) -> str | None:
        if not self._secret:
            return None
        digest = hmac.new(self._secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        return f"sha256={digest}"

    async def send(self, alert: AlertEvent) -> AlertResult:
        attempt_time = self.clock.now()
        if not self.url:
            return AlertResult(
                provider=self.name,
                success=False,
                attempt_time=attempt_time,
                failure_message="webhook URL not configured",
            )
        payload = self._build_payload(alert)
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": "VytalLink/0.1"}
        signature = self._sign(body)
        if signature:
            headers["X-VytalLink-Signature"] = signature

        try:
            import httpx  # noqa: WPS433 (kept local; httpx also used by tests)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.url, content=body, headers=headers)
            success = 200 <= resp.status_code < 300
            meta = {"status_code": resp.status_code, "signed": bool(signature)}
            if success:
                log.info("Webhook alert delivered to %s (%d)", self.safe_url, resp.status_code)
                return AlertResult(
                    provider=self.name,
                    success=True,
                    attempt_time=attempt_time,
                    response_metadata=meta,
                )
            log.warning(
                "Webhook alert to %s returned HTTP %d", self.safe_url, resp.status_code
            )
            return AlertResult(
                provider=self.name,
                success=False,
                attempt_time=attempt_time,
                failure_message=f"HTTP {resp.status_code}",
                response_metadata=meta,
            )
        except Exception as exc:
            # Network error, timeout, DNS failure, etc. Never crash.
            log.warning("Webhook alert to %s failed: %s", self.safe_url, exc)
            return AlertResult(
                provider=self.name,
                success=False,
                attempt_time=attempt_time,
                failure_message=f"{type(exc).__name__}: {exc}",
            )

    def health(self) -> HealthStatus:
        return HealthStatus.OK if self.url else HealthStatus.DISABLED
