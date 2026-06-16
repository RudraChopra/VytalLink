"""Console alert provider — always available, no credentials required.

Logs a clear, human-readable alert line at WARNING level. This is the provider
that works tonight in development.
"""

from __future__ import annotations

from vytallink.alerts.base import AlertEvent, AlertProvider, AlertResult
from vytallink.common.clock import Clock, SystemClock
from vytallink.common.logging_setup import get_logger

log = get_logger("alerts.console")


class ConsoleAlertProvider(AlertProvider):
    name = "console"

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock: Clock = clock or SystemClock()

    async def send(self, alert: AlertEvent) -> AlertResult:
        message = alert.message or alert.default_message()
        # No secrets are involved in a console alert.
        log.warning(
            "ALERT [%s] %s | event=%s source=%s confidence=%.2f detections=%d",
            "SIMULATED" if alert.simulated else "LIVE",
            message,
            alert.event_uid,
            alert.source_device,
            alert.confidence,
            alert.detection_count,
        )
        return AlertResult(
            provider=self.name,
            success=True,
            attempt_time=self.clock.now(),
            response_metadata={"message": message, "simulated": alert.simulated},
        )
