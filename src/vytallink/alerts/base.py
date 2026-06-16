"""Alert provider interface and payload/result types.

Providers are pluggable: a console provider (always available), a webhook
provider, and future SMS/email/push providers can be added without touching
event logic. The :class:`AlertDispatcher` (see ``dispatcher.py``) fans an
:class:`AlertEvent` out to all configured providers, recording every attempt.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from vytallink.common.types import HealthStatus


@dataclass(slots=True)
class AlertEvent:
    """The information an alert conveys about a confirmed fall event."""

    event_uid: str
    timestamp: datetime
    confidence: float
    source_device: str
    state: str
    detection_count: int = 0
    simulated: bool = True
    message: str = ""

    def default_message(self) -> str:
        kind = "SIMULATED " if self.simulated else ""
        return (
            f"{kind}FALL CONFIRMED — event {self.event_uid} on "
            f"{self.source_device} (confidence {self.confidence:.0%})"
        )


@dataclass(slots=True)
class AlertResult:
    """The outcome of a single provider's delivery attempt."""

    provider: str
    success: bool
    attempt_time: datetime
    failure_message: str | None = None
    response_metadata: dict[str, Any] = field(default_factory=dict)


class AlertProvider(abc.ABC):
    """Abstract alert provider. Implementations must never raise on send —
    they return an :class:`AlertResult` with ``success=False`` on failure so
    the dispatcher can record it without crashing the application."""

    #: Stable provider name recorded in the database.
    name: str = "base"

    @abc.abstractmethod
    async def send(self, alert: AlertEvent) -> AlertResult:  # pragma: no cover - interface
        ...

    def health(self) -> HealthStatus:
        """Default: providers are considered OK unless they override this."""
        return HealthStatus.OK

    async def aclose(self) -> None:
        """Release any resources (e.g. HTTP clients). Default: no-op."""
        return None


@runtime_checkable
class AlertDispatcherProtocol(Protocol):
    """What the EventManager needs from an alert dispatcher."""

    async def dispatch(self, alert: AlertEvent) -> list[AlertResult]:
        ...
