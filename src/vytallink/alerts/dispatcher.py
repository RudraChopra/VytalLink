"""AlertDispatcher: fans an AlertEvent out to providers and records attempts.

* Every provider attempt is recorded in the ``alerts`` table (success or not).
* A provider that raises unexpectedly is isolated — the error is recorded and
  the remaining providers still run. An alert failure never crashes the app.
* Returns the list of :class:`AlertResult` for the caller (EventManager).
"""

from __future__ import annotations

from typing import Sequence

from vytallink.alerts.base import AlertEvent, AlertProvider, AlertResult
from vytallink.common.clock import Clock, SystemClock, isoformat
from vytallink.common.logging_setup import get_logger
from vytallink.database.models import AlertRow
from vytallink.database.repositories import Repositories

log = get_logger("alerts.dispatcher")


class AlertDispatcher:
    def __init__(
        self,
        providers: Sequence[AlertProvider],
        repos: Repositories | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.providers = list(providers)
        self.repos = repos
        self.clock: Clock = clock or SystemClock()

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    async def dispatch(self, alert: AlertEvent) -> list[AlertResult]:
        results: list[AlertResult] = []
        for provider in self.providers:
            result = await self._safe_send(provider, alert)
            results.append(result)
            self._record(alert, result)
        delivered = sum(1 for r in results if r.success)
        log.info(
            "Dispatched alert for %s to %d provider(s); %d delivered",
            alert.event_uid,
            len(self.providers),
            delivered,
        )
        return results

    async def _safe_send(self, provider: AlertProvider, alert: AlertEvent) -> AlertResult:
        try:
            return await provider.send(alert)
        except Exception as exc:  # provider isolation
            log.error("Alert provider %s raised: %s", provider.name, exc)
            return AlertResult(
                provider=provider.name,
                success=False,
                attempt_time=self.clock.now(),
                failure_message=f"provider raised {type(exc).__name__}: {exc}",
            )

    def _record(self, alert: AlertEvent, result: AlertResult) -> None:
        if self.repos is None:
            return
        try:
            self.repos.alerts.record(
                AlertRow(
                    event_uid=alert.event_uid,
                    provider=result.provider,
                    attempt_time=isoformat(result.attempt_time),
                    success=result.success,
                    failure_message=result.failure_message,
                    response_metadata=result.response_metadata,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Failed to record alert attempt for %s: %s", alert.event_uid, exc)

    async def aclose(self) -> None:
        for provider in self.providers:
            try:
                await provider.aclose()
            except Exception:  # pragma: no cover - defensive
                pass
