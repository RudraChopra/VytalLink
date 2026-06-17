"""Factory that assembles the configured alert providers + dispatcher."""

from __future__ import annotations

from vytallink.alerts.base import AlertProvider
from vytallink.alerts.console import ConsoleAlertProvider
from vytallink.alerts.dispatcher import AlertDispatcher
from vytallink.alerts.webhook import WebhookAlertProvider
from vytallink.common.clock import Clock
from vytallink.common.logging_setup import get_logger
from vytallink.config import Settings
from vytallink.database.repositories import Repositories

log = get_logger("alerts.factory")


def build_dispatcher(
    settings: Settings, repos: Repositories, clock: Clock | None = None
) -> AlertDispatcher:
    providers: list[AlertProvider] = []
    if not settings.alerts_enabled:
        log.info("Alerts disabled via ALERTS_ENABLED=false; events recorded but never delivered")
        return AlertDispatcher(providers, repos, clock=clock)
    if settings.console_alerts_enabled:
        providers.append(ConsoleAlertProvider(clock=clock))
    if settings.webhook_enabled:
        providers.append(
            WebhookAlertProvider(
                settings.webhook_url,
                settings.webhook_secret,
                timeout=settings.webhook_timeout_seconds,
                clock=clock,
            )
        )
    if not providers:
        log.warning("No alert providers enabled; alerts will be recorded but not delivered")
    return AlertDispatcher(providers, repos, clock=clock)
