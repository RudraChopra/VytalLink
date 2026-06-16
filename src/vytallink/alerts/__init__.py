"""Alert subsystem: provider interface, console + webhook providers, dispatcher."""

from vytallink.alerts.base import (
    AlertDispatcherProtocol,
    AlertEvent,
    AlertProvider,
    AlertResult,
)
from vytallink.alerts.console import ConsoleAlertProvider
from vytallink.alerts.dispatcher import AlertDispatcher
from vytallink.alerts.factory import build_dispatcher
from vytallink.alerts.webhook import WebhookAlertProvider

__all__ = [
    "AlertEvent",
    "AlertResult",
    "AlertProvider",
    "AlertDispatcherProtocol",
    "ConsoleAlertProvider",
    "WebhookAlertProvider",
    "AlertDispatcher",
    "build_dispatcher",
]
