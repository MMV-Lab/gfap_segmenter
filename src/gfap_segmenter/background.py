"""Qt-friendly helpers for running long-running tasks off the main thread."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal


class TaskSignals(QObject):
    progress = Signal(float)
    message = Signal(str)
    error = Signal(str)
    result = Signal(object)
    finished = Signal()


class TaskRunner(QRunnable):
    """QRunnable that executes a callable and communicates via signals."""

    def __init__(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()

    def run(self) -> None:  # pragma: no cover - executed in Qt thread pool
        try:
            # Use inspect.signature to get all parameters, including keyword-only ones
            try:
                sig = inspect.signature(self.fn)
                param_names = list(sig.parameters.keys())
            except (ValueError, TypeError):
                # Fallback for functions without signatures (e.g., builtins)
                if hasattr(self.fn, "__code__"):
                    param_names = list(
                        self.fn.__code__.co_varnames[
                            : self.fn.__code__.co_argcount
                        ]
                    )
                else:
                    param_names = []

            if "progress_callback" in param_names:
                self.kwargs.setdefault(
                    "progress_callback", self.signals.progress.emit
                )
            if "message_callback" in param_names:
                self.kwargs.setdefault(
                    "message_callback", self.signals.message.emit
                )
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as exc:  # pragma: no cover - error propagation
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class BackgroundExecutor:
    """Thin wrapper around :class:`QThreadPool` for convenience."""

    def __init__(self) -> None:
        self.pool = QThreadPool.globalInstance()

    def submit(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> TaskRunner:
        runner = TaskRunner(fn, *args, **kwargs)
        self.pool.start(runner)
        return runner
