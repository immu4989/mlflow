from mlflow.mem0.autolog import patched_async_call
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "mem0"


@autologging_integration(FLAVOR_NAME)
def autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """Enable tracing for Mem0's ``AsyncMemory`` and ``AsyncMemoryClient`` APIs.

    Records ``add``, ``search``, ``get``, ``get_all``, ``update``, and ``delete``.
    Synchronous APIs are not instrumented by this implementation.

    Inputs and outputs can contain personal information and memory contents. Use
    :py:func:`mlflow.tracing.configure` with ``span_processors`` to mask sensitive
    fields before exporting traces, including the ``mem0.response`` attribute.

    Args:
        log_traces: Whether to record traces.
        disable: Whether to disable this integration.
        silent: Whether to suppress MLflow autologging warnings and event logs.
    """
    from mem0 import AsyncMemory, AsyncMemoryClient

    for cls in (AsyncMemory, AsyncMemoryClient):
        for operation in ("add", "search", "get", "get_all", "update", "delete"):
            safe_patch(FLAVOR_NAME, cls, operation, patched_async_call)
