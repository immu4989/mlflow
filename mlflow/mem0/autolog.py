import asyncio
import inspect

from pydantic import BaseModel

import mlflow
from mlflow.entities import Document, SpanType
from mlflow.utils.autologging_utils import get_autologging_config

_READ_OPERATIONS = {"search", "get", "get_all"}


def _get_inputs(original, instance, args, kwargs):
    signature = inspect.signature(original)
    arguments = signature.bind_partial(instance, *args, **kwargs).arguments
    arguments.pop("self", None)
    inputs = {}
    for name, value in arguments.items():
        if signature.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
            inputs.update(value)
        else:
            inputs[name] = value
    if isinstance(inputs.get("options"), BaseModel):
        inputs["options"] = inputs["options"].model_dump(exclude_unset=True)
    return inputs


def _get_attributes(operation, inputs):
    # Explicit kwargs override typed options in the hosted SDK, including filters.
    parameters = {**(inputs.get("options") or {}), **inputs}
    filters = parameters.get("filters") or {}
    attributes = {"mem0.operation": operation}
    for key in ("user_id", "agent_id", "run_id"):
        value = parameters.get(key, filters.get(key))
        # Compound filters can refer to multiple scopes. Keep them in inputs rather
        # than incorrectly assigning a single identity to the span.
        if isinstance(value, str):
            attributes[f"mem0.{key}"] = value
    if "memory_id" in parameters:
        attributes["mem0.memory_id"] = parameters["memory_id"]
    return attributes


def _to_documents(result):
    if result is None:
        return []
    records = result.get("results", [result]) if isinstance(result, dict) else result
    return [
        Document(
            id=record.get("id"),
            page_content=record.get("memory", ""),
            metadata={key: value for key, value in record.items() if key not in {"id", "memory"}},
        )
        for record in records
    ]


async def patched_async_call(original, instance, *args, **kwargs):
    if not get_autologging_config("mem0", "log_traces", True):
        return await original(instance, *args, **kwargs)

    operation = original.__name__
    inputs = _get_inputs(original, instance, args, kwargs)
    is_read = operation in _READ_OPERATIONS
    with mlflow.start_span(
        name=f"{type(instance).__name__}.{operation}",
        span_type=SpanType.RETRIEVER if is_read else SpanType.MEMORY,
        attributes=_get_attributes(operation, inputs),
    ) as span:
        span.set_inputs(inputs)
        try:
            result = await original(instance, *args, **kwargs)
        except asyncio.CancelledError as error:
            # CancelledError is a BaseException and is not recorded by start_span.
            try:
                span.record_exception(f"CancelledError: {error}")
            finally:
                # A tracing failure must not turn cancellation into a successful call.
                raise error
        if is_read:
            # Preserve pagination and graph data alongside the normalized documents.
            span.set_attribute("mem0.response", result)
            span.set_outputs(_to_documents(result))
        else:
            span.set_outputs(result)
        return result
