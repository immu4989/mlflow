import asyncio
import json
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from mem0 import AsyncMemory, AsyncMemoryClient
from mem0.client.types import (
    AddMemoryOptions,
    GetAllMemoryOptions,
    SearchMemoryOptions,
    UpdateMemoryOptions,
)

import mlflow.mem0
from mlflow.entities import SpanStatusCode, SpanType
from mlflow.mem0.autolog import _to_documents

from tests.tracing.helper import get_traces

pytestmark = pytest.mark.asyncio

OPERATIONS = ["add", "search", "get", "get_all", "update", "delete"]
MEMORY = {
    "id": "memory-1",
    "memory": "Prefers tea",
    "score": 0.9,
    "user_id": "user-1",
    "created_at": "2026-09-01T00:00:00Z",
    "metadata": {"source": "chat", "score": "user-defined"},
}


@pytest.fixture(autouse=True)
def reset_autolog(monkeypatch):
    yield
    mlflow.mem0.autolog(disable=True)


@pytest.fixture(params=[AsyncMemory, AsyncMemoryClient], ids=lambda cls: cls.__name__)
def memory_class(request):
    return request.param


@pytest.fixture
def instrument(monkeypatch):
    def create(cls, operation, result=None, side_effect=None):
        call = AsyncMock(return_value=result, side_effect=side_effect)

        @wraps(getattr(cls, operation))
        async def original(self, *args, **kwargs):
            return await call(*args, **kwargs)

        monkeypatch.setattr(cls, operation, original)
        mlflow.mem0.autolog()
        return object.__new__(cls), call

    return create


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_operations(memory_class, instrument, operation):
    result = deepcopy(MEMORY) if operation == "get" else {"results": [deepcopy(MEMORY)]}
    before = deepcopy(result)
    instance, call = instrument(memory_class, operation, result)
    args = () if operation == "get_all" else ("memory-1",)
    kwargs = {"text": "Prefers coffee"} if operation == "update" else {}

    returned = await getattr(instance, operation)(*args, **kwargs)

    assert returned is result
    assert result == before
    call.assert_awaited_once_with(*args, **kwargs)
    [trace] = get_traces()
    [span] = trace.data.spans
    assert span.name == f"{memory_class.__name__}.{operation}"
    assert span.attributes["mem0.operation"] == operation
    assert span.status.status_code == SpanStatusCode.OK
    if operation in {"search", "get", "get_all"}:
        assert span.span_type == SpanType.RETRIEVER
        assert span.outputs == [
            {
                "id": MEMORY["id"],
                "page_content": MEMORY["memory"],
                "metadata": {k: v for k, v in MEMORY.items() if k not in {"id", "memory"}},
            }
        ]
        assert span.attributes["mem0.response"] == result
    else:
        assert span.span_type == SpanType.MEMORY
        assert span.outputs == result
    if operation in {"get", "update", "delete"}:
        assert span.attributes["mem0.memory_id"] == "memory-1"
        assert span.inputs["memory_id"] == "memory-1"


async def test_local_add_scope(instrument):
    instance, call = instrument(AsyncMemory, "add", {"results": []})
    kwargs = {"user_id": "u", "agent_id": "a", "run_id": "r", "infer": False}
    await instance.add("Prefers tea", **kwargs)
    call.assert_awaited_once_with("Prefers tea", **kwargs)
    [span] = get_traces()[0].data.spans
    assert span.inputs == {"messages": "Prefers tea", **kwargs}
    for key in ("user_id", "agent_id", "run_id"):
        assert span.attributes[f"mem0.{key}"] == kwargs[key]


@pytest.mark.parametrize("operation", ["search", "get_all"])
async def test_filter_scope(memory_class, instrument, operation):
    instance, call = instrument(memory_class, operation, {"results": []})
    filters = {"user_id": "u", "agent_id": "a", "run_id": "r"}
    args = ("tea",) if operation == "search" else ()
    await getattr(instance, operation)(*args, filters=filters, top_k=3)
    call.assert_awaited_once_with(*args, filters=filters, top_k=3)
    [span] = get_traces()[0].data.spans
    assert span.inputs["filters"] == filters
    assert span.inputs["top_k"] == 3
    assert "kwargs" not in span.inputs
    for key, value in filters.items():
        assert span.attributes[f"mem0.{key}"] == value


@pytest.mark.parametrize(
    ("operation", "options_class", "args"),
    [
        ("add", AddMemoryOptions, ("Prefers tea",)),
        ("search", SearchMemoryOptions, ("tea",)),
        ("get_all", GetAllMemoryOptions, ()),
    ],
)
@pytest.mark.parametrize("positional", [True, False])
async def test_typed_options(instrument, operation, options_class, args, positional):
    instance, call = instrument(AsyncMemoryClient, operation, {"results": []})
    options = options_class(filters={"user_id": "u", "agent_id": "a", "run_id": "r"})
    before = options.model_dump()
    call_args = (*args, options) if positional else args
    call_kwargs = {} if positional else {"options": options}
    await getattr(instance, operation)(*call_args, **call_kwargs)
    call.assert_awaited_once_with(*call_args, **call_kwargs)
    assert options.model_dump() == before
    [span] = get_traces()[0].data.spans
    assert span.inputs["options"] == options.model_dump(exclude_unset=True)
    for key, value in options.filters.items():
        assert span.attributes[f"mem0.{key}"] == value


async def test_kwargs_replace_options_filters(instrument):
    instance, call = instrument(AsyncMemoryClient, "search", {"results": []})
    options = SearchMemoryOptions(filters={"user_id": "old", "agent_id": "old-agent"})
    await instance.search("tea", options, filters={"user_id": "new"})
    call.assert_awaited_once_with("tea", options, filters={"user_id": "new"})
    [span] = get_traces()[0].data.spans
    assert span.attributes["mem0.user_id"] == "new"
    assert "mem0.agent_id" not in span.attributes
    assert options.filters == {"user_id": "old", "agent_id": "old-agent"}


async def test_compound_filter_does_not_invent_scope(memory_class, instrument):
    instance, _ = instrument(memory_class, "search", {"results": []})
    filters = {"OR": [{"user_id": "u1"}, {"user_id": "u2"}], "agent_id": {"in": ["a", "b"]}}
    await instance.search("tea", filters=filters)
    [span] = get_traces()[0].data.spans
    assert span.inputs["filters"] == filters
    assert "mem0.user_id" not in span.attributes
    assert "mem0.agent_id" not in span.attributes


@pytest.mark.parametrize("result", [None, [], {"results": []}])
async def test_empty_retrieval(memory_class, instrument, result):
    instance, _ = instrument(memory_class, "get", result)
    assert await instance.get("missing") is result
    [span] = get_traces()[0].data.spans
    assert span.outputs == []


async def test_paginated_and_graph_response(instrument):
    result = {"count": 2, "next": "next-page", "results": [deepcopy(MEMORY)], "relations": []}
    instance, _ = instrument(AsyncMemoryClient, "get_all", result)
    assert await instance.get_all() is result
    [span] = get_traces()[0].data.spans
    assert len(span.outputs) == 1
    assert span.attributes["mem0.response"] == result


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_sdk_error_propagates(memory_class, instrument, operation):
    error = ValueError("SDK failure")
    instance, call = instrument(memory_class, operation, side_effect=error)
    args = () if operation == "get_all" else ("memory-1",)
    with pytest.raises(ValueError, match="SDK failure") as exc:
        await getattr(instance, operation)(*args)
    assert exc.value is error
    call.assert_awaited_once_with(*args)
    [span] = get_traces()[0].data.spans
    assert span.status.status_code == SpanStatusCode.ERROR
    assert mlflow.get_current_active_span() is None


async def test_cancellation_ends_span(memory_class, instrument):
    started = asyncio.Event()

    async def wait_for_cancellation(*args):
        started.set()
        await asyncio.Event().wait()

    instance, call = instrument(memory_class, "get", side_effect=wait_for_cancellation)
    task = asyncio.create_task(instance.get("memory-1"))
    await started.wait()
    task.cancel("test cancellation")
    with pytest.raises(asyncio.CancelledError, match="test cancellation"):
        await task
    call.assert_awaited_once_with("memory-1")
    [span] = get_traces()[0].data.spans
    assert span.status.status_code == SpanStatusCode.ERROR
    assert span.end_time_ns is not None
    assert mlflow.get_current_active_span() is None


async def test_concurrent_calls_preserve_parent(memory_class, instrument):
    both_started = asyncio.Event()
    active_spans = {}

    async def retrieve(memory_id):
        active_spans[memory_id] = mlflow.get_current_active_span()
        if len(active_spans) == 2:
            both_started.set()
        await both_started.wait()
        assert mlflow.get_current_active_span().span_id == active_spans[memory_id].span_id
        return {"id": memory_id, "memory": memory_id}

    instance, call = instrument(memory_class, "get", side_effect=retrieve)
    with mlflow.start_span("parent") as parent:
        results = await asyncio.gather(instance.get("0"), instance.get("1"))
        assert mlflow.get_current_active_span().span_id == parent.span_id
    assert [result["id"] for result in results] == ["0", "1"]
    assert call.await_count == 2
    [trace] = get_traces()
    assert len(trace.data.spans) == 3
    assert len({span.span_id for span in active_spans.values()}) == 2
    assert all(span.parent_id == parent.span_id for span in active_spans.values())


@pytest.mark.parametrize("config", [{"disable": True}, {"log_traces": False}])
async def test_disabled(memory_class, instrument, config):
    result = deepcopy(MEMORY)
    instance, call = instrument(memory_class, "get", result)
    mlflow.mem0.autolog(**config)
    assert await instance.get("memory-1") is result
    call.assert_awaited_once_with("memory-1")
    assert get_traces() == []


async def test_repeated_enablement(memory_class, instrument):
    instance, call = instrument(memory_class, "get", MEMORY)
    mlflow.mem0.autolog()
    mlflow.mem0.autolog()
    await instance.get("memory-1")
    call.assert_awaited_once_with("memory-1")
    assert len(get_traces()[0].data.spans) == 1


async def test_conversion_failure_does_not_repeat_sdk_call(memory_class, instrument, monkeypatch):
    result = {"results": [deepcopy(MEMORY)]}
    instance, call = instrument(memory_class, "search", result)
    monkeypatch.setenv("MLFLOW_AUTOLOGGING_TESTING", "false")

    with patch("mlflow.mem0.autolog._to_documents", side_effect=RuntimeError("conversion failed")):
        assert await instance.search("tea") is result
    call.assert_awaited_once_with("tea")
    assert mlflow.get_current_active_span() is None


async def test_input_logging_failure_still_calls_sdk(memory_class, instrument, monkeypatch):
    instance, call = instrument(memory_class, "get", MEMORY)
    monkeypatch.setenv("MLFLOW_AUTOLOGGING_TESTING", "false")
    with patch("mlflow.mem0.autolog._get_inputs", side_effect=RuntimeError("input logging failed")):
        assert await instance.get("memory-1") is MEMORY
    call.assert_awaited_once_with("memory-1")
    assert mlflow.get_current_active_span() is None


async def test_cancellation_survives_tracing_failure(memory_class, instrument, monkeypatch):
    error = asyncio.CancelledError("cancelled by caller")
    instance, call = instrument(memory_class, "get", side_effect=error)
    monkeypatch.setenv("MLFLOW_AUTOLOGGING_TESTING", "false")
    with (
        patch(
            "mlflow.entities.span.LiveSpan.record_exception", side_effect=RuntimeError("tracing")
        ),
        pytest.raises(asyncio.CancelledError, match="cancelled by caller") as exc,
    ):
        await instance.get("memory-1")
    assert exc.value is error
    call.assert_awaited_once_with("memory-1")
    assert mlflow.get_current_active_span() is None


@pytest.mark.parametrize("found", [True, False])
async def test_local_sdk_get_with_mock_vector_store(monkeypatch, found):
    instance = object.__new__(AsyncMemory)
    record = SimpleNamespace(
        id="memory-1", payload={"data": "Prefers tea", "user_id": "u", "source": "chat"}
    )
    instance.vector_store = Mock()
    instance.vector_store.get.return_value = record if found else None
    monkeypatch.setattr("mem0.memory.main.capture_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("mem0.memory.main.display_first_run_notice_async", AsyncMock())
    mlflow.mem0.autolog()
    result = await instance.get("memory-1")
    instance.vector_store.get.assert_called_once_with(vector_id="memory-1")
    [span] = get_traces()[0].data.spans
    if found:
        assert result["memory"] == "Prefers tea"
        assert span.outputs[0]["page_content"] == result["memory"]
        assert span.outputs[0]["metadata"]["metadata"] == {"source": "chat"}
    else:
        assert result is None
        assert span.outputs == []


async def test_legacy_list_response():
    assert _to_documents([MEMORY])[0].id == MEMORY["id"]


@pytest.mark.parametrize(
    ("operation", "args", "kwargs", "http_method", "path", "payload"),
    [
        (
            "add",
            ("Prefers tea", AddMemoryOptions(filters={"user_id": "u"}, infer=False)),
            {},
            "POST",
            "/v3/memories/add/",
            {
                "messages": [{"role": "user", "content": "Prefers tea"}],
                "filters": {"user_id": "u"},
                "infer": False,
            },
        ),
        (
            "search",
            ("tea", SearchMemoryOptions(filters={"user_id": "old"})),
            {"filters": {"user_id": "u"}},
            "POST",
            "/v3/memories/search/",
            {"query": "tea", "filters": {"user_id": "u"}},
        ),
        ("get", ("memory-1",), {}, "GET", "/v1/memories/memory-1/", None),
        (
            "get_all",
            (GetAllMemoryOptions(filters={"user_id": "u"}, page=2),),
            {},
            "POST",
            "/v3/memories/",
            {"filters": {"user_id": "u"}},
        ),
        (
            "update",
            ("memory-1", UpdateMemoryOptions(text="Prefers coffee", expiration_date=None)),
            {},
            "PUT",
            "/v1/memories/memory-1/",
            {"text": "Prefers coffee", "expiration_date": None},
        ),
        (
            "delete",
            ("memory-1",),
            {"delete_linked": True},
            "DELETE",
            "/v1/memories/memory-1/",
            None,
        ),
    ],
)
async def test_hosted_sdk_with_mock_transport(
    monkeypatch, operation, args, kwargs, http_method, path, payload
):
    requests = []
    response = deepcopy(MEMORY) if operation == "get" else {"results": [deepcopy(MEMORY)]}

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response)

    def validate_api_key(self):
        self.org_id = "test-org"
        self.project_id = "test-project"
        return "test@example.com"

    monkeypatch.setattr(AsyncMemoryClient, "_validate_api_key", validate_api_key)
    monkeypatch.setattr("mem0.client.main.capture_client_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("mem0.client.main._maybe_alias_anon_to_email", lambda *args: None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        client = AsyncMemoryClient(api_key="test-key", client=http_client)
        mlflow.mem0.autolog()
        result = await getattr(client, operation)(*args, **kwargs)
    assert result == response
    assert len(requests) == 1
    assert requests[0].url.path == path
    assert requests[0].method == http_method
    if payload is not None:
        assert json.loads(requests[0].content) == payload
    if operation == "get_all":
        assert requests[0].url.params["page"] == "2"
    if operation == "delete":
        assert requests[0].url.params["delete_linked"] == "true"
    [span] = get_traces()[0].data.spans
    if operation in {"add", "search", "get_all"}:
        assert span.attributes["mem0.user_id"] == "u"
    if operation in {"search", "get", "get_all"}:
        assert span.outputs[0]["page_content"] == "Prefers tea"
    else:
        assert span.outputs == response
