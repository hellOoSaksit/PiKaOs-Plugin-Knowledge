"""Consumer-driven contract test (plugin-architecture.md §13) — the knowledge plugin proves it still
honors the `knowledge.Retriever` contract the AI engine (consumer) depends on, across the no-import
boundary. Runs knowledge's real register() and asserts the binding satisfies the engine's interface, so
knowledge can never silently drop/reshape `retrieve_context` without this turning red. Also pins the
job-contribution wiring the worker relies on.

In-process (no DB): register() only binds a constructed retriever; retrieve_context is never called.
Runs inside the Core stack with the knowledge plugin linked.

    docker compose exec backend pytest app/plugins/knowledge/tests/test_retriever_contract.py
"""
from __future__ import annotations

import inspect

from app import modules, plugin_loader
from app.core import contracts
from app.core.container import Container
from app.core.events import EventBus
from app.core.contracts import Retriever


def _ctx() -> plugin_loader.PluginContext:
    return plugin_loader.PluginContext(container=Container(), events=EventBus())


def test_knowledge_provides_retriever_contract():
    ctx = _ctx()
    plugin_loader.register_plugins({"knowledge"}, modules.PLUGIN_MANIFESTS, ctx)

    impl = ctx.container.resolve(contracts.RETRIEVER)
    assert impl is not None, "knowledge.register() must bind the knowledge.Retriever contract"
    assert isinstance(impl, Retriever), "bound retriever does not satisfy the engine's Retriever (§13)"
    assert inspect.iscoroutinefunction(impl.retrieve_context), "retrieve_context must be async"
    params = inspect.signature(impl.retrieve_context).parameters
    assert {"db", "owner_id", "run_input", "k"} <= set(params), "contract signature drifted from consumer"


def test_manifest_declares_what_it_provides():
    """The runtime binding must match the manifest's `provides` (the static contract Phase-2 validates)."""
    mf = modules.PLUGIN_MANIFESTS["knowledge"]
    assert contracts.RETRIEVER in mf.provides


def test_enabled_knowledge_contributes_its_job():
    jobs = plugin_loader.collect_jobs({"knowledge"}, modules.PLUGIN_MANIFESTS)
    assert [j.__name__ for j in jobs] == ["ingest_document"]
