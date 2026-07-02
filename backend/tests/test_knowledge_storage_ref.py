import importlib
import pytest


def _fresh_ref():
    import app.plugins.knowledge.storage_ref as ref
    return importlib.reload(ref)


def test_get_storage_raises_before_wired():
    ref = _fresh_ref()
    with pytest.raises(RuntimeError):
        ref.get_storage()


def test_set_then_get_returns_facade():
    ref = _fresh_ref()
    sentinel = object()
    ref.set_storage(sentinel)
    assert ref.get_storage() is sentinel


def test_register_wires_storage_from_container():
    from app.core.container import Container
    from app.core.contracts import STORAGE
    import app.plugins.knowledge as k
    ref = _fresh_ref()
    c = Container()
    sentinel = object()
    c.bind(STORAGE, sentinel)

    class Ctx:
        container = c
        events = None
        session_factory = None
        settings = None
        config = {}

    k.register(Ctx)
    assert ref.get_storage() is sentinel
