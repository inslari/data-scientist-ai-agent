"""Domain pack registry + active-pack resolution.

Packs are resolved by name. The name comes from (in priority order):
    1. an explicit ``name`` argument,
    2. ``state["domain_pack"]`` (per-run override set at graph entry),
    3. ``settings.DOMAIN_PACK`` (global default, env ``DOMAIN_PACK``).

Unknown / "none" / empty always resolves to :class:`NullDomainPack`, so callers
never have to special-case "no pack". Pack instances are cached per name.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from src.core.config import settings

from .base import DomainPack
from .null_pack import NullDomainPack

# name → factory (lazy so importing the registry doesn't import every pack/LLM)
_FACTORIES: dict[str, Callable[[], DomainPack]] = {}
_INSTANCES: dict[str, DomainPack] = {}

_NULL = NullDomainPack()


def register_pack(name: str, factory: Callable[[], DomainPack]) -> None:
    """Register a pack factory under a (case-insensitive) name."""
    _FACTORIES[name.strip().lower()] = factory


def _default_factories() -> None:
    """Built-in packs. Imported lazily to avoid import cycles / LLM cost."""
    if "financial" not in _FACTORIES:
        def _make_financial() -> DomainPack:
            from .financial.pack import FinancialPack
            return FinancialPack()

        _FACTORIES["financial"] = _make_financial


def _resolve_name(name_or_state: Union[str, dict, None]) -> str:
    if isinstance(name_or_state, dict):
        name = name_or_state.get("domain_pack")
    else:
        name = name_or_state
    if not name:
        name = settings.DOMAIN_PACK
    return (name or "none").strip().lower()


def get_active_pack(name_or_state: Union[str, dict, None] = None) -> DomainPack:
    """Return the active pack for a name or a state dict.

    Falls back to :class:`NullDomainPack` for "none"/unknown names. Never raises:
    if a pack fails to construct, logs and returns the null pack.
    """
    _default_factories()
    name = _resolve_name(name_or_state)

    if name in ("", "none", "null", "off", "disabled"):
        return _NULL

    if name in _INSTANCES:
        return _INSTANCES[name]

    factory = _FACTORIES.get(name)
    if factory is None:
        print(f"[domains] Unknown domain pack '{name}', falling back to NullDomainPack.")
        return _NULL

    try:
        pack = factory()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[domains] Failed to construct domain pack '{name}': {exc}. Using NullDomainPack.")
        return _NULL

    _INSTANCES[name] = pack
    return pack


def list_packs() -> list[dict]:
    """Return all selectable packs as ``[{"name", "label"}]`` for UI dropdowns.

    Always begins with the null pack ("none"). Newly registered sectors appear
    automatically. A pack that fails to construct is skipped (not surfaced).
    """
    _default_factories()
    packs = [{"name": _NULL.name, "label": _NULL.label}]
    for name in sorted(_FACTORIES.keys()):
        pack = get_active_pack(name)
        if pack is _NULL:
            continue
        packs.append({"name": pack.name, "label": pack.label})
    return packs
