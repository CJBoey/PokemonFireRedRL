"""Relay scopes: six regions of the game in story order that a life is confined to while it works on them
(env.py "Relay scopes").

* :mod:`.scopes`: the six scopes as data, predicates over ``events.EVENTS``, areas = anchors + warp closure,
  validation, the per-env status transitions (``ScopeSet.step``).
* :mod:`.exits`: a scope's area exits as D-pad presses, the table of the exit mask.

    PYTHONPATH=rl:python python3 -m pokerl.relay.scopes      # prints every scope's resolved area
"""
from .scopes import (ACTIVE, CLEARED, INACTIVE, SCOPE_NAMES, SCOPES, Predicate, Scope, ScopeSet, active_scope,
                     area_maps, closure, holds, map_data, resolve_predicate, scope_set)

__all__ = ["ACTIVE", "CLEARED", "INACTIVE", "SCOPE_NAMES", "SCOPES", "Predicate", "Scope", "ScopeSet",
           "active_scope", "area_maps", "closure", "holds", "map_data", "resolve_predicate", "scope_set"]
