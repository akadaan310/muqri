"""Capability-addressed collection operations, not unsafe process-memory pointers.

A client supplies an opaque handle and scope, then selects a sorting policy.
Shape, axis and comparator must be explicit for tensors. The prior immutable
handle is a reversible snapshot of the collection; it is NOT a reversible sort
permutation unless an index-map is retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import secrets


class CollectionError(ValueError):
    pass


def _shape(x) -> tuple[int, ...]:
    if not isinstance(x, (list, tuple)):
        if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
            return ()
        raise CollectionError("Collections must contain finite numeric scalars")
    if not x:
        return (0,)
    shapes = [_shape(item) for item in x]
    if any(s != shapes[0] for s in shapes):
        raise CollectionError("Non-rectangular collection requires explicit ragged policy")
    return (len(x),) + shapes[0]


def _sum(x) -> float:
    if isinstance(x, list):
        return sum(_sum(y) for y in x)
    return float(x)


def _sort_axis(x: list, axis: int, policy: str) -> list:
    if axis == 0:
        if policy == "numeric_scalar":
            if x and isinstance(x[0], list):
                raise CollectionError("numeric_scalar requires scalar elements on sorting axis")
            return sorted(x)
        if policy == "aggregate_sum":
            return sorted(x, key=_sum)
        raise CollectionError("Unknown ordering policy")
    return [_sort_axis(item, axis - 1, policy) for item in x]


@dataclass(frozen=True, slots=True)
class Address:
    handle: str
    shape: tuple[int, ...]
    source_digest: str
    scope: str
    parent: str | None


class CollectionStore:
    """In-process scoped store; raw operating-system pointers are never accepted."""

    def __init__(self, *, max_cells: int = 1_000_000):
        self._values: dict[str, list] = {}
        self._metadata: dict[str, Address] = {}
        self.max_cells = max_cells

    def _get(self, handle: str, scope: str) -> list:
        address = self._metadata.get(handle)
        if address is None or address.scope != scope:
            raise PermissionError("No capability for requested collection")
        return self._values[handle]

    def register(self, collection: list, *, scope: str = "local", parent: str | None = None) -> Address:
        if not scope or not isinstance(collection, list):
            raise CollectionError("Explicit scope and list-shaped collection required")
        shape = _shape(collection)
        count = math.prod(shape)
        if count > self.max_cells:
            raise CollectionError("RESOURCE_LIMIT: collection too large")
        raw = json.dumps(collection, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        immutable_copy = json.loads(raw)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        handle = "unison://collection/" + secrets.token_urlsafe(24)
        address = Address(handle, shape, digest, scope, parent)
        self._values[handle] = immutable_copy
        self._metadata[handle] = address
        return address

    def shape(self, handle: str, *, scope: str = "local") -> tuple[int, ...]:
        self._get(handle, scope)
        return self._metadata[handle].shape

    def read(self, handle: str, *, scope: str = "local") -> list:
        """Return a COPY; changes cannot mutate the stored parent."""
        return json.loads(json.dumps(self._get(handle, scope)))

    def sort_options(self, handle: str, *, axis: int, scope: str = "local") -> tuple[dict, ...]:
        value = self._get(handle, scope)
        shape = _shape(value)
        if not shape or not 0 <= axis < len(shape):
            raise CollectionError("Specify a valid axis of the stored shape")
        n = shape[axis]
        runs = math.prod(shape[:axis])
        return (
            {"policy": "numeric_scalar", "admissible": axis == len(shape) - 1,
             "time_upper": f"O({runs}*{n}*log({max(n, 2)}))", "stable": True,
             "loss": "original order not recovered from sorted result alone"},
            {"policy": "aggregate_sum", "admissible": True,
             "time_upper": "O(total input cells + slices*n*log(n))", "stable": True,
             "loss": "sorting by sum is many-to-one as a ranking key"},
        )

    def sorted_copy(
        self, handle: str, *,
        axis: int, policy: str = "numeric_scalar", scope: str = "local",
    ) -> Address:
        source = self.read(handle, scope=scope)
        shape = _shape(source)
        if not shape or not 0 <= axis < len(shape):
            raise CollectionError("Valid explicit sorting axis required")
        new = _sort_axis(source, axis, policy)
        if _shape(new) != shape:
            raise CollectionError("Sort changed collection shape")
        return self.register(new, scope=scope, parent=handle)
