# A frozen snapshot of the scene settings. Lazy reads with per-site fallbacks
# let one setting carry several defaults: across 204 call sites `enabled` was
# read as both True and False.

from collections.abc import Mapping
from dataclasses import dataclass, replace as _replace
from types import MappingProxyType

# Prefix on every scene property read, so introspection can build the snapshot.
PREFIX = "scigraphs_preview_"


@dataclass(frozen=True)
class Clause:
    """A channel and a range over it, normalized to [0, 1]."""

    channel: str = 'DEGREE'
    attr_name: str = ""
    range_min: float = 0.0
    range_max: float = 1.0
    invert: bool = False
    mute: bool = False

    @property
    def narrowing(self):
        """True when this clause removes something; a full range is no clause."""
        return (not self.mute
                and (self.invert or self.range_min > 0.0 or self.range_max < 1.0))

    def signature(self):
        """The tuple a batch cache keys on; a muted clause never reaches it."""
        return (self.channel, self.attr_name,
                round(float(self.range_min), 5), round(float(self.range_max), 5),
                bool(self.invert))

    def as_dict(self):
        return {
            "channel": self.channel, "attr_name": self.attr_name,
            "range_min": float(self.range_min), "range_max": float(self.range_max),
            "invert": bool(self.invert), "mute": bool(self.mute),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            channel=d.get("channel", 'DEGREE'),
            attr_name=d.get("attr_name", ""),
            range_min=float(d.get("range_min", 0.0)),
            range_max=float(d.get("range_max", 1.0)),
            invert=bool(d.get("invert", False)),
            mute=bool(d.get("mute", False)),
        )


class Settings(Mapping):
    """An immutable snapshot; a Mapping and a namespace. Names lose the prefix."""

    __slots__ = ("_values", "_clauses", "_hash")

    def __init__(self, values=None, clauses=()):
        object.__setattr__(self, "_values",
                           MappingProxyType(dict(values or {})))
        object.__setattr__(self, "_clauses", tuple(clauses))
        object.__setattr__(self, "_hash", None)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __contains__(self, key):
        return key in self._values

    def __getattr__(self, name):
        # A missing setting raises: a typo would silently switch a filter off.
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(
                f"no setting {name!r} (have {len(self._values)})") from None

    def __setattr__(self, name, value):
        raise AttributeError("Settings is immutable; use .evolve(...)")

    @property
    def clauses(self):
        """Every clause in panel order, muted ones included."""
        return self._clauses

    @property
    def active_clauses(self):
        return tuple(c for c in self._clauses if c.narrowing)

    def clause_signature(self):
        return tuple(c.signature() for c in self.active_clauses)

    def evolve(self, **changes):
        """A copy with some settings replaced; ``clauses=`` is accepted too."""
        clauses = changes.pop("clauses", self._clauses)
        values = dict(self._values)
        unknown = set(changes) - set(values)
        if unknown:
            raise KeyError(f"unknown setting(s): {sorted(unknown)}")
        values.update(changes)
        return Settings(values, clauses)

    def signature(self, keys):
        """A hashable tuple over the named settings, in order. Call sites pick the
        keys: a threshold the vertex shader resolves must not force a rebuild."""
        return tuple(self._values.get(k) for k in keys)

    def to_dict(self):
        return {
            "values": dict(self._values),
            "clauses": [c.as_dict() for c in self._clauses],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("values", {}),
                   [Clause.from_dict(c) for c in d.get("clauses", ())])

    def __hash__(self):
        h = self._hash
        if h is None:
            h = hash((tuple(sorted(
                (k, _hashable(v)) for k, v in self._values.items())),
                self._clauses))
            object.__setattr__(self, "_hash", h)
        return h

    def __eq__(self, other):
        if not isinstance(other, Settings):
            return NotImplemented
        return (dict(self._values) == dict(other._values)
                and self._clauses == other._clauses)

    def __repr__(self):
        return (f"Settings({len(self._values)} values, "
                f"{len(self._clauses)} clauses)")


def _hashable(value):
    """Colors and vectors arrive as sequences; make them hashable."""
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value
