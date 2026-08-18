# Seed management and context managers for deterministic execution.

import random
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

# NumPy is bundled with Blender
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class SeedContext:
    """Seeds Python and NumPy on entry, restores the previous state on exit."""

    def __init__(self, seed: int):
        self.seed = seed
        self._python_state: Optional[tuple] = None
        self._numpy_state: Optional[Dict[str, Any]] = None

    def __enter__(self) -> "SeedContext":
        self._python_state = random.getstate()
        if HAS_NUMPY:
            self._numpy_state = np.random.get_state()

        random.seed(self.seed)
        if HAS_NUMPY:
            np.random.seed(self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._python_state is not None:
            random.setstate(self._python_state)
        if HAS_NUMPY and self._numpy_state is not None:
            np.random.set_state(self._numpy_state)


@contextmanager
def get_seed_context(seed: int) -> Generator[SeedContext, None, None]:
    """Run a block deterministically, then put the previous state back.

        with get_seed_context(42):
            value = random.random()
    """
    ctx = SeedContext(seed)
    with ctx:
        yield ctx


def set_deterministic_seed(seed: int) -> None:
    """Seed Python and NumPy globally, until something reseeds them.

    Use get_seed_context() when the effect should be temporary.
    """
    random.seed(seed)
    if HAS_NUMPY:
        np.random.seed(seed)


def derive_seed(base_seed: int, *components: str) -> int:
    """Derive a seed from a base seed and string components.

    Gives layout, sampling and geometry seeds that are independent of each
    other but reproducible from the same base.
    """
    import hashlib
    combined = f"{base_seed}:" + ":".join(components)
    hash_bytes = hashlib.sha256(combined.encode()).digest()
    # First 4 bytes as an unsigned int, kept inside the positive range.
    return int.from_bytes(hash_bytes[:4], 'big') % (2**31)


class DeterministicGenerator:
    """Reproducible random values that leave the global RNG state alone."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)
        if HAS_NUMPY:
            self._np_rng = np.random.RandomState(seed)
        else:
            self._np_rng = None

    def random(self) -> float:
        """Return random float in [0, 1)."""
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        """Return random integer in [a, b]."""
        return self._rng.randint(a, b)

    def choice(self, seq):
        """Return random element from sequence."""
        return self._rng.choice(seq)

    def shuffle(self, seq) -> None:
        """Shuffle sequence in place."""
        self._rng.shuffle(seq)

    def uniform(self, a: float, b: float) -> float:
        """Return random float in [a, b]."""
        return self._rng.uniform(a, b)

    def gauss(self, mu: float, sigma: float) -> float:
        """Return Gaussian random value."""
        return self._rng.gauss(mu, sigma)

    def numpy_random(self, *args, **kwargs):
        """NumPy-compatible random array."""
        if self._np_rng is None:
            raise RuntimeError("NumPy not available")
        return self._np_rng.random(*args, **kwargs)

    def numpy_normal(self, loc: float = 0.0, scale: float = 1.0, size=None):
        """NumPy-compatible normal distribution."""
        if self._np_rng is None:
            raise RuntimeError("NumPy not available")
        return self._np_rng.normal(loc, scale, size)

    def numpy_uniform(self, low: float = 0.0, high: float = 1.0, size=None):
        """NumPy-compatible uniform distribution."""
        if self._np_rng is None:
            raise RuntimeError("NumPy not available")
        return self._np_rng.uniform(low, high, size)


# Set by the executor before a run.
_current_pipeline_seed: Optional[int] = None


def get_pipeline_seed() -> Optional[int]:
    """Get the current pipeline seed, if set."""
    return _current_pipeline_seed


def set_pipeline_seed(seed: Optional[int]) -> None:
    """Set the current pipeline seed (used by executor)."""
    global _current_pipeline_seed
    _current_pipeline_seed = seed
    if seed is not None:
        set_deterministic_seed(seed)


def get_layout_seed(override: Optional[int] = None) -> int:
    """Seed for layout algorithms. The layout spec can override it."""
    if override is not None:
        return override
    base = _current_pipeline_seed or 42
    return derive_seed(base, "layout")


def get_geometry_seed(override: Optional[int] = None) -> int:
    """Seed for geometry generation, such as initial positions."""
    if override is not None:
        return override
    base = _current_pipeline_seed or 42
    return derive_seed(base, "geometry")


def get_sampling_seed(override: Optional[int] = None) -> int:
    """Seed for sampling operations, such as OSMnx OD sampling."""
    if override is not None:
        return override
    base = _current_pipeline_seed or 42
    return derive_seed(base, "sampling")
