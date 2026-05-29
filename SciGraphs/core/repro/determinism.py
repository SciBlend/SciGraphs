# Determinism utilities for reproducible SciGraphs workflows
#
# Provides seed management and context managers for deterministic execution.

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
    """
    Context for managing deterministic random state.

    Captures and restores random state when entering/exiting context.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self._python_state: Optional[tuple] = None
        self._numpy_state: Optional[Dict[str, Any]] = None

    def __enter__(self) -> "SeedContext":
        # Save current states
        self._python_state = random.getstate()
        if HAS_NUMPY:
            self._numpy_state = np.random.get_state()

        # Set seeds
        random.seed(self.seed)
        if HAS_NUMPY:
            np.random.seed(self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore previous states
        if self._python_state is not None:
            random.setstate(self._python_state)
        if HAS_NUMPY and self._numpy_state is not None:
            np.random.set_state(self._numpy_state)


@contextmanager
def get_seed_context(seed: int) -> Generator[SeedContext, None, None]:
    """
    Context manager for deterministic execution with a specific seed.

    Usage:
        with get_seed_context(42) as ctx:
            # All random operations here are deterministic
            value = random.random()

    Args:
        seed: Random seed to use

    Yields:
        SeedContext instance
    """
    ctx = SeedContext(seed)
    with ctx:
        yield ctx


def set_deterministic_seed(seed: int) -> None:
    """
    Set global random seeds for Python and NumPy.

    This affects all subsequent random operations until seeds are changed.
    For temporary deterministic execution, use get_seed_context() instead.

    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    if HAS_NUMPY:
        np.random.seed(seed)


def derive_seed(base_seed: int, *components: str) -> int:
    """
    Derive a deterministic seed from a base seed and string components.

    Useful for creating independent but reproducible seeds for different
    parts of a pipeline (e.g., layout, sampling, geometry).

    Args:
        base_seed: Base random seed
        *components: String identifiers to incorporate

    Returns:
        Derived seed value
    """
    import hashlib
    combined = f"{base_seed}:" + ":".join(components)
    hash_bytes = hashlib.sha256(combined.encode()).digest()
    # Use first 4 bytes as unsigned int, clamped to positive range
    return int.from_bytes(hash_bytes[:4], 'big') % (2**31)


class DeterministicGenerator:
    """
    A simple deterministic random generator that doesn't affect global state.

    Useful when you need reproducible random values without side effects.
    """

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


# Global pipeline seed - set by executor before running
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
    """
    Get seed for layout algorithms.

    Args:
        override: Optional override seed from layout spec

    Returns:
        Seed to use for layout
    """
    if override is not None:
        return override
    base = _current_pipeline_seed or 42
    return derive_seed(base, "layout")


def get_geometry_seed(override: Optional[int] = None) -> int:
    """
    Get seed for geometry generation (initial positions, etc.).

    Args:
        override: Optional override seed

    Returns:
        Seed to use for geometry
    """
    if override is not None:
        return override
    base = _current_pipeline_seed or 42
    return derive_seed(base, "geometry")


def get_sampling_seed(override: Optional[int] = None) -> int:
    """
    Get seed for sampling operations (OSMnx OD sampling, etc.).

    Args:
        override: Optional override seed

    Returns:
        Seed to use for sampling
    """
    if override is not None:
        return override
    base = _current_pipeline_seed or 42
    return derive_seed(base, "sampling")
