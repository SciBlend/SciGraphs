---
title: "Xàtiva in EEVEE"
tagline: "The same specification on the real-time engine"
---

The drivable street network of Xàtiva (Valencia, about 29,000 inhabitants).
Everything above the `render` block is engine-agnostic: the same dataset,
analysis, color and glyph fields produce the same geometry, and only the last
section differs from a Cycles specification.

Nodes are colored by closeness centrality under `QUANTILE` normalization, which
ranks unique values rather than samples — appropriate here because a grid-like
street plan produces large groups of nodes with identical closeness.

`ambient_occlusion` darkens the crevices between clustered nodes, which is what
lets a dense scatter read as depth rather than as a flat field.
