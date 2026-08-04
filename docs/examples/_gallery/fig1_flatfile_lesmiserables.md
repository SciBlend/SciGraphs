---
title: "Les Misérables from a GEXF file"
tagline: "Rank normalization on a betweenness distribution that is half zeros"
---

The co-appearance network of *Les Misérables*: 77 characters, 254 edges,
distributed with the specification so the example runs with no network access.

Betweenness here is extreme even by the standards of social networks. **43 of the
77 characters have betweenness exactly zero** — they appear only alongside
someone more central — and the maximum, Valjean, reaches 0.57. Normalized
linearly onto a colormap, 71 of 77 nodes fall in the lowest tenth and only 5 of
10 color bins are used at all.

`color_norm: RANK` replaces each value by its position in the sorted order. The
bins become 8, 8, 7, 8, 7, 8, 8, 7, 8, 8 — uniform by construction. That is what
the figure shows.

The limit is visible too: the 43 tied zeros share one rank, so they share one
color. Ranking fixes the distribution, not the ties.
