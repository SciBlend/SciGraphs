---
title: "A linear-programming matrix from SuiteSparse"
tagline: "A rectangular constraint matrix as a bipartite graph, laid out the way the collection lays out its own"
---

`Meszaros/nemsafm` is a linear program in standard form: 334 constraints by 2348
variables, 2826 nonzeros, 0.36 % dense. Read as a bipartite graph — one node per
row, one per column, an edge per nonzero — that is 2682 nodes and 2826 edges.

The structure the drawing shows is in the statistics the run exports. Clustering
coefficient is exactly 0, as it must be for a bipartite graph: no triangles.
Median degree is 1 against a mean of 2.11, so most nodes are leaves; assortativity
is −0.79, so those leaves hang off hubs. Together they produce the radial tufts.
Diameter 22 over 2682 nodes gives the elongated shape, and the mean column degree
of 1.20 says most variables appear in a single constraint.

The layout is `YIFAN_HU`, the multilevel force-directed method SuiteSparse itself
uses for its gallery, so the drawing is comparable with the collection's own. It
places the graph in a plane with only slight relief, so `camera_direction` looks
straight down that plane's normal; the default oblique view would foreshorten it.

The matrix ships no coordinates, so a layout is required — unlike a structural
mesh, where the auxiliary coordinate file *is* the embedding and a layout would
overwrite it.
