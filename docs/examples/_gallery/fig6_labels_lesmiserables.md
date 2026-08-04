---
title: "Labelled characters"
tagline: "Ranking, occlusion and collision, in that order"
---

The same 77-node network with the eighteen highest-betweenness characters
labelled.

Three filters run in sequence, and the run reports each: 77 nodes projected, 77
inside the frame, 59 not hidden behind geometry, 18 kept. `rank_by` decides the
order, `occlusion` removes labels whose node is behind something, and `declutter`
drops any label whose box would overlap one already placed.

`max_count` is preferable to a threshold. The centrality attributes are
normalized to [0,1], so a cutoff calibrated on one graph means nothing on the
next, whereas a count is bounded whatever the size or distribution.
