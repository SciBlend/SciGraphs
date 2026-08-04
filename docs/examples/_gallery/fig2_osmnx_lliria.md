---
title: "Llíria street network, orthographic"
tagline: "Choosing a color transform by measuring what it does to the histogram"
---

The walkable street network of Llíria (Valencia, about 23,000 inhabitants):
6037 nodes, 8067 edges, downloaded live from OpenStreetMap. There is no `layout`
block on purpose — the coordinates arrive with the data, and a force-directed
layout would discard the geography.

`camera_ortho` removes the near/far size falloff, so a node radius that encodes a
value stays comparable anywhere in the frame. The cost is the loss of depth cues,
a fair trade for a network that is essentially flat.

Betweenness here is extremely concentrated: median 0.001 against a maximum of
0.26, with 1112 nodes at exactly zero. Choosing how to map that onto a colormap
is the interesting decision, and it is one you can measure rather than guess.
Counting how many nodes each transform sends into the brightest three tenths of
the ramp:

| `color_norm` | Ramp bins used | Nodes in the brightest 30 % |
| --- | --- | --- |
| `LINEAR` | 9 of 10 | 8 (0.1 %) |
| `LINEAR`, clipped at the 99th percentile | 10 of 10 | 159 (2.6 %) |
| `RANK` | 10 of 10 | 1811 (30.0 %) |
| `LOG` | 10 of 10 | 2419 (40.1 %) |

`RANK` and `LOG` both flood the image: they equalize the histogram, which is
exactly right for a graph whose ordering you want to read, and exactly wrong for
one where the interesting fact is that a small minority of streets carry the
through traffic. Plain `LINEAR` goes too far the other way — a single extreme
node compresses everything else into one bin.

Clipping the top percentile and then mapping linearly keeps the concentration
while spending the whole ramp: the arterial routes come out as continuous bright
chains against a violet field, and their brightness still means magnitude rather
than rank.

**This example needs the Overpass API.** When it is down — which happens — the
dataset stage retries and appears to hang. Nominatim answering does not mean
Overpass will.
