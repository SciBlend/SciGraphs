"""Hierarchical and directed layout algorithms."""

from .common import *

_MAX_CUT_PASSES = 8

_DUMMY_BUDGET = 200000
_DUMMY_PRIORITY = 1 << 30
_SLACK_PASSES = 4
_PRIORITY_NODE_BUDGET = 200000

def _bfs_levels(G, start, levels, order):
    """One BFS from *start*, writing depths into *levels* and the visit order
    into *order*. An index head, not ``queue.pop(0)``: that pop is O(n) and
    makes a breadth-first sweep quadratic."""
    levels[start] = 0
    parent = {start: None}
    head = len(order)
    order.append(start)
    while head < len(order):
        node = order[head]
        head += 1
        depth = levels[node] + 1
        for neighbor in G.neighbors(node):
            if neighbor not in levels:
                levels[neighbor] = depth
                parent[neighbor] = node
                order.append(neighbor)
    return parent

def _component_roots(G):
    """One root per connected component, each an approximate center found by the
    double BFS sweep: walk to a farthest node, again to a farthest node from
    there, and take the midpoint of that path. Exact on trees, where the
    maximum-degree node is usually not the root."""
    roots = []
    seen = set()
    for start in G.nodes():
        if start in seen:
            continue
        first = {}
        order = []
        _bfs_levels(G, start, first, order)
        seen.update(order)
        second = {}
        far_order = []
        parent = _bfs_levels(G, order[-1], second, far_order)
        path = [far_order[-1]]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        roots.append(path[len(path) // 2])
    return roots

def _multi_source_levels(G, roots):
    """BFS depth from all *roots* at once. A node no root reaches seeds a level
    zero of its own, so every node ends up with a level and none needs a random
    position. On a directed graph ``neighbors`` means successors."""
    levels = {}
    queue = []
    for root in roots:
        if root in G and root not in levels:
            levels[root] = 0
            queue.append(root)
    head = 0
    pending = iter(G.nodes())
    while True:
        while head < len(queue):
            node = queue[head]
            head += 1
            depth = levels[node] + 1
            for neighbor in G.neighbors(node):
                if neighbor not in levels:
                    levels[neighbor] = depth
                    queue.append(neighbor)
        for node in pending:
            if node not in levels:
                levels[node] = 0
                queue.append(node)
                break
        else:
            return levels

def _group_by_level(levels):
    """``{level: [node, ...]}`` in BFS order, which keeps the result stable."""
    grouped = {}
    for node, level in levels.items():
        grouped.setdefault(level, []).append(node)
    return grouped

def _disk_positions(count, radius):
    """*count* points over a disk of *radius*, on evenly spaced concentric rings
    filled in order, so nodes adjacent in the input stay adjacent in the
    drawing. Ring k holds a share proportional to its circumference, which puts
    the ring count at sqrt(count / pi) and the spacing at radius / rings."""
    if count == 1:
        return [(0.0, 0.0)]
    rings = max(1, int(round(np.sqrt(count / np.pi))))
    share = np.arange(rings) + 0.5
    take = np.round(share / share.sum() * count).astype(int)
    while take.sum() > count:
        take[take.argmax()] -= 1
    while take.sum() < count:
        take[-1] += 1
    points = []
    for k, n in enumerate(take):
        if n <= 0:
            continue
        r = radius * (k + 0.5) / rings
        angle = np.arange(n) * (2 * np.pi / n)
        points.extend(zip(r * np.cos(angle), r * np.sin(angle)))
    return points

def _hierarchical_layout_3d(G, scale):
    """Tree-like layout: BFS depth sets Y, each level fills a disk in XZ whose
    radius grows with the level's node count, so density is the same on every
    level. Undirected input roots at each component's center."""
    import time
    start = time.time()
    print(f"Computing Hierarchical 3D layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    if num_nodes == 0:
        return np.zeros((0, 3))

    # Roots are the sources of a directed graph; undirected, the hub stands in.
    if G.is_directed():
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    else:
        roots = _component_roots(G)

    levels = _multi_source_levels(G, roots)
    grouped = _group_by_level(levels)

    max_level = max(grouped) if grouped else 0
    widest = max(len(nodes) for nodes in grouped.values()) if grouped else 1
    positions = np.zeros((num_nodes, 3))

    for level, nodes_at_level in grouped.items():
        z = (level / max(1, max_level)) * scale * 2 - scale
        count = len(nodes_at_level)
        radius = scale * 0.5 * np.sqrt(count / widest)
        for node, (x, y) in zip(nodes_at_level, _disk_positions(count, radius)):
            positions[node] = [x, y, z]

    print(f"  Hierarchical 3D completed in {time.time() - start:.2f}s")
    return positions

def _bipartite_parts(G):
    """Two-color every connected component, or None when *G* is not bipartite.
    ``nx.bipartite.sets`` raises AmbiguousSolution on a disconnected graph, and
    one isolated node is enough to trigger it; coloring per component does not.
    Which side of a component goes on which plane is arbitrary, so pick the
    orientation that keeps the two planes even."""
    color = {}
    set0 = []
    set1 = []
    for start in G.nodes():
        if start in color:
            continue
        part = [[start], []]
        color[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            other = 1 - color[node]
            for neighbor in G.neighbors(node):
                if neighbor not in color:
                    color[neighbor] = other
                    part[other].append(neighbor)
                    queue.append(neighbor)
                elif color[neighbor] != other:
                    return None
        skew = len(set0) - len(set1)
        if abs(skew + len(part[0]) - len(part[1])) > abs(skew + len(part[1]) - len(part[0])):
            part.reverse()
        set0.extend(part[0])
        set1.extend(part[1])
    return set0, set1

def _greedy_max_cut(G):
    """Greedy 1/2-approximate maximum cut, then single-vertex local search. At
    least half the edges are guaranteed to run between the two planes. A split
    at the median degree guarantees nothing: on a regular graph every degree
    equals the median, so every node lands on one plane and no edge crosses."""
    side = {}
    for node in G.nodes():
        placed = [0, 0]
        for neighbor in G.neighbors(node):
            if neighbor != node and neighbor in side:
                placed[side[neighbor]] += 1
        side[node] = 0 if placed[0] <= placed[1] else 1

    for _ in range(_MAX_CUT_PASSES):
        moved = 0
        for node in G.nodes():
            around = [0, 0]
            for neighbor in G.neighbors(node):
                if neighbor != node:
                    around[side[neighbor]] += 1
            if around[side[node]] > around[1 - side[node]]:
                side[node] = 1 - side[node]
                moved += 1
        if not moved:
            break

    set0 = [n for n, s in side.items() if s == 0]
    set1 = [n for n, s in side.items() if s == 1]
    return set0, set1

def _bipartite_layout_3d(G, scale):
    """Two node sets on parallel planes, one ring each. Bipartite input is
    two-colored per component; anything else falls back to a greedy maximum
    cut and says so, rather than presenting a meaningless split as a drawing."""
    import time
    start = time.time()
    print(f"Computing Bipartite 3D layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    if num_nodes == 0:
        return np.zeros((0, 3))

    parts = _bipartite_parts(G)
    if parts is None:
        print("  Not bipartite: splitting with a greedy maximum cut instead.")
        parts = _greedy_max_cut(G)
    set0, set1 = parts

    positions = np.zeros((num_nodes, 3))

    for nodes, z in ((set0, -scale * 0.5), (set1, scale * 0.5)):
        count = len(nodes)
        for i, node in enumerate(nodes):
            angle = (i / max(1, count)) * 2 * np.pi
            radius = scale * 0.6
            positions[node] = [radius * np.cos(angle), radius * np.sin(angle), z]

    print(f"  Bipartite 3D completed in {time.time() - start:.2f}s")
    return positions

def _greedy_fas_order(G):
    """Vertex sequence whose backward arcs are a small feedback arc set: the GR
    heuristic of Eades, Lin & Smyth, "A fast and effective heuristic for the
    feedback arc set problem" (1993). Peel sinks to the right, sources to the
    left, otherwise the vertex of largest out-degree minus in-degree."""
    import heapq

    nodes = list(G.nodes())
    succ = {n: [v for v in G.successors(n) if v != n] for n in nodes}
    pred = {n: [u for u in G.predecessors(n) if u != n] for n in nodes}
    out_degree = {n: len(succ[n]) for n in nodes}
    in_degree = {n: len(pred[n]) for n in nodes}
    rank = {n: i for i, n in enumerate(nodes)}
    alive = set(nodes)
    heap = [(in_degree[n] - out_degree[n], rank[n], n) for n in nodes]
    heapq.heapify(heap)
    ready = [n for n in nodes if not out_degree[n] or not in_degree[n]]
    left = []
    right = []

    def peel(node):
        alive.discard(node)
        for v in succ[node]:
            if v in alive:
                in_degree[v] -= 1
                if not in_degree[v]:
                    ready.append(v)
                heapq.heappush(heap, (in_degree[v] - out_degree[v], rank[v], v))
        for u in pred[node]:
            if u in alive:
                out_degree[u] -= 1
                if not out_degree[u]:
                    ready.append(u)
                heapq.heappush(heap, (in_degree[u] - out_degree[u], rank[u], u))

    while alive:
        while ready:
            node = ready.pop()
            if node not in alive:
                continue
            (right if not out_degree[node] else left).append(node)
            peel(node)
        if not alive:
            break
        while heap:
            key, _, node = heapq.heappop(heap)
            if node in alive and key == in_degree[node] - out_degree[node]:
                left.append(node)
                peel(node)
                break

    right.reverse()
    return left + right

def _acyclic_arcs(G):
    """Every non-loop edge oriented forward along a vertex sequence, so the
    result is a DAG that still holds all of them. A directed graph gets the
    greedy feedback-arc-set sequence and its feedback arcs come back reversed,
    not deleted; an undirected graph is oriented by node order, which is acyclic
    by construction and leaves the edge count untouched."""
    order = _greedy_fas_order(G) if G.is_directed() else list(G.nodes())
    rank = {n: i for i, n in enumerate(order)}
    arcs = set()
    for u, v in G.edges():
        if u == v:
            continue
        arcs.add((u, v) if rank[u] < rank[v] else (v, u))
    return sorted(arcs, key=lambda a: (rank[a[0]], rank[a[1]]))

def _longest_path_layers(nodes, arcs):
    """Rank by longest path from a source, over a Kahn topological order."""
    succ = {n: [] for n in nodes}
    in_degree = {n: 0 for n in nodes}
    for u, v in arcs:
        succ[u].append(v)
        in_degree[v] += 1

    layer = {n: 0 for n in nodes}
    queue = [n for n in nodes if not in_degree[n]]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        below = layer[u] + 1
        for v in succ[u]:
            if layer[v] < below:
                layer[v] = below
            in_degree[v] -= 1
            if not in_degree[v]:
                queue.append(v)
    return layer

def _reduce_slack(layer, nodes, arcs):
    """Shorten edges by sliding each node to the median of its neighbors' ranks
    inside the window its predecessors and successors leave free. Coordinate
    descent on the objective the network simplex of Gansner et al., "A Technique
    for Drawing Directed Graphs" (1993) solves exactly; this only approximates
    it, but it is what keeps the dummy vertex count down."""
    preds = {n: [] for n in nodes}
    succs = {n: [] for n in nodes}
    for u, v in arcs:
        succs[u].append(v)
        preds[v].append(u)

    for _ in range(_SLACK_PASSES):
        moved = 0
        for node in nodes:
            above, below = preds[node], succs[node]
            if not below:
                continue
            low = (max(layer[u] for u in above) + 1) if above else 0
            high = min(layer[v] for v in below) - 1
            if high <= low:
                continue
            around = sorted([layer[u] for u in above] + [layer[v] for v in below])
            target = min(high, max(low, around[len(around) // 2]))
            if target != layer[node]:
                layer[node] = target
                moved += 1
        if not moved:
            break

    used = sorted(set(layer.values()))
    remap = {r: i for i, r in enumerate(used)}
    return {n: remap[r] for n, r in layer.items()}

def _build_ordering_graph(nodes, arcs, layer):
    """Adjacency for the crossing-reduction phase: a chain of dummy vertices
    stands in for every arc spanning more than one layer, so a long edge is
    counted and routed like the sequence of short ones it is drawn as. Returns
    (layer_of, up, down, num_dummies, straight), where *straight* counts the
    arcs left out because the dummy budget ran out."""
    index = {n: i for i, n in enumerate(nodes)}
    layer_of = [layer[n] for n in nodes]
    up = [[] for _ in nodes]
    down = [[] for _ in nodes]

    spans = [layer[v] - layer[u] for u, v in arcs]
    needed = sum(s - 1 for s in spans)
    max_span = None
    if needed > _DUMMY_BUDGET:
        budget = 0
        for span in sorted(spans):
            budget += span - 1
            if budget > _DUMMY_BUDGET:
                max_span = max(1, span - 1)
                break

    straight = 0
    for (u, v), span in zip(arcs, spans):
        tail, head = index[u], index[v]
        if span == 1:
            down[tail].append(head)
            up[head].append(tail)
            continue
        if max_span is not None and span > max_span:
            straight += 1
            continue
        previous = tail
        for step in range(1, span):
            dummy = len(layer_of)
            layer_of.append(layer[u] + step)
            up.append([])
            down.append([])
            down[previous].append(dummy)
            up[dummy].append(previous)
            previous = dummy
        down[previous].append(head)
        up[head].append(previous)

    return layer_of, up, down, len(layer_of) - len(nodes), straight

def _init_order(num_layers, layer_of, up, down):
    """Seed each layer's order with a breadth-first walk so that vertices joined
    by an edge start out near each other (Gansner et al. 1993, init_order)."""
    order = [[] for _ in range(num_layers)]
    seen = set()
    queue = []
    head = 0
    for start in range(len(layer_of)):
        if start in seen:
            continue
        seen.add(start)
        order[layer_of[start]].append(start)
        queue.append(start)
        while head < len(queue):
            v = queue[head]
            head += 1
            for w in down[v]:
                if w not in seen:
                    seen.add(w)
                    order[layer_of[w]].append(w)
                    queue.append(w)
            for w in up[v]:
                if w not in seen:
                    seen.add(w)
                    order[layer_of[w]].append(w)
                    queue.append(w)
    return order

def _median_value(neighbors, position):
    """dot's weighted median: -1 with no neighbors, so the vertex stays put."""
    if not neighbors:
        return -1.0
    p = sorted(position[n] for n in neighbors)
    middle = len(p) // 2
    if len(p) % 2:
        return float(p[middle])
    if len(p) == 2:
        return (p[0] + p[1]) / 2.0
    left = p[middle - 1] - p[0]
    right = p[-1] - p[middle]
    if left + right == 0:
        return (p[middle - 1] + p[middle]) / 2.0
    return (p[middle - 1] * right + p[middle] * left) / (left + right)

def _sort_by_median(row, adjacency, position):
    """Reorder *row* by median, leaving neighborless vertices in their slots."""
    movable = [(i, v) for i, v in enumerate(row) if adjacency[v]]
    if len(movable) < 2:
        return
    keyed = sorted(movable, key=lambda iv: (_median_value(adjacency[iv[1]], position), iv[0]))
    for (slot, _), (_, v) in zip(movable, keyed):
        row[slot] = v
    for i, v in enumerate(row):
        position[v] = i

def _pair_crossings(v, w, adjacency, position):
    """Crossings between v's and w's edges when v is drawn left of w."""
    a = sorted(position[x] for x in adjacency[v])
    b = sorted(position[x] for x in adjacency[w])
    if not a or not b:
        return 0
    count = 0
    j = 0
    for x in a:
        while j < len(b) and b[j] < x:
            j += 1
        count += j
    return count

def _transpose(order, up, down, position, rounds):
    """Swap adjacent pairs while that removes crossings (Gansner et al. 1993)."""
    for _ in range(rounds):
        improved = False
        for row in order:
            for i in range(len(row) - 1):
                v, w = row[i], row[i + 1]
                before = (_pair_crossings(v, w, up, position)
                          + _pair_crossings(v, w, down, position))
                if not before:
                    continue
                after = (_pair_crossings(w, v, up, position)
                         + _pair_crossings(w, v, down, position))
                if after < before:
                    improved = True
                    row[i], row[i + 1] = w, v
                    position[v] = i + 1
                    position[w] = i
        if not improved:
            break

def _bilayer_crossings(pairs, width):
    """Crossings between two adjacent layers, with the accumulator tree of
    Barth, Junger & Mutzel, "Simple and Efficient Bilayer Cross Counting"
    (2002). *pairs* is sorted by (upper index, lower index)."""
    if width < 2 or not pairs:
        return 0
    first = 1
    while first < width:
        first *= 2
    tree = [0] * (2 * first - 1)
    first -= 1
    count = 0
    for _, lower in pairs:
        node = lower + first
        tree[node] += 1
        while node > 0:
            if node % 2:
                count += tree[node + 1]
            node = (node - 1) // 2
            tree[node] += 1
    return count

def _total_crossings(order, down, position):
    total = 0
    for level, row in enumerate(order[:-1]):
        pairs = []
        for i, v in enumerate(row):
            for w in down[v]:
                pairs.append((i, position[w]))
        pairs.sort()
        total += _bilayer_crossings(pairs, len(order[level + 1]))
    return total

def _order_layers(order, up, down, position, iterations, transpose_rounds):
    """Alternating median sweeps with a transpose step, keeping the best order
    seen. Phase three of Sugiyama, Tagawa & Toda (1981) as dot implements it."""
    best = [list(row) for row in order]
    best_count = _total_crossings(order, down, position)
    for sweep in range(iterations):
        if sweep % 2:
            for level in range(len(order) - 2, -1, -1):
                _sort_by_median(order[level], down, position)
        else:
            for level in range(1, len(order)):
                _sort_by_median(order[level], up, position)
        if transpose_rounds:
            _transpose(order, up, down, position, transpose_rounds)
        count = _total_crossings(order, down, position)
        if count < best_count:
            best_count = count
            best = [list(row) for row in order]
        if not best_count:
            break
    for level, row in enumerate(best):
        order[level] = row
        for i, v in enumerate(row):
            position[v] = i
    return best_count

def _blockers(priority):
    """Nearest index on each side holding a strictly higher priority, or None.
    Vertices keep a gap of at least one, so the nearest blocker is always the
    binding one and a linear scan per vertex would only repeat this."""
    left = [None] * len(priority)
    right = [None] * len(priority)
    stack = []
    for i in range(len(priority) - 1, -1, -1):
        while stack and priority[stack[-1]] <= priority[i]:
            stack.pop()
        right[i] = stack[-1] if stack else None
        stack.append(i)
    stack = []
    for i in range(len(priority)):
        while stack and priority[stack[-1]] <= priority[i]:
            stack.pop()
        left[i] = stack[-1] if stack else None
        stack.append(i)
    return left, right

def _priority_move(x, i, target, gap, left, right):
    """Slide vertex *i* toward *target*, pushing lower-priority neighbors along
    and stopping at the first higher-priority one: the priority method for
    horizontal coordinates from Sugiyama, Tagawa & Toda (1981)."""
    if target > x[i]:
        j = right[i]
        moved = target if j is None else min(target, x[j] - (j - i) * gap)
        if moved <= x[i]:
            return
        x[i] = moved
        for j in range(i + 1, len(x)):
            if x[j] >= x[j - 1] + gap:
                break
            x[j] = x[j - 1] + gap
    elif target < x[i]:
        j = left[i]
        moved = target if j is None else max(target, x[j] + (i - j) * gap)
        if moved >= x[i]:
            return
        x[i] = moved
        for j in range(i - 1, -1, -1):
            if x[j] <= x[j + 1] - gap:
                break
            x[j] = x[j + 1] - gap

def _assign_x(order, up, down, num_nodes, passes):
    """Horizontal coordinates: start at the layer index, then run priority
    passes against the layer above and below in turn. Brandes & Kopf (2001)
    would straighten long edges better; this keeps the code small and still
    pulls each vertex toward its neighbors without reordering them."""
    x = {}
    for row in order:
        for i, v in enumerate(row):
            x[v] = float(i)

    for step in range(passes):
        downward = step % 2 == 0
        levels = range(1, len(order)) if downward else range(len(order) - 2, -1, -1)
        reference = up if downward else down
        for level in levels:
            row = order[level]
            xs = [x[v] for v in row]
            priority = [_DUMMY_PRIORITY if v >= num_nodes else len(reference[v])
                        for v in row]
            targets = [_median_value(reference[v], x) for v in row]
            left, right = _blockers(priority)
            for i in sorted(range(len(row)), key=lambda k: (-priority[k], k)):
                if targets[i] >= 0.0:
                    _priority_move(xs, i, targets[i], 1.0, left, right)
            for v, value in zip(row, xs):
                x[v] = value
    return x

def _sugiyama_layout(G, scale):
    """Layered drawing in the z = 0 plane, after Sugiyama, Tagawa & Toda (1981).

    Cycles go through the greedy heuristic of Eades, Lin & Smyth (1993), which
    reverses arcs instead of dropping them, so no edge is lost. Ranks come from
    longest-path layering with a slack-reduction pass, and an arc spanning more
    than one rank is routed over dummy vertices. Within a rank the order is the
    median heuristic plus adjacent transposition of Gansner et al. (1993), and
    X comes from the priority method of the original paper.
    """
    import time
    start = time.time()
    print(f"Computing Sugiyama layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    if num_nodes == 0:
        return np.zeros((0, 3))

    nodes = list(G.nodes())
    arcs = _acyclic_arcs(G)
    layer = _reduce_slack(_longest_path_layers(nodes, arcs), nodes, arcs)
    layer_of, up, down, num_dummies, straight = _build_ordering_graph(nodes, arcs, layer)

    max_layer = max(layer_of) if layer_of else 0
    order = _init_order(max_layer + 1, layer_of, up, down)
    position = {}
    for row in order:
        for i, v in enumerate(row):
            position[v] = i

    total = len(layer_of)
    iterations = 8 if total <= 50000 else 4
    transpose_rounds = 4 if total <= 5000 else (2 if total <= 30000 else
                                                (1 if total <= 150000 else 0))
    crossings = _order_layers(order, up, down, position, iterations, transpose_rounds)

    widest = max(len(row) for row in order) if order else 1
    x = _assign_x(order, up, down, num_nodes,
                  4 if len(layer_of) <= _PRIORITY_NODE_BUDGET else 0)

    lo = min(x.values()) if x else 0.0
    hi = max(x.values()) if x else 0.0
    width = (hi - lo) or 1.0
    positions = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        y = ((layer_of[i] / max_layer) * 2 - 1) * scale if max_layer else 0.0
        positions[i] = [((x[i] - lo) / width * 2 - 1) * scale, y, 0.0]

    print(f"  Sugiyama: {len(arcs)} arcs over {max_layer + 1} layers, widest "
          f"{widest}, {num_dummies} dummies, {crossings} crossings"
          + (f", {straight} long arcs unrouted" if straight else ""))
    print(f"  Sugiyama layout completed in {time.time() - start:.2f}s")
    return positions

def _circular_hierarchy_layout(G, scale):
    """Roots at the center, each further level on a wider concentric ring. A
    lone root sits on the axis; several share a small ring inside level one."""
    import time
    start = time.time()
    print(f"Computing Circular Hierarchy layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    if num_nodes == 0:
        return np.zeros((0, 3))

    if G.is_directed():
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not roots:
            # Every node has a parent, so take the biggest fan-outs as roots.
            out_degrees = dict(G.out_degree())
            max_out = max(out_degrees.values()) if out_degrees else 0
            roots = [n for n, d in out_degrees.items() if d == max_out][:3]
    else:
        roots = _component_roots(G)

    levels = _multi_source_levels(G, roots)
    grouped = _group_by_level(levels)

    max_level = max(grouped) if grouped else 0
    positions = np.zeros((num_nodes, 3))

    for level, nodes_at_level in grouped.items():
        count = len(nodes_at_level)
        if level == 0 and count == 1:
            radius = 0.0
        else:
            radius = max(level, 0.35) * scale / max(2, max_level)
        for i, node in enumerate(nodes_at_level):
            angle = (i / count) * 2 * np.pi
            positions[node] = [radius * np.cos(angle), radius * np.sin(angle), 0.0]

    print(f"  Circular Hierarchy layout completed in {time.time() - start:.2f}s")
    return positions

__all__ = [name for name in globals() if not name.startswith('__')]
