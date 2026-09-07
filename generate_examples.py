"""Connect-the-dots stimuli graded by turtle-program length and structure.

Every figure is a scene of unit line segments whose endpoints sit on an
integer lattice. Complexity is the length of a short turtle program in the
Structure Validation language (Draw line, Forward, Turn, Repeat), not
DreamCoder nats. Structure is the track-wise "plain" score: larger means
more irregular / less object-like.

Higher levels reorient lines about a shared vertex. They stay one
drawing (or two multi-segment pieces). Consecutive levels must
strictly increase both turtle length and structure, keep the same line
count, and look different in orientation.

--objectSets writes one Level 3 per object set. That figure is shared
across Closed / Target / Broken basic-structure folders. Sets 13 and
14 are one connected drawing; Sets 11, 12, and 15 may be two
multi-segment groups. It is built from a *different* shape family
than that set's Level 2 (across-shapes), with the same line count.
Turtle length and structure must exceed every Level 2 in the set,
including Broken, by as much as possible.
"""

import argparse
import math
import random
import sys
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

import matplotlib.pyplot as plt


lineColor = 'black'
lineWidth = 5.5
gridColor = '0.82'
levelFracs = [0.0, 0.3, 0.6]   # fraction of lines pivoted at each level (within category)
acrossFracs = [0.0, 0.2, 0.4]  # gentler schedule used by --across
levelBias = [0.9, 0.9, 0.85]  # stay attached; irregularity comes from reoriented sides
cellsPerLine = 5          # the universal line unit: every line connects 6 dots (5 cells)
gridCoarse = 5            # coarse grid: gridCoarse x gridCoarse bands per figure
gridN = gridCoarse * cellsPerLine   # fine grid drawn under every figure
minDiff = 0.1             # consecutive levels must differ this much in line orientations
connectBias = 0.9         # chance a pivoted line prefers landing on an existing dot
maxSpan = gridCoarse - 1  # hard cap on any figure's extent, in coarse bands

# the eight lattice directions a pivoted line may take
latticeDirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def path(steps):
    # a vertex path from unit steps on the coarse lattice
    pts = [(0, 0)]
    for dx, dy in steps:
        pts.append((pts[-1][0] + dx, pts[-1][1] + dy))
    return pts

def repeat(*dirs, u=1):
    return [d for d in dirs for _ in range(u)]

R, U, Lf, D = (1, 0), (0, 1), (-1, 0), (0, -1)
NE, NW, SW, SE = (1, 1), (-1, 1), (-1, -1), (1, -1)

def latticeShapes(u):
    # every shape is a path of unit lines in the four allowed orientations;
    # a side of u steps is a chain of u standard lines with dots at joints
    return [
        ('square', repeat(R, u=u) + repeat(U, u=u) + repeat(Lf, u=u) + repeat(D, u=u)),
        ('rectangle x2', repeat(R, u=2 * u) + repeat(U, u=u) + repeat(Lf, u=2 * u) + repeat(D, u=u)),
        ('diamond', repeat(NE, u=u) + repeat(NW, u=u) + repeat(SW, u=u) + repeat(SE, u=u)),
        ('octagon', repeat(R, u=u) + repeat(NE, u=u) + repeat(U, u=u) + repeat(NW, u=u)
                    + repeat(Lf, u=u) + repeat(SW, u=u) + repeat(D, u=u) + repeat(SE, u=u)),
        ('hexagon', repeat(R, u=u) + repeat(NE, u=u) + repeat(NW, u=u)
                    + repeat(Lf, u=u) + repeat(SW, u=u) + repeat(SE, u=u)),
        ('triangle', repeat(R, u=2 * u) + repeat(NW, u=u) + repeat(SW, u=u)),
        ('trapezium', repeat(NE, u=u) + repeat(R, u=u) + repeat(SE, u=u) + repeat(Lf, u=3 * u)),
        ('parallelogram', repeat(R, u=2 * u) + repeat(NE, u=u) + repeat(Lf, u=2 * u) + repeat(SW, u=u)),
        ('L-shape', repeat(R, u=2 * u) + repeat(U, u=u) + repeat(Lf, u=u)
                    + repeat(U, u=u) + repeat(Lf, u=u) + repeat(D, u=2 * u)),
        ('zig-zag', repeat(NE, u=u) + repeat(SE, u=u) + repeat(NE, u=u) + repeat(SE, u=u)),
        ('staircase', repeat(R, u=u) + repeat(U, u=u) + repeat(R, u=u) + repeat(U, u=u)),
        ('hook', repeat(R, u=2 * u) + repeat(U, u=u) + repeat(Lf, u=u)),
        ('arrow', repeat(NE, u=u) + repeat(SE, u=u) + repeat(Lf, u=2 * u)),
    ]

def library():
    # every shape at the largest size that fits the coarse grid
    out = []
    for name, steps in latticeShapes(1):
        for u in (3, 2, 1):
            pts = path([s for s in latticeShapes(u) if s[0] == name][0][1])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if max(max(xs) - min(xs), max(ys) - min(ys)) <= maxSpan:
                out.append((name, u))
                break
    return out

def baselineScene(name, u):
    pts = path([s for s in latticeShapes(u) if s[0] == name][0][1])
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


# Pivoting lines
#
# A pivot keeps one endpoint dot of the line and swings the line onto a new
# lattice direction, spanning the same number of grid cells, so endpoints
# always land on dots and the cell budget never changes.

def cells(seg):
    (a, b) = seg
    return max(abs(b[0] - a[0]), abs(b[1] - a[1]))

def sceneBudget(scene):
    # every line is one universal unit, so the budget is cellsPerLine per line
    return len(scene) * cellsPerLine

def sceneBox(scene, margin=1):
    xs = [p[0] for seg in scene for p in seg]
    ys = [p[1] for seg in scene for p in seg]
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)

def fitsGrid(scene):
    xs = [p[0] for seg in scene for p in seg]
    ys = [p[1] for seg in scene for p in seg]
    return max(max(xs) - min(xs), max(ys) - min(ys)) <= maxSpan

def pivot(seg, rng, box=None, dots=None, bias=None):
    # swing the line about one of its endpoint dots onto a new lattice
    # orientation, one universal unit long; placements near the figure's
    # bounding box are preferred so the grid stays compact, and with
    # probability bias the far end prefers landing on an existing
    # dot, favouring connectedness without requiring it
    if bias is None:
        bias = connectBias
    (a, b) = seg
    v = (b[0] - a[0], b[1] - a[1])
    n = 1                                      # one coarse band
    options = []
    for d in latticeDirs:
        if v[0] * d[1] - v[1] * d[0] == 0:         # exclude the line's own axis
            continue
        for anchor in (a, b):
            options.append((anchor, (anchor[0] + d[0] * n, anchor[1] + d[1] * n)))
    if box:
        (x0, y0, x1, y1) = box
        inside = [o for o in options if x0 <= o[1][0] <= x1 and y0 <= o[1][1] <= y1]
        if inside:
            options = inside
    if dots and rng.random() < bias:
        connecting = [o for o in options if o[1] in dots]
        if connecting:
            options = connecting
    return rng.choice(options)

def straightThrough(p, q, r):
    # p is a pass-through joint: q and r sit collinearly on opposite sides
    if (q[0] - p[0]) * (r[1] - p[1]) != (q[1] - p[1]) * (r[0] - p[0]):
        return False
    return (q[0] - p[0]) * (r[0] - p[0]) + (q[1] - p[1]) * (r[1] - p[1]) < 0

def pivotContinues(scene, idxs):
    # a pivoted line must not continue a touching line in a straight run,
    # which would read as one line twice the universal length
    for i in idxs:
        (a, b) = scene[i]
        for j, (c, d) in enumerate(scene):
            if j == i:
                continue
            for p, q in ((a, b), (b, a)):
                if p == c and straightThrough(p, q, d):
                    return True
                if p == d and straightThrough(p, q, c):
                    return True
    return False

def onInterior(p, seg):
    # p lies strictly inside seg (collinear and between, not an endpoint)
    (a, b) = seg
    if p == a or p == b:
        return False
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if cross != 0:
        return False
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and \
           min(a[1], b[1]) <= p[1] <= max(a[1], b[1])

def validScene(scene):
    # two rules keep dots countable: no two coincident lines, and no dot may
    # sit in the interior of any line, so a dot always marks a line end and
    # each line shows exactly its two end dots
    if len({frozenset(seg) for seg in scene}) != len(scene):
        return False
    dots = {p for seg in scene for p in seg}
    return not any(onInterior(p, seg) for p in dots for seg in scene)

def pivotScene(scene, k, rng, bias=None):
    if k <= 0:
        return list(scene)
    box = sceneBox(scene)
    idxs = rng.sample(range(len(scene)), k)
    out = list(scene)
    for i in idxs:
        dots = {p for j, seg in enumerate(out) if j != i for p in seg}
        out[i] = pivot(scene[i], rng, box, dots, bias=bias)
    if pivotContinues(out, idxs):
        return None
    return out

def changedCounts(levels, nLines, schedule=None):
    # deterministic schedule: level j pivots ceil(schedule[j] * L) lines
    # (the last fraction repeats if needed)
    schedule = schedule or levelFracs
    fracs = [schedule[min(j, len(schedule) - 1)] for j in range(levels)]
    counts = [math.ceil(f * nLines) for f in fracs]
    for j in range(1, levels):
        counts[j] = min(nLines, max(counts[j], counts[j - 1] + 1))
    return counts


# Turtle-program length
#
# Draw line(n) / Forward(n) / Turn(θ) / Repeat(k). Length is command count,
# Repeat = 1 + body. Start pose is free. Collinear unit steps merge.
# Clockwise turns are positive; heading 0 is east, 90 is south (y-up).

DIR_HEADING = {
    (1, 0): 0, (1, -1): 45, (0, -1): 90, (-1, -1): 135,
    (-1, 0): 180, (-1, 1): 225, (0, 1): 270, (1, 1): 315,
}
dirIndex = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
            (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7}

def unitEdges(scene):
    edges = []
    for (a, b) in scene:
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = max(abs(dx), abs(dy))
        if n == 0:
            continue
        sx, sy = dx // n, dy // n
        x, y = a
        for _ in range(n):
            nx, ny = x + sx, y + sy
            edges.append(((x, y), (nx, ny)))
            x, y = nx, ny
    return edges

def edgeCanon(seg):
    a, b = seg
    return (a, b) if a <= b else (b, a)

def turnDelta(h0, h1):
    d = (h1 - h0) % 360
    return d - 360 if d > 180 else d

def programLength(prog):
    total = 0
    for cmd in prog:
        if cmd[0] == 'Repeat':
            total += 1 + programLength(cmd[2])
        else:
            total += 1
    return total

@lru_cache(maxsize=None)
def _compress(cmds):
    # shortest Repeat encoding of a linear command sequence; Repeat is
    # used only when 1 + body is strictly shorter than unrolling
    n = len(cmds)
    if n == 0:
        return ()
    dp_len = [0] * (n + 1)
    dp_prog = [()] * (n + 1)
    for i in range(n - 1, -1, -1):
        best_len = 1 + dp_len[i + 1]
        best_prog = (cmds[i],) + dp_prog[i + 1]
        remaining = n - i
        for p in range(1, remaining // 2 + 1):
            body = cmds[i:i + p]
            body_prog = _compress(body)
            body_len = programLength(body_prog)
            k = 2
            while i + k * p <= n and cmds[i + (k - 1) * p:i + k * p] == body:
                if 1 + body_len < k * p:
                    total = 1 + body_len + dp_len[i + k * p]
                    if total < best_len:
                        best_len = total
                        best_prog = (('Repeat', k, body_prog),) + dp_prog[i + k * p]
                k += 1
        dp_len[i] = best_len
        dp_prog[i] = best_prog
    return dp_prog[0]

def compressProgram(cmds):
    return list(_compress(tuple(cmds)))

def compressedLength(cmds):
    if any(cmd[0] == 'Repeat' for cmd in cmds):
        return programLength(cmds)
    return programLength(compressProgram(cmds))

def formatTurtle(cmds, indent=0):
    pad = '  ' * indent
    lines = []
    for cmd in cmds:
        op = cmd[0]
        if op == 'Repeat':
            lines.append(f"{pad}Repeat({cmd[1]}) {{")
            lines.extend(formatTurtle(cmd[2], indent + 1))
            lines.append(f"{pad}}}")
        elif op == 'Draw':
            lines.append(f"{pad}Draw line({cmd[1]})")
        elif op == 'Forward':
            lines.append(f"{pad}Forward({cmd[1]})")
        else:
            lines.append(f"{pad}Turn({cmd[1]})")
    return lines

def _graph(edges):
    adj = defaultdict(Counter)
    for a, b in edges:
        a, b = edgeCanon((a, b))
        adj[a][b] += 1
        adj[b][a] += 1
    return adj

def _nearestUnused(pos, adj):
    targets = [v for v, nbrs in adj.items() if sum(nbrs.values()) > 0]
    if not targets:
        return None
    if pos in targets:
        return pos, []
    target_set = set(targets)
    q = deque([pos])
    prev = {pos: None}
    found = None
    while q and len(prev) < 500:
        u = q.popleft()
        if u in target_set and u != pos:
            found = u
            break
        x, y = u
        for d in DIR_HEADING:
            v = (x + d[0], y + d[1])
            if v not in prev:
                prev[v] = u
                q.append(v)
    if found is None:
        found = min(targets, key=lambda t: abs(t[0] - pos[0]) + abs(t[1] - pos[1]))
        path = []
        x, y = pos
        while (x, y) != found:
            sx = 0 if found[0] == x else (1 if found[0] > x else -1)
            sy = 0 if found[1] == y else (1 if found[1] > y else -1)
            x, y = x + sx, y + sy
            path.append((x, y))
        return found, path
    path = []
    cur = found
    while prev[cur] is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return found, path

def _greedyTour(edges, start):
    adj = _graph(edges)
    steps = []
    pos = start
    unused = sum(sum(c.values()) for c in adj.values()) // 2
    guard = 0
    while unused > 0 and guard < 500:
        guard += 1
        nbrs = [v for v, k in adj[pos].items() if k > 0]
        if not nbrs:
            nxt = _nearestUnused(pos, adj)
            if nxt is None:
                break
            dest, path = nxt
            if dest == pos:
                break
            cur = pos
            for p in path:
                steps.append((cur[0], cur[1], p[0], p[1], 'F'))
                cur = p
            pos = dest
            continue
        prefer = None
        if steps:
            lx0, ly0, lx1, ly1 = steps[-1][:4]
            hd = (lx1 - lx0, ly1 - ly0)
            cand = (pos[0] + hd[0], pos[1] + hd[1])
            if cand in nbrs:
                prefer = cand
        nxt = prefer if prefer is not None else nbrs[0]
        adj[pos][nxt] -= 1
        adj[nxt][pos] -= 1
        unused -= 1
        steps.append((pos[0], pos[1], nxt[0], nxt[1], 'D'))
        pos = nxt
    return steps

def _stepsToCmds(steps):
    if not steps:
        return []
    x0, y0, x1, y1, kind = steps[0]
    heading = DIR_HEADING[(x1 - x0, y1 - y0)]
    cmds = []
    i = 0
    while i < len(steps):
        x0, y0, x1, y1, kind = steps[i]
        d = (x1 - x0, y1 - y0)
        h = DIR_HEADING[d]
        delta = turnDelta(heading, h)
        if delta and i > 0:
            cmds.append(('Turn', delta))
            heading = h
        elif i == 0:
            heading = h
        n = 1
        x, y = x1, y1
        while i + n < len(steps):
            a, b, c, e, k2 = steps[i + n]
            if k2 != kind or (a, b) != (x, y):
                break
            dd = (c - a, e - b)
            if DIR_HEADING.get(dd) != heading:
                break
            n += 1
            x, y = c, e
        cmds.append(('Forward' if kind == 'F' else 'Draw', n))
        i += n
    drawn = [s for s in steps if s[4] == 'D']
    extra = []
    if (drawn and (drawn[-1][2], drawn[-1][3]) == (drawn[0][0], drawn[0][1])
            and all(s[4] == 'D' for s in steps)):
        h0 = DIR_HEADING[(drawn[0][2] - drawn[0][0], drawn[0][3] - drawn[0][1])]
        delta = turnDelta(heading, h0)
        if delta:
            extra.append(('Turn', delta))
    c0, c1 = cmds, cmds + extra
    return c0 if compressedLength(c0) <= compressedLength(c1) else c1

def _reverseSteps(steps):
    return [(x1, y1, x0, y0, kind) for x0, y0, x1, y1, kind in reversed(steps)]

def _commandVariants(cmds):
    # start pose is free, so drop leading turns; closed tours may be rotated
    seen = set()
    variants = []

    def add(seq):
        key = tuple(seq)
        if key and key not in seen:
            seen.add(key)
            variants.append(list(seq))

    def add_with_stripped(seq):
        add(seq)
        stripped = list(seq)
        while stripped and stripped[0][0] == 'Turn':
            stripped.pop(0)
        add(stripped)

    add_with_stripped(cmds)
    if (cmds and cmds[-1][0] == 'Turn'
            and all(cmd[0] != 'Forward' for cmd in cmds)):
        for i in range(1, len(cmds)):
            add_with_stripped(cmds[i:] + cmds[:i])
    return variants

def turtleProgram(scene):
    edges = [edgeCanon(e) for e in unitEdges(scene)]
    if not edges:
        return [], 0
    deg = Counter()
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    odds = [p for p, d in deg.items() if d % 2]
    starts = odds if odds else list(deg)
    best_prog, best_len = None, 10 ** 9
    for start in starts:
        steps = _greedyTour(edges, start)
        for tour in (steps, _reverseSteps(steps)):
            cmds = _stepsToCmds(tour)
            for variant in _commandVariants(cmds):
                prog = compressProgram(variant)
                ln = programLength(prog)
                if ln < best_len:
                    best_len = ln
                    best_prog = prog
    return list(best_prog or []), best_len

def turtleLength(scene):
    return turtleProgram(scene)[1]


def displaceScene(scene, k, rng, grid_max=None):
    # translate k lines to empty lattice slots, keeping length and orientation
    if k <= 0:
        return list(scene)
    if grid_max is None:
        grid_max = maxSpan
    idxs = rng.sample(range(len(scene)), k)
    remaining = [seg for i, seg in enumerate(scene) if i not in idxs]
    occupied = {p for seg in remaining for p in seg}
    used = {edgeCanon(seg) for seg in remaining}
    out = list(remaining)
    for i in idxs:
        (a, b) = scene[i]
        dx, dy = b[0] - a[0], b[1] - a[1]
        cands = []
        for x in range(0, grid_max + 1):
            for y in range(0, grid_max + 1):
                na, nb = (x, y), (x + dx, y + dy)
                if not (0 <= nb[0] <= grid_max and 0 <= nb[1] <= grid_max):
                    continue
                if edgeCanon((na, nb)) in used:
                    continue
                if na in occupied or nb in occupied:
                    continue
                cands.append((na, nb))
        if not cands:
            return None
        pick = rng.choice(cands)
        out.append(pick)
        used.add(edgeCanon(pick))
        occupied.update(pick)
    if not validScene(out):
        return None
    return out


def mutateScene(scene, k, rng, bias=None):
    # Higher levels reorient lines about a shared vertex. They stay one
    # drawing; they do not become a spray of disconnected sticks.
    return pivotScene(scene, k, rng, bias=bias)


def sceneComponents(scene):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for a, b in scene:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i, (a, b) in enumerate(scene):
        groups[find(a)].append(i)
    return list(groups.values())


def isConnectedDrawing(scene):
    # one piece, like Previous-stimuli Level 2s: a warped object, not confetti
    return bool(scene) and len(sceneComponents(scene)) == 1


def segmentsCross(scene):
    segs = list(scene)
    for i in range(len(segs)):
        (x1, y1), (x2, y2) = segs[i]
        for j in range(i + 1, len(segs)):
            (x3, y3), (x4, y4) = segs[j]
            if len({(x1, y1), (x2, y2), (x3, y3), (x4, y4)}) < 4:
                continue
            def orient(ax, ay, bx, by, cx, cy):
                return (by - ay) * (cx - bx) - (bx - ax) * (cy - by)
            o1 = orient(x1, y1, x2, y2, x3, y3)
            o2 = orient(x1, y1, x2, y2, x4, y4)
            o3 = orient(x3, y3, x4, y4, x1, y1)
            o4 = orient(x3, y3, x4, y4, x2, y2)
            if o1 * o2 < 0 and o3 * o4 < 0:
                return True
    return False


def mergeSides(scene):
    # maximal collinear runs of unit segments, so a rectangle side swings as one piece
    unused = {edgeCanon(e) for e in unitEdges(scene)}
    used = set()
    sides = []
    for a, b in list(unused):
        e = edgeCanon((a, b))
        if e in used:
            continue
        d = (b[0] - a[0], b[1] - a[1])
        start, end = a, b
        while True:
            prev = (start[0] - d[0], start[1] - d[1])
            ee = edgeCanon((prev, start))
            if ee not in unused or ee in used:
                break
            used.add(ee)
            start = prev
        used.add(e)
        while True:
            nxt = (end[0] + d[0], end[1] + d[1])
            ee = edgeCanon((end, nxt))
            if ee not in unused or ee in used:
                break
            used.add(ee)
            end = nxt
        sides.append((start, end))
    return sides


def splitSide(seg):
    return unitEdges([seg])


def pivotLong(seg, rng, box=None, dots=None, bias=None, grid_max=None):
    if bias is None:
        bias = connectBias
    if grid_max is None:
        grid_max = maxSpan
    (a, b) = seg
    v = (b[0] - a[0], b[1] - a[1])
    n = max(abs(v[0]), abs(v[1]))
    options = []
    for d in latticeDirs:
        if v[0] * d[1] - v[1] * d[0] == 0:
            continue
        for anchor in (a, b):
            far = (anchor[0] + d[0] * n, anchor[1] + d[1] * n)
            if min(far[0], far[1]) < 0 or max(far[0], far[1]) > grid_max:
                continue
            options.append((anchor, far))
    if not options:
        return None
    if box:
        (x0, y0, x1, y1) = box
        inside = [o for o in options if x0 <= o[1][0] <= x1 and y0 <= o[1][1] <= y1]
        if inside:
            options = inside
    if dots and rng.random() < bias:
        connecting = [o for o in options if o[1] in dots]
        if connecting:
            options = connecting
    return rng.choice(options)


def pivotSides(scene, k, rng, bias=None, grid_max=None):
    # swing k whole collinear sides about a vertex, then explode back to units
    if k <= 0:
        return list(scene)
    sides = mergeSides(scene)
    if k > len(sides):
        return None
    idxs = rng.sample(range(len(sides)), k)
    box = sceneBox(scene)
    remaining = [sides[i] for i in range(len(sides)) if i not in idxs]
    dots = {p for seg in remaining for p in seg}
    new_sides = list(remaining)
    for i in idxs:
        swung = pivotLong(sides[i], rng, box, dots, bias=bias, grid_max=grid_max)
        if swung is None:
            return None
        new_sides.append(swung)
        dots.update(swung)
    out = []
    for seg in new_sides:
        out.extend(splitSide(seg))
    if not validScene(out) or segmentsCross(out):
        return None
    return out


# Figure measures

def headings(scene):
    # undirected orientation of each line in degrees, 0-179
    return [round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 180
            for a, b in scene]

def headingProfile(scene):
    return Counter(headings(scene))

def dissimilarity(a, b):
    # 1 - overlap of two orientation profiles; 0 same look, 1 nothing shared
    total = max(sum(a.values()), sum(b.values()))
    return 1 - sum((a & b).values()) / total if total else 0.0

def headingEntropy(scene):
    hs = headings(scene)
    if not hs:
        return 0.0
    total = len(hs)
    return abs(-sum((c / total) * math.log2(c / total) for c in Counter(hs).values()))

def dotCount(scene):
    return len({p for seg in scene for p in seg})

def geomSig(scene):
    return tuple(sorted(tuple(sorted(seg)) for seg in scene))


# The structure score ("plain"): lines are filed into row and column tracks,
# each track scored independently, and plain is the sum over all tracks

def orient(line):
    (c1, r1), (c2, r2) = line
    if r1 == r2:
        return 'H'
    if c1 == c2:
        return 'V'
    return 'D' if (r2 - r1) * (c2 - c1) < 0 else 'A'

def turnAngle(o1, o2):
    if o1 == o2:
        return 0
    return 90 if {o1, o2} in ({'H', 'V'}, {'D', 'A'}) else 45

def structureValue(lines):
    n = len(lines)
    if n == 0:
        return 0
    ori = [orient(l) for l in lines]

    def connections(i):
        # how many of the line's endpoints are shared with any other line
        return sum(1 for p in lines[i] if any(j != i and p in lines[j] for j in range(n)))

    # a line spanning rows goes into row band min(r); spanning columns, column band min(c)
    rowTracks, colTracks = defaultdict(list), defaultdict(list)
    for i, ((c1, r1), (c2, r2)) in enumerate(lines):
        if r1 != r2:
            rowTracks[min(r1, r2)].append(i)
        if c1 != c2:
            colTracks[min(c1, c2)].append(i)

    rowLones = [ori[idx[0]] for idx in rowTracks.values() if len(idx) == 1]
    colLones = [ori[idx[0]] for idx in colTracks.values() if len(idx) == 1]

    def alongKey(kind, i):
        (c1, r1), (c2, r2) = lines[i]
        return (min(c1, c2), min(r1, r2)) if kind == 'row' else (min(r1, r2), min(c1, c2))

    def multiScore(idxs, kind):
        # base: 0 if parallel or only opposite diagonals, else 1
        # steps: walk sorted lines in non-overlapping triples, +1 when the
        # triple's two consecutive turn-angles differ
        os = {ori[i] for i in idxs}
        base = 0 if (len(os) == 1 or os <= {'D', 'A'}) else 1
        s = sorted(idxs, key=lambda i: alongKey(kind, i))
        steps = sum(1 for t in range(0, len(s) - 2, 3)
                    if turnAngle(ori[s[t]], ori[s[t + 1]]) != turnAngle(ori[s[t + 1]], ori[s[t + 2]]))
        return base + steps

    def loneScore(i, lones):
        # scored by connection count and whether a parallel lone peer exists
        peer = lones.count(ori[i]) >= 2
        c = connections(i)
        if peer:
            return 1 if c < 2 else 0
        return 2 if c == 0 else 1

    total = 0
    for kind, tracks, lones in (('row', rowTracks, rowLones), ('col', colTracks, colLones)):
        for idxs in tracks.values():
            total += multiScore(idxs, kind) if len(idxs) >= 2 else loneScore(idxs[0], lones)
    return total


# Assembling one example (within one shape category)
#
# Line count and cell budget are identical across levels by construction,
# so candidates only need to satisfy ascending structure and turtle length,
# distinct geometry, and visible orientation difference between consecutive levels.

def buildCandidates(name, u, levels, rng, rounds=1200):
    base = baselineScene(name, u)
    pool = {level: [] for level in range(levels)}
    if not fitsGrid(base):
        return pool
    counts = changedCounts(levels, len(base))
    for level, k in enumerate(counts):
        bias = levelBias[min(level, len(levelBias) - 1)]
        for _ in range(rounds if k else 1):
            scene = mutateScene(base, k, rng, bias=bias) if k else list(base)
            if scene and validScene(scene) and fitsGrid(scene):
                pool[level].append((structureValue(scene), turtleLength(scene), scene, k))
    return pool

def assemble(pool, levels, rng, attempts=4000, perBucket=40):
    if not all(pool[level] for level in range(levels)):
        return None
    ranked = {level: pool[level][:perBucket] for level in range(levels)}
    for _ in range(attempts):
        picks = [rng.choice(ranked[level]) for level in range(levels)]
        structs = [p[0] for p in picks]
        if not all(structs[j] < structs[j + 1] for j in range(levels - 1)):
            continue
        turtles = [p[1] for p in picks]
        if not all(turtles[j] < turtles[j + 1] for j in range(levels - 1)):
            continue
        if len({geomSig(p[2]) for p in picks}) != levels:
            continue
        profs = [headingProfile(p[2]) for p in picks]
        if any(dissimilarity(profs[j], profs[j + 1]) < minDiff - 1e-9 for j in range(levels - 1)):
            continue
        return [p[2] for p in picks], [p[3] for p in picks], [p[1] for p in picks]
    return None

def buildExample(name, u, levels, rng, attempts=4):
    for _ in range(attempts):
        result = assemble(buildCandidates(name, u, levels, rng), levels, rng)
        if result:
            return result
    return None


# Assembling one example in --objects mode
#
# Stimulus 1 is an interconnected chain: starting anywhere, each next line
# is drawn in a random direction from the end of the last; it never revisits
# a dot or crosses itself, so no closed shape can form, and its structure
# value may only be 0 or 1. Stimulus 2 is a baseline shape with at most two
# pivoted lines and a structure value of at least 2. Stimulus 3 is the usual
# top level. The shape is built first so the chain can copy its line count,
# which keeps lines and budget equal across the whole set.

def crossesDiagonal(seg, diagCells):
    # two diagonals through the same unit cell cross and enclose a region
    (a, b) = seg
    if a[0] == b[0] or a[1] == b[1]:
        return False
    return (min(a[0], b[0]), min(a[1], b[1])) in diagCells

def buildChain(nLines, rng):
    # the chain is a few long straight runs: nLines is split into 2-4 runs,
    # each run drawing one random direction, so the chain has at most three
    # turns and long rows of parallel lines, and unambiguously compresses to
    # a shorter program than any pivoted shape. It never revisits a dot (so
    # no closed shape can form) and never crosses an earlier diagonal.
    lowest = -(-nLines // maxSpan)                 # runs must fit on the grid
    nRuns = max(lowest, rng.choice((2, 3)) if nLines <= 12 else rng.choice((3, 4)))
    nRuns = max(1, min(nLines // 2, nRuns + rng.choice((0, 1))))
    runLens = []
    remaining = nLines
    for runsLeft in range(nRuns, 0, -1):           # bounded composition, 2..maxSpan
        low = max(2, remaining - (runsLeft - 1) * maxSpan)
        high = min(maxSpan, remaining - 2 * (runsLeft - 1))
        if low > high:
            return None
        part = rng.randint(low, high) if runsLeft > 1 else remaining
        runLens.append(part)
        remaining -= part

    cur = (0, 0)
    scene = []
    visited = {cur}
    diagCells = set()
    lo = [0, 0]
    hi = [0, 0]
    prev = None
    for length in runLens:
        dirs = [d for d in latticeDirs if d != prev and d != (-prev[0], -prev[1])] \
               if prev else list(latticeDirs)
        rng.shuffle(dirs)
        placed = None
        for d in dirs:                             # a direction the whole run can take
            pts = [(cur[0] + d[0] * s, cur[1] + d[1] * s) for s in range(1, length + 1)]
            xs = [p[0] for p in pts] + [lo[0], hi[0]]
            ys = [p[1] for p in pts] + [lo[1], hi[1]]
            if max(xs) - min(xs) > maxSpan or max(ys) - min(ys) > maxSpan:
                continue
            if any(p in visited for p in pts):
                continue
            segs = [((cur if s == 0 else pts[s - 1]), pts[s]) for s in range(length)]
            if d[0] and d[1] and any(crossesDiagonal(seg, diagCells) for seg in segs):
                continue
            placed = (d, pts, segs)
            break
        if placed is None:
            return None
        d, pts, segs = placed
        prev = d
        for seg, p in zip(segs, pts):
            scene.append(seg)
            visited.add(p)
            if d[0] and d[1]:
                diagCells.add((min(seg[0][0], seg[1][0]), min(seg[0][1], seg[1][1])))
        lo = [min(lo[0], min(p[0] for p in pts)), min(lo[1], min(p[1] for p in pts))]
        hi = [max(hi[0], max(p[0] for p in pts)), max(hi[1], max(p[1] for p in pts))]
        cur = pts[-1]
    return scene

def hasDiagonal(scene):
    return any(a[0] != b[0] and a[1] != b[1] for a, b in scene)

def chainOk(chain):
    # a diagonal line files into both a row and a column track, doubling its
    # lone-line penalties in plain, so diagonal chains get a compensated
    # bound of 2; purely horizontal / vertical chains must score 0 or 1
    return structureValue(chain) <= (2 if hasDiagonal(chain) else 1)

# The --missMisplace sub-mode of --objects
#
# Both lower levels unfold a closed baseline: a line pivots about its own
# endpoint with its far end left free. Level 1 unfolds two lines, level 2
# Both lower levels unfold a closed baseline: a line pivots about its own
# endpoint with its far end left free. Level 1 unfolds two lines, level 2
# unfolds one. Turtle length and structure must still ascend.

def baselineRing(name, u):
    steps = [s for s in latticeShapes(u) if s[0] == name][0][1]
    pts = path(steps)
    return pts[:-1] if pts[0] == pts[-1] else None

def insideRing(pt, ring):
    # even-odd ray cast; pt may be fractional
    x, y = pt
    inside = False
    n = len(ring)
    for i in range(n):
        (x1, y1), (x2, y2) = ring[i], ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                inside = not inside
    return inside

def ringHeads(base):
    return [dirIndex[(b[0] - a[0], b[1] - a[1])] for a, b in base]

def compressedProgram(items):
    # items: 'E'-style turn expressions per stroke, or ('pivot', angle, walk)
    # runs of identical plain strokes fold into forLoop, so the program is
    # the shortest loop-structured description of the figure
    body = '$0'
    i = len(items) - 1
    while i >= 0:
        it = items[i]
        if isinstance(it, tuple):
            if it[0] == 'flap':
                _, a1, a2, w1, w2 = it
                floater = (f"(logo_FWRT logo_ZL {eighth(a1)} (logo_FWRT logo_UL {eighth(a2)} "
                           f"(logo_FWRT logo_UL logo_ZA $0)))")
                walker = f"(logo_FWRT logo_UL {w1} (logo_FWRT logo_UL {w2} $0))"
            else:
                _, angle, walk = it
                floater = f"(logo_FWRT logo_ZL {angle} (logo_FWRT logo_UL logo_ZA $0))"
                walker = f"(logo_FWRT logo_UL {walk} $0)"
            body = f"(logo_GETSET (lambda {floater}) (logo_PT (lambda {walker}) {body}))"
            i -= 1
            continue
        j = i
        while j >= 0 and items[j] == it:
            j -= 1
        run = i - j
        stroke = f"(logo_FWRT logo_UL {it} $0)"
        if run >= 2:
            body = f"(logo_forLoop {run} (lambda (lambda (logo_FWRT logo_UL {it} $0))) {body})"
        else:
            body = f"(logo_FWRT logo_UL {it} {body})"
        i = j
    return body

def unfoldPair(base, rng):
    # two consecutive lines swing away as one connected arm: line i pivots
    # about its hinge and line i+1 continues from line i's new far end; both
    # arm dots end up free of the rest, so the shape is opened by a two-line
    # flap, the pairwise analogue of the single unfolded line
    L = len(base)
    heads = ringHeads(base)
    for _ in range(40):
        i = rng.randrange(L - 1)
        hinge = base[i][0]
        d1s = [d for d in latticeDirs
               if d[0] * (base[i][1][1] - base[i][0][1])
               - d[1] * (base[i][1][0] - base[i][0][0]) != 0]
        rng.shuffle(d1s)
        done = None
        for d1 in d1s:
            p1 = (hinge[0] + d1[0], hinge[1] + d1[1])
            d2s = [d for d in latticeDirs if d != (-d1[0], -d1[1]) and d != d1]
            rng.shuffle(d2s)
            for d2 in d2s:
                p2 = (p1[0] + d2[0], p1[1] + d2[1])
                scene = list(base)
                scene[i] = (hinge, p1)
                scene[i + 1] = (p1, p2)
                rest = {p for j, s in enumerate(scene) if j not in (i, i + 1) for p in s}
                if p1 in rest or p2 in rest:
                    continue
                if not validScene(scene) or not fitsGrid(scene):
                    continue
                if pivotContinues(scene, [i, i + 1]):
                    continue
                a1 = (dirIndex[d1] - heads[i]) % 8
                a2 = (dirIndex[d2] - dirIndex[d1]) % 8
                done = (scene, (i, a1, a2))
                break
            if done:
                break
        if done:
            return done
    return None, None

def compressedNats(heads, pivots=None, flap=None):
    # pivots maps stroke slots to drawn angles; each becomes a GETSET / PT
    # fragment inside the loop-compressed ring program
    turns = [(heads[i + 1] - heads[i]) % 8 for i in range(len(heads) - 1)] + [0]
    items = []
    skip = set()
    for i, t in enumerate(turns):
        if i in skip:
            continue
        if flap and i == flap[0]:
            items.append(('flap', flap[1], flap[2], eighth(t), eighth(turns[i + 1])))
            skip.add(i + 1)
        elif pivots and i in pivots:
            items.append(('pivot', eighth(pivots[i]), eighth(t)))
        else:
            items.append(eighth(t))
    body = compressedProgram(items)
    raise RuntimeError('nats scoring removed; use turtleLength(scene)')

def wrongTurn(base, rng):
    # alter one turn of the ring: the downstream arc rotates as a block, so
    # the object stays one connected chain but its geometry no longer closes;
    # the program changes by a single turn token plus a loop split
    heads = ringHeads(base)
    n = len(heads)
    for _ in range(30):
        slot = rng.randrange(1, n)
        delta = rng.choice((1, 2, -1, -2, 3, -3))
        newHeads = heads[:slot] + [(h + delta) % 8 for h in heads[slot:]]
        cur = base[0][0]
        scene = []
        for h in newHeads:
            d = latticeDirs[h] if False else [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)][h]
            nxt = (cur[0] + d[0], cur[1] + d[1])
            scene.append((cur, nxt))
            cur = nxt
        if scene[-1][1] == base[0][0]:              # still closes: not a violation
            continue
        if validScene(scene) and fitsGrid(scene):
            return scene, slot
    return None, None

def misplaceInward(base, ring, rng):
    # remove one line and re-attach it at another dot, pointing inside
    i = rng.randrange(len(base))
    rest = base[:i] + base[i + 1:]
    dots = list({p for seg in rest for p in seg})
    rng.shuffle(dots)
    dirs = list(latticeDirs)
    for anchor in dots:
        rng.shuffle(dirs)
        for d in dirs:
            far = (anchor[0] + d[0], anchor[1] + d[1])
            mid = (anchor[0] + d[0] / 2, anchor[1] + d[1] / 2)
            if not (insideRing(far, ring) and insideRing(mid, ring)):
                continue
            scene = rest + [(anchor, far)]
            if not validScene(scene):
                continue
            if pivotContinues(scene, [len(scene) - 1]):
                continue
            return scene
    return None

def openOutline(base, rng, k=1, idxs=None):
    # k standard pivots with no connectedness bias, every far end left free,
    # so the outline visibly unfolds at k places
    if idxs is None:
        idxs = rng.sample(range(len(base)), k)
    scene = list(base)
    pivots = {}
    for i in idxs:
        seg = pivot(base[i], rng, sceneBox(base))
        scene[i] = seg
        anchor = seg[0]
        d = (seg[1][0] - anchor[0], seg[1][1] - anchor[1])
        pivots[i] = (dirIndex[d] - ringHeads(base)[i]) % 8
    for i in idxs:
        far = scene[i][1]
        if far in {p for j, s in enumerate(scene) if j != i for p in s}:
            return None, None
    if not validScene(scene) or not fitsGrid(scene):
        return None, None
    if pivotContinues(scene, idxs):
        return None, None
    return scene, pivots

def missingSide(base, rng):
    # remove one whole side of the ring: a maximal run of same-direction
    # lines, so the shape is unfinished rather than perturbed
    heads = ringHeads(base)
    L = len(base)
    runs = []
    start = 0
    for i in range(1, L + 1):
        if i == L or heads[i] != heads[start]:
            runs.append((start, i))
            start = i
    if len(runs) > 1 and heads[0] == heads[-1]:     # ring wrap joins a side
        (s0, e0), (s1, e1) = runs[0], runs.pop()
        runs[0] = (s1, e1 + e0)                     # indices mod L
    (s, e) = rng.choice(runs)
    keep = [base[j % L] for j in range(e, s + L)]
    return keep

def buildMissPool(rng, rounds=200):
    # level 1: unfinished shapes (one whole side missing); level 2: complete
    # shapes with one line unfolded; keys carry the line count for matching
    pool = defaultdict(list)
    for name, _ in library():
        for u in (1, 2, 3):
            base = baselineScene(name, u)
            if not validScene(base) or not fitsGrid(base):
                continue
            if baselineRing(name, u) is None:
                continue
            baseHeads = ringHeads(base)
            for _ in range(rounds // 10):
                scene = missingSide(base, rng)
                if validScene(scene) and fitsGrid(scene):
                    pool[(0, len(scene))].append((structureValue(scene),
                                                  turtleLength(scene),
                                                  name, scene, len(base) - len(scene)))
            ring = baselineRing(name, u)
            for _ in range(rounds):
                scene, pivots = openOutline(base, rng, 1)
                if not scene:
                    continue
                i = next(iter(pivots))
                (a, b) = scene[i]
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                if insideRing(b, ring) or insideRing(mid, ring):
                    continue                        # must point outward: open percept
                pool[(1, len(base))].append((structureValue(scene),
                                             turtleLength(scene),
                                             name, scene, 1))
    return pool

def buildMissAcross(pool, rng, attempts=20000, anchor=None):
    sizes = {key[1] for key in pool}
    shared = [s for s in sizes if all(pool.get((level, s)) for level in range(2))]
    if not shared:
        return None
    for _ in range(attempts):
        size = rng.choice(shared)
        picks = [rng.choice(pool[(level, size)]) for level in range(2)]
        names = [p[2] for p in picks]
        if len(set(names)) != 2:                    # two distinct categories
            continue
        if anchor and names[0] != anchor:           # level 1 anchors the sweep
            continue
        if len({geomSig(p[3]) for p in picks}) != 2:
            continue
        if not (picks[0][0] < picks[1][0] and picks[0][1] < picks[1][1]):
            continue
        return names, [p[3] for p in picks], [p[4] for p in picks], [p[1] for p in picks]
    return None

def buildObjects(name, u, rng, rounds=600, attempts=8):
    base = baselineScene(name, u)
    if not fitsGrid(base):
        return None
    nLines = len(base)
    k3 = changedCounts(3, nLines)[2]

    for _ in range(attempts):
        pool = {0: [], 1: [], 2: []}
        diag = 0
        for _ in range(60 * rounds):               # low-structure chains are rare
            chain = buildChain(nLines, rng)
            if not (chain and validScene(chain) and fitsGrid(chain) and chainOk(chain)):
                continue
            if hasDiagonal(chain):
                diag += 1
            elif len(pool[0]) - diag >= 20:        # keep room for diagonal chains
                continue
            pool[0].append((structureValue(chain), turtleLength(chain), chain, 0))
            if len(pool[0]) >= 40:
                break
        for _ in range(rounds):
            k2 = rng.choice((1, 2))
            scene = mutateScene(base, k2, rng, bias=levelBias[1])
            if scene and validScene(scene) and fitsGrid(scene) and structureValue(scene) >= 2:
                pool[1].append((structureValue(scene), turtleLength(scene), scene, k2))
        for _ in range(2 * rounds):
            scene = mutateScene(base, k3, rng, bias=levelBias[2])
            if scene and validScene(scene) and fitsGrid(scene):
                pool[2].append((structureValue(scene), turtleLength(scene), scene, k3))
        result = assemble(pool, 3, rng)
        if result:
            scenes, ks, turtles = result
            if turtles[0] < turtles[1] < turtles[2]:
                return result
    return None

def buildPool(levels, rng, rounds=120):
    pool = defaultdict(list)
    for name, _ in library():
        for u in (1, 2, 3):
            base = baselineScene(name, u)
            if not validScene(base) or not fitsGrid(base):
                continue
            counts = changedCounts(levels, len(base), acrossFracs)
            for _ in range(rounds):
                for level, k in enumerate(counts):
                    scene = mutateScene(base, k, rng, bias=levelBias[min(level, 2)])
                    if not scene or not validScene(scene) or not fitsGrid(scene):
                        continue
                    pool[(level, len(base))].append((structureValue(scene), turtleLength(scene),
                                                     name, scene, k))
    return pool

def buildAcross(pool, levels, rng, attempts=20000, anchor=None):
    sizes = {key[1:] for key in pool}
    shared = [s for s in sizes if all(pool.get((level,) + s) for level in range(levels))]
    if not shared:
        return None
    for _ in range(attempts):
        size = rng.choice(shared)
        picks = [rng.choice(pool[(level,) + size]) for level in range(levels)]
        names = [p[2] for p in picks]
        if len(set(names)) != levels:               # three distinct categories
            continue
        if anchor and names[0] != anchor:            # level 1 must be the anchor shape
            continue
        scenes = [p[3] for p in picks]
        structs = [structureValue(s) for s in scenes]
        if not all(structs[j] < structs[j + 1] for j in range(levels - 1)):
            continue
        turtles = [p[1] for p in picks]
        if not all(turtles[j] < turtles[j + 1] for j in range(levels - 1)):
            continue                                # mixing shapes can invert turtle length
        if len({geomSig(s) for s in scenes}) != levels:
            continue
        profs = [headingProfile(s) for s in scenes]
        if any(dissimilarity(profs[j], profs[j + 1]) < minDiff - 1e-9 for j in range(levels - 1)):
            continue
        return names, scenes, [p[4] for p in picks], [p[1] for p in picks]
    return None


# Plotting
#
# One constant grid for the whole run: gridN is sized to the largest figure
# generated, panels are shifted by whole lattice steps only, so every
# endpoint sits exactly on a drawn dot.

def drawStimulus(ax, scene, full_grid=False):
    c = cellsPerLine
    gx = [i for i in range(gridN + 1) for _ in range(gridN + 1)]
    gy = [j for _ in range(gridN + 1) for j in range(gridN + 1)]
    ax.scatter(gx, gy, s=10, color=gridColor, zorder=0)
    if scene:
        xs = [p[0] for seg in scene for p in seg]
        ys = [p[1] for seg in scene for p in seg]
        ox = ((gridCoarse - (max(xs) - min(xs))) // 2 - min(xs)) * c
        oy = ((gridCoarse - (max(ys) - min(ys))) // 2 - min(ys)) * c
        for (a, b) in scene:
            ax.plot([a[0] * c + ox, b[0] * c + ox], [a[1] * c + oy, b[1] * c + oy],
                    color=lineColor, linewidth=lineWidth, zorder=2)
        px = [p[0] * c + ox for seg in scene for p in seg]
        py = [p[1] * c + oy for seg in scene for p in seg]
        ax.scatter(px, py, s=70, color=lineColor, zorder=3)
    if full_grid:
        ax.set_xlim(-0.5, gridN + 0.5)
        ax.set_ylim(-0.5, gridN + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')


def stimulusFigure(scene, size=4, full_grid=False):
    fig, ax = plt.subplots(figsize=(size, size))
    drawStimulus(ax, scene, full_grid=full_grid)
    fig.tight_layout()
    return fig


def render(scenes, path):
    fig, axes = plt.subplots(1, len(scenes), figsize=(4 * len(scenes), 4))
    axes = axes if len(scenes) > 1 else [axes]
    for ax, scene in zip(axes, scenes):
        drawStimulus(ax, scene)
    fig.tight_layout()
    fig.savefig(path + '.eps', format='eps', bbox_inches='tight')
    fig.savefig(path + '.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def renderStimulus(scene, path):
    # one grid per file, same styling as the example strips
    fig = stimulusFigure(scene)
    fig.savefig(path + '.eps', format='eps', bbox_inches='tight')
    fig.savefig(path + '.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


# Object-set Level 3s
#
# One Level 3 per set, reused in every basic-structure sub-condition.
# Across-shapes: same line count as that set's Level 2, but a different
# closed family, then warped into a connected drawing (Previous-stimuli
# Level 2 style). Isolated sticks are out. Turtle length and structure
# must exceed Closed and Target Level 2s.

OBJECT_SET_ROOT = Path(__file__).resolve().parent / 'object_sets'
BASIC_ROOT = OBJECT_SET_ROOT / 'Basic structure condition'
BASIC_FOLDERS = ('Closed object', 'Target', 'Broken object')

def tu(segs):
    return [((a[0] // 5, a[1] // 5), (b[0] // 5, b[1] // 5)) for a, b in segs]

def fine(scene):
    return [((a[0] * 5, a[1] * 5), (b[0] * 5, b[1] * 5)) for a, b in scene]

OBJECT_CLOSED = {
    11: tu([((0, 5), (0, 10)), ((0, 5), (5, 5)), ((0, 10), (0, 15)), ((0, 15), (5, 15)),
            ((5, 5), (10, 5)), ((5, 15), (10, 15)), ((10, 5), (15, 5)), ((10, 15), (15, 15)),
            ((15, 5), (20, 5)), ((15, 15), (20, 15)), ((20, 5), (20, 10)), ((20, 10), (20, 15))]),
    12: tu([((0, 10), (5, 10)), ((0, 10), (5, 15)), ((5, 10), (10, 10)), ((5, 15), (10, 20)),
            ((10, 10), (15, 10)), ((10, 20), (15, 15)), ((15, 10), (20, 10)), ((15, 15), (20, 10))]),
    13: tu([((5, 10), (5, 15)), ((5, 10), (10, 5)), ((5, 15), (10, 20)), ((10, 5), (15, 5)),
            ((10, 20), (15, 20)), ((15, 5), (20, 10)), ((15, 20), (20, 15)), ((20, 10), (20, 15))]),
    14: tu([((5, 10), (10, 5)), ((5, 10), (10, 15)), ((10, 5), (15, 5)), ((10, 15), (15, 15)),
            ((15, 5), (20, 10)), ((15, 15), (20, 10))]),
    15: tu([((0, 5), (5, 5)), ((0, 5), (5, 10)), ((5, 5), (10, 5)), ((5, 10), (10, 10)),
            ((10, 5), (15, 5)), ((10, 10), (15, 10)), ((15, 5), (20, 5)), ((15, 10), (20, 5))]),
}
OBJECT_TARGET = {
    11: tu([((0, 5), (0, 10)), ((0, 5), (5, 5)), ((0, 10), (0, 15)), ((0, 15), (5, 15)),
            ((5, 5), (10, 5)), ((10, 5), (15, 5)), ((10, 15), (15, 15)), ((10, 15), (15, 20)),
            ((15, 5), (20, 5)), ((15, 15), (20, 15)), ((20, 5), (20, 10)), ((20, 10), (20, 15))]),
    12: tu([((0, 10), (5, 10)), ((0, 10), (5, 15)), ((5, 10), (10, 10)), ((5, 15), (10, 20)),
            ((10, 5), (15, 10)), ((10, 20), (15, 15)), ((15, 10), (20, 10)), ((15, 15), (20, 10))]),
    13: tu([((5, 10), (5, 15)), ((5, 10), (10, 10)), ((5, 15), (10, 20)), ((10, 5), (15, 5)),
            ((10, 20), (15, 20)), ((15, 5), (20, 10)), ((15, 20), (20, 15)), ((20, 10), (20, 15))]),
    14: tu([((5, 10), (10, 5)), ((5, 10), (10, 15)), ((5, 20), (10, 15)), ((10, 5), (15, 5)),
            ((15, 5), (20, 10)), ((15, 15), (20, 10))]),
    15: tu([((0, 5), (5, 5)), ((0, 5), (5, 10)), ((5, 5), (10, 5)), ((5, 10), (10, 10)),
            ((10, 0), (15, 5)), ((10, 10), (15, 10)), ((15, 5), (20, 5)), ((15, 10), (20, 5))]),
}
OBJECT_BROKEN = {
    # The opened Target line is the misplaced line (11/13/15 inside, 12/14 outside).
    11: tu([((0, 5), (0, 10)), ((0, 5), (5, 5)), ((0, 10), (0, 15)), ((0, 15), (5, 15)),
            ((5, 5), (10, 5)), ((10, 5), (15, 5)), ((10, 15), (15, 15)), ((15, 5), (20, 5)),
            ((15, 15), (20, 15)), ((20, 5), (20, 10)), ((20, 10), (20, 15)), ((5, 5), (10, 10))]),
    12: tu([((0, 10), (5, 10)), ((0, 10), (5, 15)), ((5, 10), (10, 10)), ((5, 15), (10, 20)),
            ((10, 20), (15, 15)), ((15, 10), (20, 10)), ((15, 15), (20, 10)), ((20, 5), (25, 10))]),
    13: tu([((5, 10), (5, 15)), ((5, 15), (10, 20)), ((10, 5), (15, 5)), ((10, 20), (15, 20)),
            ((15, 5), (20, 10)), ((15, 20), (20, 15)), ((20, 10), (20, 15)), ((10, 10), (15, 10))]),
    14: tu([((5, 10), (10, 5)), ((5, 10), (10, 15)), ((10, 5), (15, 5)), ((15, 5), (20, 10)),
            ((15, 15), (20, 10)), ((15, 20), (20, 15))]),
    15: tu([((0, 5), (5, 5)), ((0, 5), (5, 10)), ((5, 5), (10, 5)), ((5, 10), (10, 10)),
            ((10, 10), (15, 10)), ((15, 5), (20, 5)), ((15, 10), (20, 5)), ((5, 5), (10, 10))]),
}

PNG_SIZE = 843
PNG_ORIGIN = 19.309091 * 3
PNG_STEP = 9.687272727 * 3
PNG_GRAY = (214, 214, 214)
PNG_BLACK = (0, 0, 0)


def renderObjectSet(scene_tu, path):
    segs = fine(scene_tu)
    image = Image.new('RGB', (PNG_SIZE, PNG_SIZE), 'white')
    draw = ImageDraw.Draw(image)

    def lattice(i, j):
        return (PNG_ORIGIN + i * PNG_STEP, PNG_SIZE - 1 - (PNG_ORIGIN + j * PNG_STEP))

    def dot(center, radius, color):
        x, y = center
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    for i in range(26):
        for j in range(26):
            dot(lattice(i, j), 5, PNG_GRAY)
    nodes = set()
    for a, b in segs:
        draw.line((lattice(*a), lattice(*b)), fill=PNG_BLACK, width=17)
        nodes.update((a, b))
    for node in nodes:
        dot(lattice(*node), 13, PNG_BLACK)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


OBJECT_GRID_MAX = 5

# Sets 13–14 Level 3s are one connected drawing. Sets 11, 12, and 15 are two groups.
OBJECT_L3_CONNECTED = {13, 14}
OBJECT_L3_GROUPS = {11, 12, 15}
OBJECT_L2_SHAPE = {
    11: 'rectangle x2',
    12: 'triangle',
    13: 'octagon',
    14: 'hexagon',
    15: 'trapezium',
}

# Preferred other families at the same line count, most different first.
OBJECT_L3_SOURCES = {
    11: [('square', 3)],
    12: [('L-shape', 1), ('staircase', 2), ('diamond', 2), ('hook', 2)],
    13: [('diamond', 2), ('L-shape', 1), ('staircase', 2), ('square', 2)],
    14: [('trapezium', 1), ('parallelogram', 1), ('rectangle x2', 1)],
    15: [('staircase', 2), ('hook', 2), ('arrow', 2), ('square', 2), ('L-shape', 1)],
}


def fitsObjectGrid(scene):
    return all(0 <= x <= OBJECT_GRID_MAX and 0 <= y <= OBJECT_GRID_MAX
               for seg in scene for x, y in seg)


def placeOnObjectGrid(scene, rng=None):
    xs = [p[0] for seg in scene for p in seg]
    ys = [p[1] for seg in scene for p in seg]
    minx, miny = min(xs), min(ys)
    w, h = max(xs) - minx, max(ys) - miny
    if w > OBJECT_GRID_MAX or h > OBJECT_GRID_MAX:
        return None
    if rng is None:
        dx = (OBJECT_GRID_MAX - w) // 2 - minx
        dy = max(1, (OBJECT_GRID_MAX - h) // 2) - miny
        if dy + h > OBJECT_GRID_MAX:
            dy = OBJECT_GRID_MAX - h - miny
    else:
        dx = rng.randint(0, OBJECT_GRID_MAX - w) - minx
        dy = rng.randint(0, OBJECT_GRID_MAX - h) - miny
    return [((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy)) for a, b in scene]


def otherShapesForSet(set_id, forbid_names=None):
    n = len(OBJECT_CLOSED[set_id])
    forbid = {OBJECT_L2_SHAPE[set_id]}
    if forbid_names:
        forbid |= set(forbid_names)
    ordered = []
    seen = set()
    for name, u in OBJECT_L3_SOURCES[set_id]:
        if name in forbid or (name, u) in seen:
            continue
        seen.add((name, u))
        ordered.append((name, u))
    for name, steps in latticeShapes(1):
        if name in forbid:
            continue
        for u in (1, 2, 3, 4):
            if (name, u) in seen:
                continue
            if len(baselineScene(name, u)) != n:
                continue
            seen.add((name, u))
            ordered.append((name, u))
    out = []
    for name, u in ordered:
        placed = placeOnObjectGrid(baselineScene(name, u))
        if placed and len(placed) == n and validScene(placed) and fitsObjectGrid(placed):
            out.append((name, u, placed))
    return out


def figureOK(scene, connected=False, grouped=False):
    # One connected drawing, or two multi-segment pieces
    if not scene or not validScene(scene) or not fitsObjectGrid(scene):
        return False
    if segmentsCross(scene):
        return False
    sizes = [len(c) for c in sceneComponents(scene)]
    if grouped:
        return len(sizes) == 2 and min(sizes) >= 2
    if len(sizes) == 1:
        return True
    if connected:
        return False
    return len(sizes) == 2 and min(sizes) >= 2


def figureQuality(scene):
    # Previous L2s: one piece, mixed H/V/diag, a few junctions — not a triangulated mesh
    ncomp = len(sceneComponents(scene))
    oris = len({orient(seg) for seg in scene})
    deg = Counter(p for seg in scene for p in seg)
    branches = sum(1 for d in deg.values() if d >= 3)
    return (ncomp == 1, min(oris, 3), 1 if 1 <= branches <= 3 else 0)


def spreadComponents(scene):
    # Two pieces: slide the smaller one around the object grid to lengthen
    # the turtle tour (more Forward between pieces) without scattering.
    comps = sceneComponents(scene)
    if len(comps) != 2:
        return scene
    segs = list(scene)
    pieces = [[segs[i] for i in idxs] for idxs in comps]
    small, big = sorted(pieces, key=len)
    xs = [p[0] for seg in small for p in seg]
    ys = [p[1] for seg in small for p in seg]
    best, best_t = scene, turtleLength(scene)
    for dx in range(-min(xs), OBJECT_GRID_MAX - max(xs) + 1):
        for dy in range(-min(ys), OBJECT_GRID_MAX - max(ys) + 1):
            moved = [((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy))
                     for a, b in small]
            cand = big + moved
            if not figureOK(cand, grouped=True):
                continue
            t = turtleLength(cand)
            if t > best_t:
                best, best_t = cand, t
    return best


def warpConnected(scene, k, rng, bias=0.9, connected=False):
    # successive one-line pivots that keep the figure attached
    out = list(scene)
    moved = 0
    for _ in range(k * 40):
        if moved >= k:
            break
        cand = pivotScene(out, 1, rng, bias=bias)
        if cand and figureOK(cand, connected=connected):
            out = cand
            moved += 1
    if moved < max(2, (k + 1) // 2):
        return None
    return out


def warpFromClosed(base, k, rng, connected=False, grouped=False):
    n_sides = len(mergeSides(base))
    if n_sides >= 3 and rng.random() < 0.5:
        ks = rng.randint(1, min(2, n_sides - 1))
        scene = pivotSides(base, ks, rng, bias=0.9, grid_max=OBJECT_GRID_MAX)
        if scene and figureOK(scene, connected=connected):
            extra = max(0, k - ks)
            if extra:
                warped = warpConnected(scene, extra, rng, connected=connected)
                if warped:
                    return warped
            return scene
    if grouped:
        bias = 0.55
    elif connected:
        bias = 0.9
    else:
        bias = 0.9 if rng.random() < 0.65 else 0.55
    return warpConnected(base, k, rng, bias=bias, connected=connected)


def objectSetThresholds(set_id):
    scenes = [OBJECT_CLOSED[set_id], OBJECT_TARGET[set_id], OBJECT_BROKEN[set_id]]
    turtles = [turtleLength(s) for s in scenes]
    structs = [structureValue(s) for s in scenes]
    # L3 must beat every L2, especially Broken, on turtle length and structure
    return max(turtles), max(structs), turtles, structs


def buildObjectSetLevel3(set_id, rng, attempts=700, forbid_names=None):
    need_t, need_s, t2, s2 = objectSetThresholds(set_id)
    n = len(OBJECT_CLOSED[set_id])
    closed_prof = headingProfile(OBJECT_CLOSED[set_id])
    connected = set_id in OBJECT_L3_CONNECTED
    grouped = set_id in OBJECT_L3_GROUPS
    best_pass = None
    best_any = None
    sources = otherShapesForSet(set_id, forbid_names)
    if not sources:
        return None
    k_lo = max(2, n // 3)
    k_hi = max(5, n - 1)
    for name, u, base0 in sources:
        for k in range(k_hi, k_lo - 1, -1):
            for _ in range(attempts):
                base = placeOnObjectGrid(base0, rng) or base0
                scene = warpFromClosed(base, k, rng, connected=connected, grouped=grouped)
                if not scene or not figureOK(scene, connected=connected, grouped=grouped):
                    continue
                if grouped:
                    scene = spreadComponents(scene)
                if dissimilarity(headingProfile(scene), closed_prof) < minDiff:
                    continue
                s = structureValue(scene)
                t = turtleLength(scene)
                rec = (scene, k, t, s, need_t, need_s, t2, s2, name)
                # Largest turtle/structure gap over L2, then drawing quality
                qual = (t - need_t, s - need_s, t, s) + figureQuality(scene)
                if t > need_t and s > need_s:
                    if best_pass is None or qual > best_pass[0]:
                        best_pass = (qual, rec)
                if best_any is None or qual > best_any[0]:
                    best_any = (qual, rec)
    if best_pass:
        return best_pass[1]
    return None if best_any is None else best_any[1]


def writeObjectSetLevel3s(rng, only_sets=None):
    rows = []
    used = set()
    for set_id in (11, 12, 13, 14, 15):
        if only_sets is not None and set_id not in only_sets:
            continue
        result = buildObjectSetLevel3(set_id, rng, forbid_names=used)
        if result is None:
            print(f'Set{set_id}: no Level 3 candidate')
            continue
        scene, k, t, s, need_t, need_s, t2, s2, src = result
        used.add(src)
        ok = t > need_t and s > need_s
        name = f'Set{set_id}Level3.png'
        for folder in BASIC_FOLDERS:
            renderObjectSet(scene, BASIC_ROOT / folder / name)
        rows.append((set_id, src, k, t, s, need_t, need_s, t2, s2, ok))
        flag = 'ok' if ok else 'BEST-EFFORT (did not beat all L2s, including Broken)'
        ncomp = len(sceneComponents(scene))
        print(f'Set{set_id} L3 {flag} from {src} (not {OBJECT_L2_SHAPE[set_id]}): '
              f'k={k} turtle {t}>{need_t} structure {s}>{need_s} comps={ncomp} '
              f'(L2 turtle C/T/B={t2} struct={s2})')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--examples', type=int, default=6)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--across', action='store_true',
                        help='draw each level from a different shape category')
    parser.add_argument('--library', action='store_true',
                        help='one example per distinct baseline shape')
    parser.add_argument('--objects', action='store_true',
                        help='chain, lightly pivoted shape, then the usual top level')
    parser.add_argument('--missMisplace', action='store_true',
                        help='with --objects: inward misplacement, opened outline, top level')
    parser.add_argument('--objectSets', action='store_true',
                        help='write Level 3s for Structure Validation object sets 11-15')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.objectSets:
        writeObjectSetLevel3s(rng)
        return

    rng = random.Random(args.seed)
    shapes = library()
    rng.shuffle(shapes)
    pool = buildPool(args.levels, rng) if args.across else None

    examples = []
    if args.objects and args.missMisplace:
        args.levels = 2                             # this sub-mode has two levels
        missPool = buildMissPool(rng)
        if args.library:
            for name, u in library():
                result = buildMissAcross(missPool, rng, anchor=name)
                if result:
                    examples.append(result)
                else:
                    print(f"skipped {name} (no missMisplace partners at its line count)")
        else:
            anchors = sorted({e[2] for key in missPool if key[0] == 0
                              for e in missPool[key]})
            feasible = [a for a in anchors if buildMissAcross(missPool, rng, 4000, a)]
            rng.shuffle(feasible)
            for i in range(args.examples):          # strict cycle: equal shares
                result = None
                while result is None:
                    result = buildMissAcross(missPool, rng, 20000,
                                             feasible[i % len(feasible)])
                examples.append(result)
    elif args.library and args.objects:
        for name, u in library():
            built = buildObjects(name, u, rng)
            if built:
                examples.append((['chain', name, name], built[0], built[1], built[2]))
            else:
                print(f"skipped {name} (no valid objects example)")
    elif args.objects:
        si = 0
        for i in range(args.examples):
            built = None
            while built is None:
                name, u = shapes[si % len(shapes)]
                si += 1
                built = buildObjects(name, u, rng)
            examples.append((['chain', name, name], built[0], built[1], built[2]))
    elif args.library and args.across:
        for name, u in library():
            result = buildAcross(pool, args.levels, rng, anchor=name)
            if result:
                examples.append(result)
            else:
                print(f"skipped {name} (no across partners at its line count)")
    elif args.library:
        for name, u in library():
            built = buildExample(name, u, args.levels, rng)
            if built:
                examples.append(([name] * args.levels, built[0], built[1], built[2]))
            else:
                print(f"skipped {name} (no valid example)")
    else:
        si = 0
        for i in range(args.examples):
            result = None
            while result is None:
                if args.across:
                    result = buildAcross(pool, args.levels, rng)
                else:
                    name, u = shapes[si % len(shapes)]
                    si += 1
                    built = buildExample(name, u, args.levels, rng)
                    if built:
                        result = ([name] * args.levels, built[0], built[1], built[2])
            examples.append(result)

    manifest = []
    stim = 0
    for i, (names, scenes, counts, turtles) in enumerate(examples, start=1):
        render(scenes, f"example_{i}")
        for level, (name, scene, nRand, n) in enumerate(
                zip(names, scenes, counts, turtles), start=1):
            stim += 1
            renderStimulus(scene, f"Stimulus{stim}")
            manifest.append((f"example_{i}", f"Stimulus{stim}", name, len(scene),
                             dotCount(scene), sceneBudget(scene), level, n,
                             structureValue(scene), nRand, headingEntropy(scene)))

    with open('examples_turtle.txt', 'w') as f:
        f.write(f"{'file':<12}{'stimulus':<12}{'shape':<15}{'lines':>7}{'dots':>6}"
                f"{'budget':>8}{'level':>7}{'turtle':>8}{'structure':>11}{'nRand':>7}{'turnH':>8}\n")
        for row in manifest:
            f.write(f"{row[0]:<12}{row[1]:<12}{row[2]:<15}{row[3]:>7}{row[4]:>6}"
                    f"{row[5]:>8}{row[6]:>7}{row[7]:>8}{row[8]:>11}{row[9]:>7}{row[10]:>8.2f}\n")

    mode = ('library sweep' if args.library
            else 'across categories' if args.across or args.missMisplace
            else 'within category')
    print(f"generated {len(examples)} examples ({mode}) on the fixed {gridN}x{gridN} grid")


if __name__ == '__main__':
    main()
