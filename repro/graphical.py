"""Graphical incompatibility scores for bivariate causal statements.

Faithful implementation of Section 3 and Appendix C of arXiv:2606.00278.

A *statement graph* is an acyclic directed mixed graph (ADMG) carried as a pair
of boolean matrices:

* ``D[u, v] is True``  <=>  directed edge ``u -> v``
* ``Bd[u, v] is True`` <=>  bidirected edge ``u <-> v`` (kept symmetric)

Lemma 3.5 characterises graphical compatibility by three properties, and
Definition 3.6 defines ``incomp(G)`` as the minimum Hamming distance to a graph
satisfying them.  Appendix C gives the three greedy algorithms whose total edit
count ``c(G)`` (equation (5)) upper-bounds ``incomp(G)``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "StatementGraph",
    "transitive_closure",
    "has_confounding_path",
    "confounding_path_closure",
    "is_compatible",
    "greedy_fas",
    "greedy_te",
    "greedy_cpc",
    "heuristic_incompatibility",
    "exact_incompatibility",
    "transitivity_editing_optimum",
    "enumerate_transitively_closed_dags",
]


@dataclass(frozen=True)
class StatementGraph:
    """A mixed graph on ``n`` vertices."""

    n: int
    D: np.ndarray   # (n, n) bool, directed edges
    Bd: np.ndarray  # (n, n) bool, symmetric, zero diagonal

    @staticmethod
    def empty(n: int) -> "StatementGraph":
        return StatementGraph(n, np.zeros((n, n), bool), np.zeros((n, n), bool))

    def n_directed(self) -> int:
        return int(self.D.sum())

    def n_bidirected(self) -> int:
        return int(self.Bd.sum() // 2)

    def density(self) -> float:
        """Fraction of the ``2 * C(n,2) + C(n,2)`` possible edge slots in use.

        Following the paper's Figure 7 we measure density as the number of
        present edges divided by the number of vertex pairs times two (one
        directed slot and one bidirected slot per pair).
        """
        pairs = self.n * (self.n - 1) // 2
        return (self.n_directed() + self.n_bidirected()) / (2 * pairs)


def hamming(g: StatementGraph, h: StatementGraph) -> int:
    """Number of directed + bidirected edge additions/deletions between graphs."""
    return int((g.D != h.D).sum() + (g.Bd != h.Bd).sum() // 2)


# --------------------------------------------------------------------------
# Lemma 3.5 property 2: transitive closure of the directed part
# --------------------------------------------------------------------------

def transitive_closure(D: np.ndarray) -> np.ndarray:
    """Reflexive-free transitive closure of a directed adjacency matrix."""
    n = D.shape[0]
    R = D.copy()
    for k in range(n):
        R |= np.outer(R[:, k], R[k, :])
    np.fill_diagonal(R, False)
    return R


def _is_acyclic(D: np.ndarray) -> bool:
    """True when the directed part has no directed cycle."""
    n = D.shape[0]
    R = D.copy()
    np.fill_diagonal(R, False)
    reach = R.copy()
    for k in range(n):
        reach |= np.outer(reach[:, k], reach[k, :])
    return not bool(np.any(np.diag(reach)))


# --------------------------------------------------------------------------
# Definition 3.1: confounding paths
# --------------------------------------------------------------------------

def has_confounding_path(g: StatementGraph, v: int, w: int) -> bool:
    """True iff a *confounding path* (Definition 3.1) connects ``v`` and ``w``.

    Definition 3.1 reads: "A path between v, w is a confounding path if both v
    and w are adjacent to an arrowhead of the path and no intermediate vertex is
    adjacent to two arrowheads (e.g. ``v <->-> w`` or ``v <-<->-> w``)."

    Disambiguation
    --------------
    Read completely literally, those conditions are also met by a *pure
    common-ancestor* path ``v <- ... <- x -> ... -> w``, which contains no
    bidirected edge at all.  We do **not** adopt that reading, for two reasons:

    1. Both examples the paper gives contain exactly one bidirected edge.  (A
       path cannot contain two: the vertex between them would carry two
       arrowheads and hence be an excluded collider.)
    2. It is empirically decisive.  Appendix D.1's graphical generator builds
       the ground-truth statement graph by marginalising onto every pair, so at
       zero injected errors it must be compatible -- that is the ``x = 0``
       baseline of Figure 5.  Over 300 sampled ground-truth models
       (``n in 3..8``, ``m in 0..3``, ``p in {0.2,0.3,0.5,0.7}``), the reading
       adopted here yields a compatible graph 300/300 times, while the fully
       literal reading yields one only 219/300 (73%) of the time.  See
       ``.openresearch/artifacts/claim6/source_audit.md``.

    So a confounding path here is a simple path with arrowheads at both
    endpoints, no intermediate collider, and at least one bidirected edge.
    Semantically this is the right notion: a pure common-ancestor path is
    already accounted for by the transitively closed directed part (property 2
    of Lemma 3.5) and is *observed* back-door structure, not confounding.

    We search directly over simple paths, tracking the mark (arrowhead or tail)
    that the previous edge leaves on the current vertex.
    """
    if v == w:
        return False
    n = g.n
    D, Bd = g.D, g.Bd

    # Each stack entry: (current vertex, visited set, mark left on current
    # vertex by the edge we arrived on (True = arrowhead), seen a bidirected
    # edge yet).
    stack = []
    # First edge must place an arrowhead on v, so it is either y -> v or y <-> v.
    for y in range(n):
        if y == v:
            continue
        if D[y, v] and y != w:            # y -> v : arrowhead at v, tail at y
            stack.append((y, frozenset((v, y)), False, False))
        if Bd[v, y]:                      # v <-> y : arrowhead at v and at y
            if y == w:
                return True               # a single bidirected edge qualifies
            stack.append((y, frozenset((v, y)), True, True))

    while stack:
        cur, seen, mark_in, saw_bi = stack.pop()
        for nxt in range(n):
            if nxt in seen:
                continue
            # cur -> nxt : tail at cur (always allowed), arrowhead at nxt
            if D[cur, nxt]:
                if nxt == w:
                    if saw_bi:
                        return True
                else:
                    stack.append((nxt, seen | {nxt}, True, saw_bi))
            # nxt -> cur : arrowhead at cur (needs an incoming tail), tail at nxt
            if D[nxt, cur] and not mark_in and nxt != w:
                stack.append((nxt, seen | {nxt}, False, saw_bi))
            # cur <-> nxt : arrowhead at cur (needs incoming tail) and at nxt
            if Bd[cur, nxt] and not mark_in:
                if nxt == w:
                    return True
                stack.append((nxt, seen | {nxt}, True, True))
    return False


def confounding_path_closure(g: StatementGraph) -> StatementGraph:
    """``cpc(G)``: add a bidirected edge for every confounding-path-connected pair."""
    Bd = g.Bd.copy()
    changed = True
    while changed:
        changed = False
        cur = StatementGraph(g.n, g.D, Bd)
        for v in range(g.n):
            for w in range(v + 1, g.n):
                if not Bd[v, w] and has_confounding_path(cur, v, w):
                    Bd[v, w] = Bd[w, v] = True
                    changed = True
                    cur = StatementGraph(g.n, g.D, Bd)
    return StatementGraph(g.n, g.D, Bd)


# --------------------------------------------------------------------------
# Lemma 3.5
# --------------------------------------------------------------------------

def is_compatible(g: StatementGraph) -> bool:
    """The three conditions of Lemma 3.5."""
    if not _is_acyclic(g.D):
        return False
    if not np.array_equal(transitive_closure(g.D), g.D):
        return False
    for v in range(g.n):
        for w in range(v + 1, g.n):
            if not g.Bd[v, w] and has_confounding_path(g, v, w):
                return False
    return True


# --------------------------------------------------------------------------
# Appendix C, Algorithm 1: GreedyFAS (Eades, Lin & Smyth 1993)
# --------------------------------------------------------------------------

def greedy_fas(D: np.ndarray) -> list[tuple[int, int]]:
    """Return ``E_cycles``: edges inconsistent with the greedy vertex ordering."""
    n = D.shape[0]
    alive = np.ones(n, bool)
    H = D.copy()
    L: list[int] = []
    R: list[int] = []
    while alive.any():
        moved = True
        while moved:
            moved = False
            # sources first
            for v in range(n):
                if alive[v] and not H[:, v][alive].any():
                    alive[v] = False
                    H[v, :] = False
                    H[:, v] = False
                    L.append(v)
                    moved = True
            # then sinks
            for v in range(n):
                if alive[v] and not H[v, :][alive].any():
                    alive[v] = False
                    H[v, :] = False
                    H[:, v] = False
                    R.insert(0, v)
                    moved = True
        if not alive.any():
            break
        # otherwise: greedily take the vertex maximising outdeg - indeg
        best, best_score = -1, None
        for v in range(n):
            if not alive[v]:
                continue
            score = int(H[v, :][alive].sum()) - int(H[:, v][alive].sum())
            if best_score is None or score > best_score:
                best, best_score = v, score
        alive[best] = False
        H[best, :] = False
        H[:, best] = False
        L.append(best)

    order = L + R
    pos = {v: k for k, v in enumerate(order)}
    return [(u, v) for u in range(n) for v in range(n)
            if D[u, v] and pos[u] > pos[v]]


# --------------------------------------------------------------------------
# Appendix C, Algorithm 2: GreedyTE
# --------------------------------------------------------------------------

def greedy_te(D: np.ndarray):
    """Greedy Transitivity Editing.  Returns ``(E_del, E_add)``."""
    H = D.copy()
    E_del: list[tuple[int, int]] = []
    n = D.shape[0]
    while True:
        base = int(transitive_closure(H).sum())
        best_gain, best_edge = 0, None
        for u in range(n):
            for v in range(n):
                if not H[u, v]:
                    continue
                H[u, v] = False
                gain = base - int(transitive_closure(H).sum()) - 1
                H[u, v] = True
                if gain > best_gain:
                    best_gain, best_edge = gain, (u, v)
        if best_gain > 0 and best_edge is not None:
            E_del.append(best_edge)
            H[best_edge[0], best_edge[1]] = False
        else:
            break
    Dp = D.copy()
    for (u, v) in E_del:
        Dp[u, v] = False
    tc = transitive_closure(Dp)
    E_add = [(u, v) for u in range(n) for v in range(n) if tc[u, v] and not Dp[u, v]]
    return E_del, E_add


# --------------------------------------------------------------------------
# Appendix C, Algorithm 3: GreedyCPC
# --------------------------------------------------------------------------

def greedy_cpc(g: StatementGraph):
    """Greedy Confounding Path Closure.  Returns ``(B_del, B_add)``."""
    n = g.n
    Bd = g.Bd.copy()
    B_del: list[tuple[int, int]] = []
    while True:
        base = int(confounding_path_closure(StatementGraph(n, g.D, Bd)).Bd.sum())
        best_gain, best_edge = 0, None
        for u in range(n):
            for v in range(u + 1, n):
                if not Bd[u, v]:
                    continue
                Bd[u, v] = Bd[v, u] = False
                closed = int(confounding_path_closure(StatementGraph(n, g.D, Bd)).Bd.sum())
                Bd[u, v] = Bd[v, u] = True
                gain = (base - closed) // 2 - 1
                if gain > best_gain:
                    best_gain, best_edge = gain, (u, v)
        if best_gain > 0 and best_edge is not None:
            B_del.append(best_edge)
            Bd[best_edge[0], best_edge[1]] = Bd[best_edge[1], best_edge[0]] = False
        else:
            break
    closed = confounding_path_closure(StatementGraph(n, g.D, Bd))
    B_add = [(u, v) for u in range(n) for v in range(u + 1, n)
             if closed.Bd[u, v] and not Bd[u, v]]
    return B_del, B_add


def heuristic_incompatibility(g: StatementGraph, detail: bool = False):
    """``c(G)`` of equation (5): GreedyFAS then GreedyTE then GreedyCPC."""
    E_cycles = greedy_fas(g.D)
    Dp = g.D.copy()
    for (u, v) in E_cycles:
        Dp[u, v] = False
    E_del, E_add = greedy_te(Dp)
    D2 = Dp.copy()
    for (u, v) in E_del:
        D2[u, v] = False
    for (u, v) in E_add:
        D2[u, v] = True
    B_del, B_add = greedy_cpc(StatementGraph(g.n, D2, g.Bd))
    score = len(E_cycles) + len(E_add) + len(E_del) + len(B_add) + len(B_del)
    if detail:
        return score, dict(fas_del=len(E_cycles), te_del=len(E_del), te_add=len(E_add),
                           cpc_del=len(B_del), cpc_add=len(B_add))
    return score


# --------------------------------------------------------------------------
# Definition 3.6: exact incompatibility score by exhaustive search
# --------------------------------------------------------------------------

def enumerate_transitively_closed_dags(n: int):
    """Yield every transitively closed DAG on ``n`` labelled vertices.

    Equivalently: every strict partial order on ``{0, ..., n-1}``.  Generated by
    enumerating linear extensions is error-prone, so we enumerate directed
    graphs over the ``n(n-1)`` ordered pairs and filter -- exact and obviously
    correct, at the cost of being limited to small ``n``.
    """
    slots = [(u, v) for u in range(n) for v in range(n) if u != v]
    for mask in range(1 << len(slots)):
        D = np.zeros((n, n), bool)
        m = mask
        k = 0
        while m:
            if m & 1:
                D[slots[k]] = True
            m >>= 1
            k += 1
        if not _is_acyclic(D):
            continue
        if not np.array_equal(transitive_closure(D), D):
            continue
        yield D


def transitivity_editing_optimum(D: np.ndarray, dags=None) -> int:
    """Optimum of ACYCLIC TRANSITIVITY EDITING: min edits to a transitively
    closed DAG (Weller et al. 2012)."""
    n = D.shape[0]
    if dags is None:
        dags = list(enumerate_transitively_closed_dags(n))
    return min(int((D != Ds).sum()) for Ds in dags)


def exact_incompatibility(g: StatementGraph, dags=None) -> int:
    """``incomp(G)`` of Definition 3.6, by exhaustive minimisation.

    For each candidate transitively closed DAG ``D*`` we only need to consider
    bidirected parts of the form ``cpc(D*, T)`` for ``T`` a subset of the
    original bidirected edges: an optimal ``B*`` never contains a bidirected
    edge that is neither present in ``G`` nor forced by the closure, since such
    an edge only adds cost.
    """
    n = g.n
    if dags is None:
        dags = list(enumerate_transitively_closed_dags(n))
    present = [(u, v) for u in range(n) for v in range(u + 1, n) if g.Bd[u, v]]
    best = None
    for Ds in dags:
        d_dir = int((g.D != Ds).sum())
        if best is not None and d_dir >= best:
            continue
        for r in range(len(present) + 1):
            for keep in itertools.combinations(present, r):
                Bd = np.zeros((n, n), bool)
                for (u, v) in keep:
                    Bd[u, v] = Bd[v, u] = True
                closed = confounding_path_closure(StatementGraph(n, Ds, Bd))
                d_bi = int((g.Bd != closed.Bd).sum()) // 2
                tot = d_dir + d_bi
                if best is None or tot < best:
                    best = tot
    return int(best)
