"""Routing algorithms for NetTwin.

All strategies run Dijkstra's algorithm over *operational* nodes and links
only (failed nodes/links are never used) with different non-negative link costs:

1. Latency routing:   cost(link) = latency_ms
   -> provably minimises total configured latency.

2. Capacity-aware:    cost(link) = latency_ms / L_max  +  w * u^2
   where L_max = largest latency among operational links (unit normalisation),
   u = (existing_load + demand) / bandwidth   (projected utilisation, >1 means
   the link would be overloaded) and w = ``congestion_weight`` (default 1).
   Costs are additive and non-negative, so Dijkstra finds the minimum-cost
   path *for this formula*. The formula is a heuristic trade-off between
   latency and congestion; the result is NOT claimed to be globally optimal
   for any multi-objective problem. Links with bandwidth <= 0 are excluded.

3. Alternative paths: k loop-free paths in increasing order of total latency
   (Yen's algorithm via NetworkX), useful after failures.

Ties are broken deterministically (fewer hops, then lexicographic node order).
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import networkx as nx

from topology import Link, Topology

STRATEGY_LATENCY = "Shortest path (latency)"
STRATEGY_CAPACITY = "Capacity-aware"
STRATEGY_ALTERNATIVE = "Alternative path"


@dataclass(frozen=True)
class RouteResult:
    """Outcome of a routing request."""

    strategy: str
    source: str
    destination: str
    found: bool
    path: tuple[str, ...] = ()
    link_ids: tuple[str, ...] = ()
    hop_count: int = 0
    latency_ms: float = 0.0
    bottleneck_mbps: float | None = None
    cost: float | None = None
    # link_id -> projected utilisation fraction (existing load + demand) / capacity
    link_utilization: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    message: str = ""


def _utilization(load: float, bandwidth: float) -> float:
    """Safe utilisation fraction (no division by zero)."""
    if bandwidth <= 0:
        return 0.0 if load <= 0 else math.inf
    return load / bandwidth


def _check_numbers(demand_mbps: float, congestion_weight: float) -> None:
    for label, value in (("demand_mbps", demand_mbps), ("congestion_weight", congestion_weight)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a number.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be a finite number >= 0.")


def _endpoint_problem(topo: Topology, source: str, destination: str) -> str | None:
    for node_id in (source, destination):
        if not topo.has_node(node_id):
            return f"Node {node_id!r} does not exist."
    if source == destination:
        return "Source and destination must be different nodes."
    for node_id in (source, destination):
        if not topo.get_node(node_id).active:
            return f"Node {node_id!r} is failed, so no route can start or end there."
    return None


def _dijkstra(
    topo: Topology,
    source: str,
    destination: str,
    cost_fn: Callable[[Link], float | None],
) -> tuple[tuple[str, ...], float] | None:
    """Deterministic Dijkstra. ``cost_fn`` returns None to exclude a link."""
    adjacency = topo.active_adjacency()
    heap: list[tuple[float, int, tuple[str, ...]]] = [(0.0, 0, (source,))]
    settled: set[str] = set()
    while heap:
        cost, hops, path = heapq.heappop(heap)
        node = path[-1]
        if node in settled:
            continue
        settled.add(node)
        if node == destination:
            return path, cost
        for neighbour, link in adjacency.get(node, []):
            if neighbour in settled:
                continue
            step = cost_fn(link)
            if step is None or not math.isfinite(step):
                continue
            heapq.heappush(heap, (cost + step, hops + 1, path + (neighbour,)))
    return None


def _no_path(strategy: str, source: str, destination: str, message: str) -> RouteResult:
    return RouteResult(strategy, source, destination, False, message=message)


def _build_result(
    topo: Topology,
    strategy: str,
    path: tuple[str, ...],
    cost: float | None,
    demand_mbps: float,
    loads: Mapping[str, float],
    reasons: list[str],
) -> RouteResult:
    links = [topo.get_link(a, b) for a, b in zip(path, path[1:])]
    utilization = {
        l.link_id: _utilization(loads.get(l.link_id, 0.0) + demand_mbps, l.bandwidth_mbps)
        for l in links
    }
    over = [lid for lid, u in utilization.items() if u > 1.0]
    if over:
        reasons.append(
            "Warning: projected load exceeds capacity on " + ", ".join(over) + "."
        )
    return RouteResult(
        strategy=strategy,
        source=path[0],
        destination=path[-1],
        found=True,
        path=path,
        link_ids=tuple(l.link_id for l in links),
        hop_count=len(links),
        latency_ms=sum(l.latency_ms for l in links),
        bottleneck_mbps=min(l.bandwidth_mbps for l in links),
        cost=cost,
        link_utilization=utilization,
        reasons=tuple(reasons),
        message="Route found.",
    )


def shortest_path(
    topo: Topology,
    source: str,
    destination: str,
    demand_mbps: float = 0.0,
    link_loads: Mapping[str, float] | None = None,
) -> RouteResult:
    """Minimum total-latency route over operational nodes and links."""
    _check_numbers(demand_mbps, 0.0)
    problem = _endpoint_problem(topo, source, destination)
    if problem:
        return _no_path(STRATEGY_LATENCY, source, destination, problem)
    found = _dijkstra(topo, source, destination, lambda l: l.latency_ms)
    if found is None:
        return _no_path(
            STRATEGY_LATENCY,
            source,
            destination,
            f"No valid path exists between {source} and {destination} "
            "using operational nodes and links.",
        )
    path, cost = found
    reasons = [
        f"Lowest total configured latency ({cost:.2f} ms) among all paths that use "
        "only operational nodes and links.",
        "Link bandwidth and current load were not considered by this strategy.",
    ]
    return _build_result(
        topo, STRATEGY_LATENCY, path, cost, demand_mbps, link_loads or {}, reasons
    )


def capacity_aware_path(
    topo: Topology,
    source: str,
    destination: str,
    demand_mbps: float = 0.0,
    link_loads: Mapping[str, float] | None = None,
    congestion_weight: float = 1.0,
) -> RouteResult:
    """Route minimising  sum(latency/L_max + w * u^2)  over operational links."""
    _check_numbers(demand_mbps, congestion_weight)
    loads = link_loads or {}
    problem = _endpoint_problem(topo, source, destination)
    if problem:
        return _no_path(STRATEGY_CAPACITY, source, destination, problem)

    operational = [l for l in topo.links if topo.is_link_operational(l)]
    l_max = max((l.latency_ms for l in operational), default=1.0)

    def cost_fn(link: Link) -> float | None:
        if link.bandwidth_mbps <= 0:
            return None
        u = (loads.get(link.link_id, 0.0) + demand_mbps) / link.bandwidth_mbps
        return link.latency_ms / l_max + congestion_weight * u * u

    found = _dijkstra(topo, source, destination, cost_fn)
    if found is None:
        return _no_path(
            STRATEGY_CAPACITY,
            source,
            destination,
            f"No valid path exists between {source} and {destination} "
            "using operational nodes and links with usable capacity.",
        )
    path, cost = found
    reasons = [
        f"Minimum combined cost {cost:.3f} = sum of latency/{l_max:g} ms "
        f"+ {congestion_weight:g} x (projected utilisation)^2 over the path's links.",
    ]
    baseline = shortest_path(topo, source, destination, demand_mbps, loads)
    if baseline.found:
        if baseline.path == path:
            reasons.append("Same route as latency-only routing: no better trade-off was found.")
        else:
            chosen_links = [topo.get_link(a, b) for a, b in zip(path, path[1:])]
            chosen_peak = max(
                _utilization(loads.get(l.link_id, 0.0) + demand_mbps, l.bandwidth_mbps)
                for l in chosen_links
            )
            base_peak = max(baseline.link_utilization.values())
            latency = sum(l.latency_ms for l in chosen_links)
            reasons.append(
                f"Differs from the latency-only route ({' -> '.join(baseline.path)}, "
                f"{baseline.latency_ms:.2f} ms): this route has {latency:.2f} ms latency "
                f"but peak projected utilisation {chosen_peak:.0%} versus {base_peak:.0%}."
            )
    return _build_result(topo, STRATEGY_CAPACITY, path, cost, demand_mbps, loads, reasons)


def alternative_paths(
    topo: Topology,
    source: str,
    destination: str,
    k: int = 3,
    demand_mbps: float = 0.0,
    link_loads: Mapping[str, float] | None = None,
) -> list[RouteResult]:
    """Up to ``k`` loop-free paths ordered by increasing total latency."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    _check_numbers(demand_mbps, 0.0)
    if _endpoint_problem(topo, source, destination):
        return []
    graph = topo.to_graph(active_only=True)
    try:
        generator = nx.shortest_simple_paths(graph, source, destination, weight="latency_ms")
        paths = list(itertools.islice(generator, k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    results = []
    for rank, nodes in enumerate(paths, start=1):
        path = tuple(nodes)
        total = sum(topo.get_link(a, b).latency_ms for a, b in zip(path, path[1:]))
        reasons = [f"Alternative #{rank} by total latency ({total:.2f} ms)."]
        results.append(
            _build_result(
                topo, STRATEGY_ALTERNATIVE, path, total, demand_mbps, link_loads or {}, reasons
            )
        )
    return results
