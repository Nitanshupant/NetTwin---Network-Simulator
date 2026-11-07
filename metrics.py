"""Resilience and performance metrics for NetTwin.

DEFINITIONS
* Node-pair connectivity = reachable pairs / total pairs. Pairs are counted
  over ALL configured nodes, so a failed node lowers connectivity (its pairs
  are unreachable). Reachable pairs come from the connected components of the
  operational graph: sum of C(size, 2). With fewer than 2 nodes there are no
  pairs and the value is defined as 1.0.
* served / congested / disconnected demands: see ``simulation`` module.
* Delivery ratio = delivered Mbps / offered Mbps (1.0 when nothing is offered).
* Average route latency = mean of configured route latency over demands that
  have a route (served + congested); None if there are none. It excludes
  queueing delay.
* Utilisation figures are fractions (1.0 = 100 %). Average utilisation is over
  operational links only.

LIMITATIONS: these are modelled values for a given topology and demand set;
they say nothing about real hardware behaviour and no single "resilience score"
is invented - use the individual metrics and the before/after deltas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from simulation import (
    STATUS_CONGESTED,
    STATUS_DISCONNECTED,
    STATUS_SERVED,
    SimulationResult,
)
from topology import Topology


@dataclass(frozen=True)
class ConnectivityMetrics:
    total_nodes: int
    operational_nodes: int
    failed_nodes: int
    components: int
    largest_component: int
    reachable_pairs: int
    total_pairs: int
    pair_connectivity: float


def connectivity(topo: Topology) -> ConnectivityMetrics:
    """Connectivity of the operational part of ``topo``."""
    graph = topo.to_graph(active_only=True)
    sizes = [len(c) for c in nx.connected_components(graph)]
    total_nodes = len(topo.nodes)
    total_pairs = total_nodes * (total_nodes - 1) // 2
    reachable = sum(s * (s - 1) // 2 for s in sizes)
    return ConnectivityMetrics(
        total_nodes=total_nodes,
        operational_nodes=graph.number_of_nodes(),
        failed_nodes=len(topo.failed_nodes()),
        components=len(sizes),
        largest_component=max(sizes, default=0),
        reachable_pairs=reachable,
        total_pairs=total_pairs,
        pair_connectivity=(reachable / total_pairs) if total_pairs else 1.0,
    )


@dataclass(frozen=True)
class NetworkMetrics:
    strategy: str
    connectivity: ConnectivityMetrics
    total_demands: int
    served_demands: int
    congested_demands: int
    disconnected_demands: int
    reachable_demands: int
    offered_mbps: float
    delivered_mbps: float
    delivery_ratio: float
    avg_latency_ms: float | None
    max_utilization: float
    avg_utilization: float | None
    violation_count: int
    congested_link_count: int
    total_excess_mbps: float

    def to_flat_dict(self) -> dict[str, Any]:
        """Flat dictionary suitable for CSV / SQLite storage."""
        data = {k: v for k, v in self.__dict__.items() if k != "connectivity"}
        data.update(
            {f"conn_{k}": v for k, v in self.connectivity.__dict__.items()}
        )
        return data


def compute_metrics(topo: Topology, result: SimulationResult) -> NetworkMetrics:
    """Summarise a simulation result run on ``topo``."""
    items = result.demand_results
    count = lambda status: sum(1 for r in items if r.status == status)  # noqa: E731
    offered = sum(r.demand.rate_mbps for r in items)
    delivered = sum(r.delivered_mbps for r in items)
    latencies = [r.latency_ms for r in items if r.latency_ms is not None]
    operational = [s for s in result.link_states.values() if s.operational]
    served, congested = count(STATUS_SERVED), count(STATUS_CONGESTED)
    return NetworkMetrics(
        strategy=result.strategy.label,
        connectivity=connectivity(topo),
        total_demands=len(items),
        served_demands=served,
        congested_demands=congested,
        disconnected_demands=count(STATUS_DISCONNECTED),
        reachable_demands=served + congested,
        offered_mbps=offered,
        delivered_mbps=delivered,
        delivery_ratio=(delivered / offered) if offered > 0 else 1.0,
        avg_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
        max_utilization=max((s.utilization for s in operational), default=0.0),
        avg_utilization=(
            sum(s.utilization for s in operational) / len(operational) if operational else None
        ),
        violation_count=len(result.capacity_violations),
        congested_link_count=len(result.congested_links()),
        total_excess_mbps=sum(s.excess_mbps for s in result.capacity_violations),
    )


@dataclass(frozen=True)
class DemandImpact:
    demand_id: str
    change: str  # unchanged | rerouted | congestion_changed | lost | restored
    before_status: str
    after_status: str
    before_path: tuple[str, ...]
    after_path: tuple[str, ...]
    latency_delta_ms: float | None


@dataclass(frozen=True)
class Comparison:
    before: NetworkMetrics
    after: NetworkMetrics
    deltas: dict[str, float | None]  # after - before
    impacts: tuple[DemandImpact, ...]
    newly_failed_nodes: tuple[str, ...]
    newly_failed_links: tuple[str, ...]
    recovered_nodes: tuple[str, ...]
    recovered_links: tuple[str, ...]
    utilization_changes: dict[str, tuple[float, float]]  # link -> (before, after)

    @property
    def affected_demands(self) -> tuple[DemandImpact, ...]:
        return tuple(i for i in self.impacts if i.change != "unchanged")


_DELTA_FIELDS = (
    "served_demands", "congested_demands", "disconnected_demands", "reachable_demands",
    "delivered_mbps", "delivery_ratio", "avg_latency_ms", "max_utilization",
    "avg_utilization", "violation_count", "total_excess_mbps",
)


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _classify(before, after) -> DemandImpact:
    b_routed = before.status != STATUS_DISCONNECTED
    a_routed = after.status != STATUS_DISCONNECTED
    if b_routed and not a_routed:
        change = "lost"
    elif a_routed and not b_routed:
        change = "restored"
    elif not b_routed and not a_routed:
        change = "unchanged"
    elif before.route.path != after.route.path:
        change = "rerouted"
    elif before.status != after.status:
        change = "congestion_changed"
    else:
        change = "unchanged"
    return DemandImpact(
        before.demand.demand_id, change, before.status, after.status,
        before.route.path, after.route.path, _delta(before.latency_ms, after.latency_ms),
    )


def compare_results(
    before_topo: Topology,
    before: SimulationResult,
    after_topo: Topology,
    after: SimulationResult,
) -> Comparison:
    """Compare two runs of the SAME demand set (e.g. baseline vs failure)."""
    before_ids = [r.demand.demand_id for r in before.demand_results]
    after_ids = [r.demand.demand_id for r in after.demand_results]
    if before_ids != after_ids:
        raise ValueError("Both results must contain the same demands in the same order.")
    m_before, m_after = compute_metrics(before_topo, before), compute_metrics(after_topo, after)
    deltas = {name: _delta(getattr(m_before, name), getattr(m_after, name)) for name in _DELTA_FIELDS}
    deltas["pair_connectivity"] = (
        m_after.connectivity.pair_connectivity - m_before.connectivity.pair_connectivity
    )
    impacts = tuple(_classify(b, a) for b, a in zip(before.demand_results, after.demand_results))
    nodes_b, nodes_a = set(before_topo.failed_nodes()), set(after_topo.failed_nodes())
    links_b, links_a = set(before_topo.failed_links()), set(after_topo.failed_links())
    shared = set(before.link_states) & set(after.link_states)
    changes = {
        lid: (before.link_states[lid].utilization, after.link_states[lid].utilization)
        for lid in sorted(shared)
        if abs(before.link_states[lid].utilization - after.link_states[lid].utilization) > 1e-12
    }
    return Comparison(
        m_before, m_after, deltas, impacts,
        tuple(sorted(nodes_a - nodes_b)), tuple(sorted(links_a - links_b)),
        tuple(sorted(nodes_b - nodes_a)), tuple(sorted(links_b - links_a)),
        changes,
    )


# ------------------------------------------------------- resilience analysis
@dataclass(frozen=True)
class SweepRow:
    """Effect of failing ONE component, compared with the unfailed network."""

    kind: str  # "node" | "link"
    target: str
    pair_connectivity: float
    delivery_ratio: float
    delivery_ratio_change: float
    connectivity_change: float
    disconnected_demands: int
    congested_demands: int
    affected_demands: int  # demands whose path or status changed
    violation_count: int


def failure_sweep(topo: Topology, demands, strategy="capacity") -> list[SweepRow]:
    """Fail each operational node and link on its own and measure the impact.

    Components that are already failed are skipped. Rows are sorted from most
    to least harmful: largest delivery-ratio drop, then most disconnected
    demands, then component name (deterministic). A demand's own endpoints are
    included, so failing them naturally disconnects that demand.
    """
    from simulation import FailureEvent, apply_events, run_simulation  # avoid import cycle at load

    demands = list(demands)
    base_result = run_simulation(topo, demands, strategy)
    base = compute_metrics(topo, base_result)
    candidates = [("node", n.node_id) for n in topo.nodes if n.active] + [
        ("link", l.link_id) for l in topo.links if l.active and topo.is_link_operational(l)
    ]
    rows: list[SweepRow] = []
    for kind, target in candidates:
        failed = apply_events(topo, [FailureEvent(kind, target, "fail")])
        result = run_simulation(failed, demands, strategy)
        cmp = compare_results(topo, base_result, failed, result)
        after = cmp.after
        rows.append(SweepRow(
            kind=kind,
            target=target,
            pair_connectivity=after.connectivity.pair_connectivity,
            delivery_ratio=after.delivery_ratio,
            delivery_ratio_change=after.delivery_ratio - base.delivery_ratio,
            connectivity_change=after.connectivity.pair_connectivity - base.connectivity.pair_connectivity,
            disconnected_demands=after.disconnected_demands,
            congested_demands=after.congested_demands,
            affected_demands=len(cmp.affected_demands),
            violation_count=after.violation_count,
        ))
    rows.sort(key=lambda r: (r.delivery_ratio_change, -r.disconnected_demands, r.kind, r.target))
    return rows


def structural_weak_points(topo: Topology) -> tuple[list[str], list[str]]:
    """Single points of failure in the operational graph, independent of traffic.

    Returns (articulation_nodes, bridge_link_ids): removing any listed node or
    link splits a connected part of the network in two.
    """
    graph = topo.to_graph(active_only=True)
    nodes = sorted(nx.articulation_points(graph))
    bridges = sorted(graph[a][b]["link_id"] for a, b in nx.bridges(graph))
    return nodes, bridges
