"""Traffic allocation and failure/recovery scenarios for NetTwin.

MODEL AND ASSUMPTIONS (modelled estimates, NOT packet-level measurements)
* Each demand follows exactly ONE path (no traffic splitting).
* Demands are routed one at a time in a fixed order: higher priority first,
  then larger rate, then demand ID. This makes every run deterministic.
* The *offered* rate of a demand is added to the load of every link on its
  path. Link utilisation = load / capacity (a fraction; reported capped at
  ``UTIL_CAP`` so values stay finite).
* A link whose load exceeds capacity is a *capacity violation*. On such a link
  every flow gets the same share ``capacity / load`` (proportional sharing).
  A demand's delivered rate is ``rate * min(share over its links)``. This is a
  single-bottleneck approximation: traffic dropped at one link is not removed
  from the load seen by other links.
* Route latency is the sum of configured link latencies. Queueing delay under
  load is NOT modelled.
* Demand status: ``served`` (routed, no overloaded link on its path),
  ``congested`` (routed but crosses an overloaded link) or ``disconnected``
  (no valid path over operational nodes/links).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from routing import RouteResult, capacity_aware_path, shortest_path
from topology import Topology, TopologyError, make_link_id

UTIL_CAP = 1000.0
MAX_RATE_MBPS = 1_000_000.0
CAPACITY_EPS = 1e-9
CONGESTION_THRESHOLD = 0.8  # links at or above 80 % utilisation are "congested"

STATUS_SERVED = "served"
STATUS_CONGESTED = "congested"
STATUS_DISCONNECTED = "disconnected"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class SimulationError(ValueError):
    """Raised for invalid simulation input."""


class DemandError(SimulationError):
    """Raised for an invalid traffic demand."""


class Strategy(str, Enum):
    """Routing strategy used by a simulation run."""

    LATENCY = "latency"
    CAPACITY = "capacity"

    @property
    def label(self) -> str:
        return "Shortest path (latency)" if self is Strategy.LATENCY else "Capacity-aware"

    @classmethod
    def parse(cls, value: Any) -> "Strategy":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise SimulationError("Strategy must be 'latency' or 'capacity'.")


@dataclass(frozen=True)
class Demand:
    """Offered traffic from a source node to a destination node."""

    demand_id: str
    source: str
    destination: str
    rate_mbps: float
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.demand_id, str) or not _ID_PATTERN.match(self.demand_id):
            raise DemandError("Demand ID must be 1-32 characters: letters, digits, '_' or '-'.")
        for label, value in (("Source", self.source), ("Destination", self.destination)):
            if not isinstance(value, str) or not value:
                raise DemandError(f"{label} must be a node ID.")
        if self.source == self.destination:
            raise DemandError("Source and destination must be different nodes.")
        rate = self.rate_mbps
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise DemandError("Rate must be a number.")
        if not math.isfinite(rate) or rate <= 0 or rate > MAX_RATE_MBPS:
            raise DemandError(f"Rate must be greater than 0 and at most {MAX_RATE_MBPS:g} Mbps.")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int) \
                or not 0 <= self.priority <= 10:
            raise DemandError("Priority must be an integer from 0 to 10.")
        object.__setattr__(self, "rate_mbps", float(rate))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.demand_id,
            "source": self.source,
            "destination": self.destination,
            "rate_mbps": self.rate_mbps,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Demand":
        if not isinstance(data, Mapping):
            raise DemandError("A demand must be an object.")
        missing = {"id", "source", "destination", "rate_mbps"} - set(data)
        if missing:
            raise DemandError(f"Demand is missing fields: {sorted(missing)}.")
        return cls(
            data["id"], data["source"], data["destination"],
            data["rate_mbps"], data.get("priority", 0),
        )


@dataclass(frozen=True)
class LinkState:
    """Load on one link after allocation."""

    link_id: str
    operational: bool
    capacity_mbps: float
    latency_ms: float
    load_mbps: float
    utilization: float  # fraction, capped at UTIL_CAP

    @property
    def utilization_pct(self) -> float:
        return self.utilization * 100.0

    @property
    def excess_mbps(self) -> float:
        return max(0.0, self.load_mbps - self.capacity_mbps)

    @property
    def is_violation(self) -> bool:
        return self.load_mbps > self.capacity_mbps + CAPACITY_EPS


@dataclass(frozen=True)
class DemandResult:
    """Outcome for one demand."""

    demand: Demand
    status: str
    route: RouteResult
    delivered_mbps: float
    latency_ms: float | None
    message: str


@dataclass(frozen=True)
class SimulationResult:
    """Complete result of one traffic simulation run."""

    strategy: Strategy
    demand_results: tuple[DemandResult, ...]  # same order as the input demands
    link_states: dict[str, LinkState]

    @property
    def capacity_violations(self) -> list[LinkState]:
        return [s for s in self.link_states.values() if s.is_violation]

    def congested_links(self, threshold: float = CONGESTION_THRESHOLD) -> list[LinkState]:
        return [
            s for s in self.link_states.values()
            if s.operational and s.load_mbps > 0 and s.utilization >= threshold
        ]

    def result_for(self, demand_id: str) -> DemandResult:
        for item in self.demand_results:
            if item.demand.demand_id == demand_id:
                return item
        raise SimulationError(f"No result for demand {demand_id!r}.")


def _utilization(load: float, capacity: float) -> float:
    if load <= 0:
        return 0.0
    if capacity <= 0:
        return UTIL_CAP
    return min(load / capacity, UTIL_CAP)


def _share(load: float, capacity: float) -> float:
    """Fraction of offered traffic a link can carry."""
    if load <= capacity + CAPACITY_EPS:
        return 1.0
    return capacity / load if capacity > 0 else 0.0


def run_simulation(
    topo: Topology,
    demands: Iterable[Demand],
    strategy: Strategy | str = Strategy.CAPACITY,
) -> SimulationResult:
    """Allocate demands over ``topo``. Never mutates the topology."""
    strategy = Strategy.parse(strategy)
    items = list(demands)
    ids = [d.demand_id for d in items]
    if len(set(ids)) != len(ids):
        raise SimulationError("Demand IDs must be unique.")

    order = sorted(
        range(len(items)),
        key=lambda i: (-items[i].priority, -items[i].rate_mbps, items[i].demand_id),
    )
    loads = {link.link_id: 0.0 for link in topo.links}
    routes: dict[int, RouteResult] = {}
    for index in order:
        d = items[index]
        if strategy is Strategy.CAPACITY:
            route = capacity_aware_path(topo, d.source, d.destination, d.rate_mbps, loads)
        else:
            route = shortest_path(topo, d.source, d.destination, d.rate_mbps, loads)
        routes[index] = route
        if route.found:
            for link_id in route.link_ids:
                loads[link_id] += d.rate_mbps

    link_states: dict[str, LinkState] = {}
    shares: dict[str, float] = {}
    for link in topo.links:
        load = loads[link.link_id]
        link_states[link.link_id] = LinkState(
            link.link_id,
            topo.is_link_operational(link),
            link.bandwidth_mbps,
            link.latency_ms,
            load,
            _utilization(load, link.bandwidth_mbps),
        )
        shares[link.link_id] = _share(load, link.bandwidth_mbps)

    results: list[DemandResult] = []
    for index, d in enumerate(items):
        route = routes[index]
        if not route.found:
            results.append(DemandResult(d, STATUS_DISCONNECTED, route, 0.0, None, route.message))
            continue
        factor = min(shares[lid] for lid in route.link_ids)
        delivered = d.rate_mbps * factor
        if factor < 1.0:
            status = STATUS_CONGESTED
            message = f"Routed, but delivered {delivered:.2f} of {d.rate_mbps:.2f} Mbps (overloaded link on path)."
        else:
            status = STATUS_SERVED
            message = "Fully served."
        results.append(DemandResult(d, status, route, delivered, route.latency_ms, message))
    return SimulationResult(strategy, tuple(results), link_states)


# ----------------------------------------------------------------- scenarios
@dataclass(frozen=True)
class FailureEvent:
    """A controlled failure or recovery of one node or link."""

    kind: str  # "node" or "link"
    target: str  # node ID, or link ID "A<->B"
    action: str  # "fail" or "recover"

    def __post_init__(self) -> None:
        if self.kind not in ("node", "link"):
            raise SimulationError("Event kind must be 'node' or 'link'.")
        if self.action not in ("fail", "recover"):
            raise SimulationError("Event action must be 'fail' or 'recover'.")
        if not isinstance(self.target, str) or not self.target:
            raise SimulationError("Event target must be a non-empty string.")
        if self.kind == "link" and len(self.target.split("<->")) != 2:
            raise SimulationError("Link targets look like 'A<->B'.")

    @classmethod
    def for_link(cls, a: str, b: str, action: str) -> "FailureEvent":
        return cls("link", make_link_id(a, b), action)

    @property
    def inverse(self) -> "FailureEvent":
        return FailureEvent(self.kind, self.target, "recover" if self.action == "fail" else "fail")

    def describe(self) -> str:
        verb = "Fail" if self.action == "fail" else "Recover"
        return f"{verb} {self.kind} {self.target}"

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "target": self.target, "action": self.action}

    @classmethod
    def from_dict(cls, data: Any) -> "FailureEvent":
        if not isinstance(data, Mapping) or not {"kind", "target", "action"} <= set(data):
            raise SimulationError("An event needs 'kind', 'target' and 'action'.")
        return cls(data["kind"], data["target"], data["action"])


def apply_event(topo: Topology, event: FailureEvent) -> None:
    """Apply one event in place. Redundant events (already failed etc.) raise."""
    want_active = event.action == "recover"
    try:
        if event.kind == "node":
            if topo.get_node(event.target).active == want_active:
                raise SimulationError(
                    f"Node {event.target!r} is already {'active' if want_active else 'failed'}."
                )
            topo.set_node_active(event.target, want_active)
        else:
            a, b = event.target.split("<->")
            if topo.get_link(a, b).active == want_active:
                raise SimulationError(
                    f"Link {event.target!r} is already {'active' if want_active else 'failed'}."
                )
            topo.set_link_active(a, b, want_active)
    except TopologyError as exc:
        raise SimulationError(str(exc)) from exc


def apply_events(baseline: Topology, events: Sequence[FailureEvent]) -> Topology:
    """Return a COPY of ``baseline`` with ``events`` applied in order."""
    scenario = baseline.copy()
    for event in events:
        apply_event(scenario, event)
    return scenario


def invert_events(events: Sequence[FailureEvent]) -> tuple[FailureEvent, ...]:
    """Events that undo ``events`` (reverse order, opposite action)."""
    return tuple(e.inverse for e in reversed(events))


@dataclass(frozen=True)
class ScenarioRun:
    """Baseline and scenario results side by side."""

    events: tuple[FailureEvent, ...]
    baseline_topology: Topology
    scenario_topology: Topology
    baseline_result: SimulationResult
    scenario_result: SimulationResult


def run_scenario(
    baseline: Topology,
    demands: Sequence[Demand],
    events: Sequence[FailureEvent],
    strategy: Strategy | str = Strategy.CAPACITY,
) -> ScenarioRun:
    """Simulate the baseline, apply ``events`` to a copy, and simulate again.

    The baseline topology passed in is never modified.
    """
    demands = list(demands)
    baseline_copy = baseline.copy()
    scenario = apply_events(baseline_copy, events)
    return ScenarioRun(
        tuple(events),
        baseline_copy,
        scenario,
        run_simulation(baseline_copy, demands, strategy),
        run_simulation(scenario, demands, strategy),
    )


# --------------------------------------------------------------- experiments
EXPERIMENT_KIND = "nettwin-experiment"
MAX_IMPORT_DEMANDS = 1_000
MAX_IMPORT_EVENTS = 200


@dataclass(frozen=True)
class Experiment:
    """Everything needed to reproduce a run: topology, demands, events, strategy."""

    name: str
    topology: Topology
    demands: tuple[Demand, ...]
    events: tuple[FailureEvent, ...]
    strategy: Strategy

    def run(self) -> ScenarioRun:
        """Execute the experiment (deterministic for a given configuration)."""
        return run_scenario(self.topology, self.demands, self.events, self.strategy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": EXPERIMENT_KIND,
            "name": self.name,
            "strategy": self.strategy.value,
            "topology": self.topology.to_dict(),
            "demands": [d.to_dict() for d in self.demands],
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Experiment":
        """Validate and build an experiment from untrusted plain data."""
        if not isinstance(data, Mapping) or data.get("kind") != EXPERIMENT_KIND:
            raise SimulationError(f"Not a NetTwin experiment (kind must be '{EXPERIMENT_KIND}').")
        demands, events = data.get("demands"), data.get("events")
        if not isinstance(demands, list) or not isinstance(events, list):
            raise SimulationError("An experiment needs 'demands' and 'events' lists.")
        if len(demands) > MAX_IMPORT_DEMANDS or len(events) > MAX_IMPORT_EVENTS:
            raise SimulationError("Experiment exceeds the maximum allowed size.")
        from topology import validate_name  # local import keeps the header tidy

        try:
            topology = Topology.from_dict(data.get("topology"))
            name = validate_name(data.get("name", topology.name))
        except TopologyError as exc:
            raise SimulationError(str(exc)) from exc
        parsed = tuple(Demand.from_dict(d) for d in demands)
        if len({d.demand_id for d in parsed}) != len(parsed):
            raise SimulationError("Demand IDs must be unique.")
        return cls(
            name, topology, parsed,
            tuple(FailureEvent.from_dict(e) for e in events),
            Strategy.parse(data.get("strategy")),
        )
