"""Network topology model for NetTwin.

Pure-Python domain model (no Streamlit dependency). A ``Topology`` holds nodes
and undirected links, validates every change, and can export itself to a
NetworkX graph or a plain dictionary (for JSON / SQLite storage).

Simplifications (documented limitations):
* Links are undirected and full-duplex is NOT modelled: one bandwidth value is
  shared by both directions.
* At most one link exists between any two nodes.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import networkx as nx

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_NAME_LENGTH = 64
MAX_LATENCY_MS = 100_000.0
MAX_BANDWIDTH_MBPS = 1_000_000.0
MAX_NODES = 500
MAX_LINKS = 5_000


class TopologyError(ValueError):
    """Raised when a topology operation or imported data is invalid."""


class NodeType(str, Enum):
    """Supported node categories."""

    HOST = "Host"
    SWITCH = "Switch"
    ROUTER = "Router"
    SERVER = "Server"

    @classmethod
    def parse(cls, value: Any) -> "NodeType":
        """Convert a string (case-insensitive) or NodeType into a NodeType."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            wanted = value.strip().lower()
            for member in cls:
                if member.value.lower() == wanted:
                    return member
        allowed = ", ".join(m.value for m in cls)
        raise TopologyError(f"Invalid node type {value!r}. Allowed: {allowed}.")


def validate_node_id(node_id: Any) -> str:
    """Return a valid node ID or raise TopologyError."""
    if not isinstance(node_id, str) or not ID_PATTERN.match(node_id):
        raise TopologyError(
            "Node ID must be 1-32 characters: letters, digits, '_' or '-'."
        )
    return node_id


def validate_name(name: Any) -> str:
    """Return a stripped display name or raise TopologyError."""
    if not isinstance(name, str) or not name.strip():
        raise TopologyError("Name must be a non-empty string.")
    name = name.strip()
    if len(name) > MAX_NAME_LENGTH:
        raise TopologyError(f"Name must be at most {MAX_NAME_LENGTH} characters.")
    return name


def _check_positive(value: Any, field: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TopologyError(f"{field} must be a number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise TopologyError(f"{field} must be a finite number greater than 0.")
    if number > maximum:
        raise TopologyError(f"{field} must not exceed {maximum:g}.")
    return number


def validate_latency(value: Any) -> float:
    """Validate a link latency in milliseconds."""
    return _check_positive(value, "Latency (ms)", MAX_LATENCY_MS)


def validate_bandwidth(value: Any) -> float:
    """Validate a link bandwidth in Mbps."""
    return _check_positive(value, "Bandwidth (Mbps)", MAX_BANDWIDTH_MBPS)


def _check_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise TopologyError(f"{field} must be true or false.")
    return value


def make_link_id(a: str, b: str) -> str:
    """Canonical, order-independent link identifier."""
    first, second = sorted((a, b))
    return f"{first}<->{second}"


@dataclass
class Node:
    """A network device."""

    node_id: str
    name: str
    node_type: NodeType
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "name": self.name,
            "type": self.node_type.value,
            "active": self.active,
        }


@dataclass
class Link:
    """An undirected link between two nodes."""

    source: str
    target: str
    latency_ms: float
    bandwidth_mbps: float
    active: bool = True

    @property
    def link_id(self) -> str:
        return make_link_id(self.source, self.target)

    def other_end(self, node_id: str) -> str:
        """Return the endpoint that is not ``node_id``."""
        if node_id == self.source:
            return self.target
        if node_id == self.target:
            return self.source
        raise TopologyError(f"{node_id!r} is not an endpoint of {self.link_id}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "latency_ms": self.latency_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
            "active": self.active,
        }


class Topology:
    """A validated collection of nodes and links."""

    def __init__(self, name: str = "Untitled topology") -> None:
        self.name = validate_name(name)
        self._nodes: dict[str, Node] = {}
        self._links: dict[str, Link] = {}

    # ------------------------------------------------------------ queries
    @property
    def nodes(self) -> list[Node]:
        return [self._nodes[k] for k in sorted(self._nodes)]

    @property
    def links(self) -> list[Link]:
        return [self._links[k] for k in sorted(self._links)]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise TopologyError(f"Node {node_id!r} does not exist.") from None

    def has_link(self, a: str, b: str) -> bool:
        return make_link_id(a, b) in self._links

    def get_link(self, a: str, b: str) -> Link:
        try:
            return self._links[make_link_id(a, b)]
        except KeyError:
            raise TopologyError(f"Link {a!r}-{b!r} does not exist.") from None

    def links_of(self, node_id: str) -> list[Link]:
        return [l for l in self.links if node_id in (l.source, l.target)]

    def is_link_operational(self, link: Link) -> bool:
        """A link works only if it and both endpoints are active."""
        return (
            link.active
            and self._nodes[link.source].active
            and self._nodes[link.target].active
        )

    def failed_nodes(self) -> list[str]:
        return [n.node_id for n in self.nodes if not n.active]

    def failed_links(self) -> list[str]:
        """Links marked failed administratively (not due to a failed endpoint)."""
        return [l.link_id for l in self.links if not l.active]

    # ---------------------------------------------------------- node ops
    def add_node(
        self, node_id: str, name: str, node_type: Any, active: bool = True
    ) -> Node:
        validate_node_id(node_id)
        if node_id in self._nodes:
            raise TopologyError(f"Node ID {node_id!r} already exists.")
        if len(self._nodes) >= MAX_NODES:
            raise TopologyError(f"A topology may hold at most {MAX_NODES} nodes.")
        node = Node(
            node_id,
            validate_name(name),
            NodeType.parse(node_type),
            _check_bool(active, "Node status"),
        )
        self._nodes[node_id] = node
        return node

    def update_node(
        self, node_id: str, name: str | None = None, node_type: Any = None
    ) -> Node:
        node = self.get_node(node_id)
        new_name = validate_name(name) if name is not None else node.name
        new_type = NodeType.parse(node_type) if node_type is not None else node.node_type
        node.name, node.node_type = new_name, new_type
        return node

    def remove_node(self, node_id: str) -> list[str]:
        """Delete a node and its links. Returns the removed link IDs."""
        self.get_node(node_id)
        removed = [l.link_id for l in self.links_of(node_id)]
        for link_id in removed:
            del self._links[link_id]
        del self._nodes[node_id]
        return removed

    def set_node_active(self, node_id: str, active: bool) -> None:
        self.get_node(node_id).active = _check_bool(active, "Node status")

    # ---------------------------------------------------------- link ops
    def add_link(
        self,
        source: str,
        target: str,
        latency_ms: Any,
        bandwidth_mbps: Any,
        active: bool = True,
    ) -> Link:
        if source not in self._nodes or target not in self._nodes:
            raise TopologyError("Both link endpoints must be existing nodes.")
        if source == target:
            raise TopologyError("A link cannot connect a node to itself.")
        if self.has_link(source, target):
            raise TopologyError(f"A link between {source!r} and {target!r} already exists.")
        if len(self._links) >= MAX_LINKS:
            raise TopologyError(f"A topology may hold at most {MAX_LINKS} links.")
        link = Link(
            source,
            target,
            validate_latency(latency_ms),
            validate_bandwidth(bandwidth_mbps),
            _check_bool(active, "Link status"),
        )
        self._links[link.link_id] = link
        return link

    def update_link(
        self,
        a: str,
        b: str,
        latency_ms: Any = None,
        bandwidth_mbps: Any = None,
    ) -> Link:
        link = self.get_link(a, b)
        new_lat = validate_latency(latency_ms) if latency_ms is not None else link.latency_ms
        new_bw = (
            validate_bandwidth(bandwidth_mbps)
            if bandwidth_mbps is not None
            else link.bandwidth_mbps
        )
        link.latency_ms, link.bandwidth_mbps = new_lat, new_bw
        return link

    def remove_link(self, a: str, b: str) -> None:
        link = self.get_link(a, b)
        del self._links[link.link_id]

    def set_link_active(self, a: str, b: str, active: bool) -> None:
        self.get_link(a, b).active = _check_bool(active, "Link status")

    # ------------------------------------------------------ derived views
    def active_adjacency(self) -> dict[str, list[tuple[str, Link]]]:
        """Adjacency of operational nodes/links, sorted for deterministic routing."""
        adjacency: dict[str, list[tuple[str, Link]]] = {
            n.node_id: [] for n in self.nodes if n.active
        }
        for link in self.links:
            if self.is_link_operational(link):
                adjacency[link.source].append((link.target, link))
                adjacency[link.target].append((link.source, link))
        for neighbours in adjacency.values():
            neighbours.sort(key=lambda item: item[0])
        return adjacency

    def to_graph(self, active_only: bool = True) -> nx.Graph:
        """Export to a NetworkX graph (operational parts only by default)."""
        graph = nx.Graph(name=self.name)
        for node in self.nodes:
            if active_only and not node.active:
                continue
            graph.add_node(
                node.node_id,
                name=node.name,
                node_type=node.node_type.value,
                active=node.active,
            )
        for link in self.links:
            if active_only and not self.is_link_operational(link):
                continue
            graph.add_edge(
                link.source,
                link.target,
                link_id=link.link_id,
                latency_ms=link.latency_ms,
                bandwidth_mbps=link.bandwidth_mbps,
                active=link.active,
            )
        return graph

    def copy(self) -> "Topology":
        """Independent deep copy (used to preserve baselines)."""
        return copy.deepcopy(self)

    # -------------------------------------------------------- serialising
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [l.to_dict() for l in self.links],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Topology":
        """Build a topology from untrusted data (e.g. parsed JSON).

        Only plain data is read; nothing is executed. Raises TopologyError on
        any structural or value problem.
        """
        if not isinstance(data, Mapping):
            raise TopologyError("Topology data must be a JSON object.")
        nodes, links = data.get("nodes"), data.get("links")
        if not isinstance(nodes, list) or not isinstance(links, list):
            raise TopologyError("Topology data needs 'nodes' and 'links' lists.")
        if len(nodes) > MAX_NODES or len(links) > MAX_LINKS:
            raise TopologyError("Topology exceeds the maximum allowed size.")
        topo = cls(data.get("name", "Imported topology"))
        for entry in nodes:
            if not isinstance(entry, Mapping):
                raise TopologyError("Each node must be an object.")
            missing = {"id", "name", "type"} - set(entry)
            if missing:
                raise TopologyError(f"Node is missing fields: {sorted(missing)}.")
            topo.add_node(entry["id"], entry["name"], entry["type"], entry.get("active", True))
        for entry in links:
            if not isinstance(entry, Mapping):
                raise TopologyError("Each link must be an object.")
            missing = {"source", "target", "latency_ms", "bandwidth_mbps"} - set(entry)
            if missing:
                raise TopologyError(f"Link is missing fields: {sorted(missing)}.")
            topo.add_link(
                entry["source"],
                entry["target"],
                entry["latency_ms"],
                entry["bandwidth_mbps"],
                entry.get("active", True),
            )
        return topo


def sample_topology() -> Topology:
    """A 12-node campus-style network with several redundant paths."""
    topo = Topology("Sample Campus Network")
    for node_id, name, kind in [
        ("H1", "Lab PC 1", "Host"),
        ("H2", "Lab PC 2", "Host"),
        ("H3", "Office PC 1", "Host"),
        ("H4", "Office PC 2", "Host"),
        ("SW1", "Lab Switch", "Switch"),
        ("SW2", "Office Switch", "Switch"),
        ("R1", "Core Router 1", "Router"),
        ("R2", "Core Router 2", "Router"),
        ("R3", "Edge Router", "Router"),
        ("S1", "Web Server", "Server"),
        ("S2", "Database Server", "Server"),
    ]:
        topo.add_node(node_id, name, kind)
    for a, b, latency, bandwidth in [
        ("H1", "SW1", 1, 100),
        ("H2", "SW1", 1, 100),
        ("SW1", "R1", 2, 1000),
        ("SW1", "R2", 4, 1000),
        ("R1", "R2", 3, 1000),
        ("R1", "R3", 5, 500),
        ("R2", "R3", 2, 500),
        ("R2", "SW2", 2, 1000),
        ("R3", "SW2", 3, 1000),
        ("SW2", "H3", 1, 100),
        ("SW2", "H4", 1, 100),
        ("R1", "S1", 4, 200),
        ("R3", "S1", 1, 1000),
        ("R2", "S2", 2, 1000),
    ]:
        topo.add_link(a, b, latency, bandwidth)
    return topo
