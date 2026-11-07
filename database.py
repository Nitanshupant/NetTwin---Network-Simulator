"""SQLite persistence for NetTwin (built-in ``sqlite3`` only).

Schema (see ``SCHEMA``):
  topologies 1--* nodes, links          saved, editable topologies
  simulation_runs 1--* failure_events   one row per saved experiment
                  1--* simulation_metrics (phase = baseline | scenario)
                  1--* routing_results    (per demand and phase)
                  1--* link_results       (per link and phase)

Design choices
* Every run stores a JSON *snapshot* of its baseline topology and demands, so
  editing or deleting a saved topology never changes historical runs and any
  run can be reproduced exactly (``load_experiment``).
* A SHA-256 ``config_hash`` of the configuration prevents saving the same
  deterministic experiment twice by accident (``DuplicateRunError``).
* Every multi-step write runs inside one transaction: all or nothing.
* All SQL is parameterised; foreign keys are enforced on every connection.
* A new connection is opened per operation, which is simple and safe with
  Streamlit's script reruns.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from metrics import NetworkMetrics, compute_metrics
from simulation import (
    Demand,
    Experiment,
    FailureEvent,
    ScenarioRun,
    SimulationResult,
    Strategy,
)
from topology import Topology, TopologyError, validate_name

SCHEMA_VERSION = 1
PHASES = ("baseline", "scenario")

SCHEMA = """
CREATE TABLE IF NOT EXISTS topologies (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
    topology_id INTEGER NOT NULL REFERENCES topologies(id) ON DELETE CASCADE,
    node_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    node_type   TEXT NOT NULL CHECK (node_type IN ('Host','Switch','Router','Server')),
    active      INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
    PRIMARY KEY (topology_id, node_id)
);
CREATE TABLE IF NOT EXISTS links (
    topology_id    INTEGER NOT NULL,
    link_id        TEXT NOT NULL,
    source         TEXT NOT NULL,
    target         TEXT NOT NULL,
    latency_ms     REAL NOT NULL CHECK (latency_ms > 0),
    bandwidth_mbps REAL NOT NULL CHECK (bandwidth_mbps > 0),
    active         INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
    PRIMARY KEY (topology_id, link_id),
    CHECK (source <> target),
    FOREIGN KEY (topology_id, source) REFERENCES nodes(topology_id, node_id) ON DELETE CASCADE,
    FOREIGN KEY (topology_id, target) REFERENCES nodes(topology_id, node_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS simulation_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    topology_id   INTEGER REFERENCES topologies(id) ON DELETE SET NULL,
    topology_name TEXT NOT NULL,
    strategy      TEXT NOT NULL CHECK (strategy IN ('latency','capacity')),
    config_hash   TEXT NOT NULL,
    topology_json TEXT NOT NULL,
    demands_json  TEXT NOT NULL,
    notes         TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS failure_events (
    run_id INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    seq    INTEGER NOT NULL,
    kind   TEXT NOT NULL CHECK (kind IN ('node','link')),
    target TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('fail','recover')),
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS simulation_metrics (
    run_id INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    phase  TEXT NOT NULL CHECK (phase IN ('baseline','scenario')),
    metric TEXT NOT NULL,
    value  REAL,
    PRIMARY KEY (run_id, phase, metric)
);
CREATE TABLE IF NOT EXISTS routing_results (
    run_id         INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    phase          TEXT NOT NULL CHECK (phase IN ('baseline','scenario')),
    demand_id      TEXT NOT NULL,
    source         TEXT NOT NULL,
    destination    TEXT NOT NULL,
    rate_mbps      REAL NOT NULL,
    priority       INTEGER NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('served','congested','disconnected')),
    path           TEXT NOT NULL,
    hop_count      INTEGER NOT NULL,
    latency_ms     REAL,
    bottleneck_mbps REAL,
    delivered_mbps REAL NOT NULL,
    PRIMARY KEY (run_id, phase, demand_id)
);
CREATE TABLE IF NOT EXISTS link_results (
    run_id        INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    phase         TEXT NOT NULL CHECK (phase IN ('baseline','scenario')),
    link_id       TEXT NOT NULL,
    operational   INTEGER NOT NULL CHECK (operational IN (0,1)),
    capacity_mbps REAL NOT NULL,
    latency_ms    REAL NOT NULL,
    load_mbps     REAL NOT NULL,
    utilization   REAL NOT NULL,
    PRIMARY KEY (run_id, phase, link_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON simulation_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_hash    ON simulation_runs(config_hash);
"""

_RUN_NAME_MAX = 100
_NOTES_MAX = 2_000


class DatabaseError(Exception):
    """Raised for any persistence problem (wraps sqlite3 errors)."""


class DuplicateRunError(DatabaseError):
    """An identical experiment configuration is already saved."""

    def __init__(self, run_id: int) -> None:
        super().__init__(f"An identical experiment is already saved as run #{run_id}.")
        self.run_id = run_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


class Database:
    """Thin, explicit data-access layer around one SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # --------------------------------------------------------- connection
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection; commit on success, roll back on any error."""
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        except sqlite3.Error as exc:
            raise DatabaseError(f"Database error ({self.path.name}): {exc}") from exc
        finally:
            if conn is not None:
                conn.close()

    def initialize(self) -> None:
        """Create the database file and tables if they do not exist."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatabaseError(f"Cannot create database folder: {exc}") from exc
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ---------------------------------------------------------- topologies
    def save_topology(self, topo: Topology, overwrite: bool = False) -> int:
        """Save ``topo`` under its name. Returns the topology ID."""
        now = _now()
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM topologies WHERE name = ?", (topo.name,)).fetchone()
            if row and not overwrite:
                raise DatabaseError(f"A topology named {topo.name!r} already exists.")
            if row:
                topology_id = row["id"]
                conn.execute("UPDATE topologies SET updated_at = ? WHERE id = ?", (now, topology_id))
                conn.execute("DELETE FROM nodes WHERE topology_id = ?", (topology_id,))  # cascades to links
            else:
                cur = conn.execute(
                    "INSERT INTO topologies (name, created_at, updated_at) VALUES (?, ?, ?)",
                    (topo.name, now, now),
                )
                topology_id = int(cur.lastrowid)
            conn.executemany(
                "INSERT INTO nodes (topology_id, node_id, name, node_type, active) VALUES (?,?,?,?,?)",
                [(topology_id, n.node_id, n.name, n.node_type.value, int(n.active)) for n in topo.nodes],
            )
            conn.executemany(
                "INSERT INTO links (topology_id, link_id, source, target, latency_ms,"
                " bandwidth_mbps, active) VALUES (?,?,?,?,?,?,?)",
                [
                    (topology_id, l.link_id, l.source, l.target, l.latency_ms,
                     l.bandwidth_mbps, int(l.active))
                    for l in topo.links
                ],
            )
        return topology_id

    def list_topologies(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT t.id, t.name, t.created_at, t.updated_at,"
                " (SELECT COUNT(*) FROM nodes n WHERE n.topology_id = t.id) AS node_count,"
                " (SELECT COUNT(*) FROM links l WHERE l.topology_id = t.id) AS link_count"
                " FROM topologies t ORDER BY t.name"
            ).fetchall()
        return [dict(r) for r in rows]

    def load_topology(self, ref: int | str) -> Topology:
        """Load a saved topology by ID (int) or name (str)."""
        column = "id" if isinstance(ref, int) and not isinstance(ref, bool) else "name"
        with self._connect() as conn:
            head = conn.execute(f"SELECT id, name FROM topologies WHERE {column} = ?", (ref,)).fetchone()
            if head is None:
                raise DatabaseError(f"Topology {ref!r} was not found.")
            nodes = conn.execute(
                "SELECT node_id, name, node_type, active FROM nodes WHERE topology_id = ? ORDER BY node_id",
                (head["id"],),
            ).fetchall()
            links = conn.execute(
                "SELECT source, target, latency_ms, bandwidth_mbps, active FROM links"
                " WHERE topology_id = ? ORDER BY link_id",
                (head["id"],),
            ).fetchall()
        data = {
            "name": head["name"],
            "nodes": [
                {"id": r["node_id"], "name": r["name"], "type": r["node_type"], "active": bool(r["active"])}
                for r in nodes
            ],
            "links": [
                {"source": r["source"], "target": r["target"], "latency_ms": r["latency_ms"],
                 "bandwidth_mbps": r["bandwidth_mbps"], "active": bool(r["active"])}
                for r in links
            ],
        }
        try:
            return Topology.from_dict(data)
        except TopologyError as exc:
            raise DatabaseError(f"Stored topology is invalid: {exc}") from exc

    def delete_topology(self, topology_id: int) -> None:
        """Delete a saved topology. Saved runs keep their own snapshot."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM topologies WHERE id = ?", (topology_id,))
            if cur.rowcount == 0:
                raise DatabaseError(f"Topology #{topology_id} was not found.")

    # ----------------------------------------------------------------- runs
    def save_run(
        self,
        run: ScenarioRun,
        name: str,
        topology_id: int | None = None,
        notes: str = "",
        allow_duplicate: bool = False,
    ) -> int:
        """Persist a scenario run with its full configuration and results."""
        try:
            name = validate_name(name)
        except TopologyError as exc:
            raise DatabaseError(f"Invalid run name: {exc}") from exc
        if len(name) > _RUN_NAME_MAX or not isinstance(notes, str) or len(notes) > _NOTES_MAX:
            raise DatabaseError("Run name or notes are too long.")
        demands = [r.demand for r in run.baseline_result.demand_results]
        strategy = run.baseline_result.strategy
        topology_dict = run.baseline_topology.to_dict()
        demand_dicts = [d.to_dict() for d in demands]
        event_dicts = [e.to_dict() for e in run.events]
        config_hash = hashlib.sha256(
            _canonical({
                "t": topology_dict,
                "d": sorted(demand_dicts, key=lambda d: d["id"]),
                "e": event_dicts,
                "s": strategy.value,
            }).encode("utf-8")
        ).hexdigest()
        with self._connect() as conn:
            if not allow_duplicate:
                existing = conn.execute(
                    "SELECT id FROM simulation_runs WHERE config_hash = ? ORDER BY id LIMIT 1",
                    (config_hash,),
                ).fetchone()
                if existing:
                    raise DuplicateRunError(existing["id"])
            cur = conn.execute(
                "INSERT INTO simulation_runs (name, created_at, topology_id, topology_name, strategy,"
                " config_hash, topology_json, demands_json, notes) VALUES (?,?,?,?,?,?,?,?,?)",
                (name, _now(), topology_id, run.baseline_topology.name, strategy.value,
                 config_hash, _canonical(topology_dict), _canonical(demand_dicts), notes),
            )
            run_id = int(cur.lastrowid)
            self._insert_events(conn, run_id, run.events)
            for phase, topo, result in (
                ("baseline", run.baseline_topology, run.baseline_result),
                ("scenario", run.scenario_topology, run.scenario_result),
            ):
                self._insert_metrics(conn, run_id, phase, compute_metrics(topo, result))
                self._insert_results(conn, run_id, phase, result)
        return run_id

    @staticmethod
    def _insert_events(conn: sqlite3.Connection, run_id: int, events: Sequence[FailureEvent]) -> None:
        conn.executemany(
            "INSERT INTO failure_events (run_id, seq, kind, target, action) VALUES (?,?,?,?,?)",
            [(run_id, i, e.kind, e.target, e.action) for i, e in enumerate(events)],
        )

    @staticmethod
    def _insert_metrics(conn: sqlite3.Connection, run_id: int, phase: str, metrics: NetworkMetrics) -> None:
        rows = [
            (run_id, phase, key, None if value is None else float(value))
            for key, value in metrics.to_flat_dict().items()
            if key != "strategy"
        ]
        conn.executemany(
            "INSERT INTO simulation_metrics (run_id, phase, metric, value) VALUES (?,?,?,?)", rows
        )

    @staticmethod
    def _insert_results(conn: sqlite3.Connection, run_id: int, phase: str, result: SimulationResult) -> None:
        conn.executemany(
            "INSERT INTO routing_results (run_id, phase, demand_id, source, destination, rate_mbps,"
            " priority, status, path, hop_count, latency_ms, bottleneck_mbps, delivered_mbps)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (run_id, phase, r.demand.demand_id, r.demand.source, r.demand.destination,
                 r.demand.rate_mbps, r.demand.priority, r.status, ">".join(r.route.path),
                 r.route.hop_count, r.latency_ms, r.route.bottleneck_mbps, r.delivered_mbps)
                for r in result.demand_results
            ],
        )
        conn.executemany(
            "INSERT INTO link_results (run_id, phase, link_id, operational, capacity_mbps,"
            " latency_ms, load_mbps, utilization) VALUES (?,?,?,?,?,?,?,?)",
            [
                (run_id, phase, s.link_id, int(s.operational), s.capacity_mbps,
                 s.latency_ms, s.load_mbps, s.utilization)
                for s in result.link_states.values()
            ],
        )

    def list_runs(self) -> list[dict[str, Any]]:
        """Summaries of saved runs, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT r.id, r.name, r.created_at, r.topology_name, r.strategy, r.notes,"
                " (SELECT COUNT(*) FROM failure_events e WHERE e.run_id = r.id) AS event_count,"
                " (SELECT COUNT(*) FROM routing_results d WHERE d.run_id = r.id AND d.phase = 'baseline')"
                "   AS demand_count,"
                " (SELECT value FROM simulation_metrics m WHERE m.run_id = r.id AND m.phase = 'baseline'"
                "   AND m.metric = 'delivery_ratio') AS baseline_delivery_ratio,"
                " (SELECT value FROM simulation_metrics m WHERE m.run_id = r.id AND m.phase = 'scenario'"
                "   AND m.metric = 'delivery_ratio') AS scenario_delivery_ratio,"
                " (SELECT value FROM simulation_metrics m WHERE m.run_id = r.id AND m.phase = 'scenario'"
                "   AND m.metric = 'conn_pair_connectivity') AS scenario_connectivity,"
                " (SELECT value FROM simulation_metrics m WHERE m.run_id = r.id AND m.phase = 'scenario'"
                "   AND m.metric = 'disconnected_demands') AS scenario_disconnected"
                " FROM simulation_runs r ORDER BY r.id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: int) -> dict[str, Any]:
        """Full stored detail of one run (plain data, ready for export)."""
        with self._connect() as conn:
            head = conn.execute("SELECT * FROM simulation_runs WHERE id = ?", (run_id,)).fetchone()
            if head is None:
                raise DatabaseError(f"Run #{run_id} was not found.")
            events = conn.execute(
                "SELECT seq, kind, target, action FROM failure_events WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
            metric_rows = conn.execute(
                "SELECT phase, metric, value FROM simulation_metrics WHERE run_id = ? ORDER BY metric",
                (run_id,),
            ).fetchall()
            routing = conn.execute(
                "SELECT phase, demand_id, source, destination, rate_mbps, priority, status, path,"
                " hop_count, latency_ms, bottleneck_mbps, delivered_mbps"
                " FROM routing_results WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
            links = conn.execute(
                "SELECT phase, link_id, operational, capacity_mbps, latency_ms, load_mbps, utilization"
                " FROM link_results WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        metrics: dict[str, dict[str, float | None]] = {p: {} for p in PHASES}
        for row in metric_rows:
            metrics[row["phase"]][row["metric"]] = row["value"]
        routing_rows = []
        for row in routing:
            item = dict(row)
            item["path"] = item["path"].split(">") if item["path"] else []
            routing_rows.append(item)
        link_rows = []
        for row in links:
            item = dict(row)
            item["operational"] = bool(item["operational"])
            link_rows.append(item)
        return {
            "id": head["id"],
            "name": head["name"],
            "created_at": head["created_at"],
            "topology_name": head["topology_name"],
            "strategy": head["strategy"],
            "strategy_label": Strategy.parse(head["strategy"]).label,
            "notes": head["notes"],
            "config_hash": head["config_hash"],
            "demands": json.loads(head["demands_json"]),
            "events": [dict(e) for e in events],
            "metrics": metrics,
            "routing": routing_rows,
            "links": link_rows,
        }

    def load_experiment(self, run_id: int) -> Experiment:
        """Rebuild the exact configuration of a saved run so it can be re-run."""
        with self._connect() as conn:
            head = conn.execute(
                "SELECT name, strategy, topology_json, demands_json FROM simulation_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if head is None:
                raise DatabaseError(f"Run #{run_id} was not found.")
            events = conn.execute(
                "SELECT kind, target, action FROM failure_events WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        try:
            return Experiment(
                head["name"],
                Topology.from_dict(json.loads(head["topology_json"])),
                tuple(Demand.from_dict(d) for d in json.loads(head["demands_json"])),
                tuple(FailureEvent(e["kind"], e["target"], e["action"]) for e in events),
                Strategy.parse(head["strategy"]),
            )
        except (ValueError, TypeError) as exc:  # includes TopologyError / SimulationError
            raise DatabaseError(f"Stored run #{run_id} could not be rebuilt: {exc}") from exc

    def compare_runs(self, run_ids: Sequence[int]) -> list[dict[str, Any]]:
        """Metrics of several runs side by side (baseline and scenario phases)."""
        if not run_ids:
            raise DatabaseError("Select at least one run to compare.")
        if len(set(run_ids)) != len(run_ids):
            raise DatabaseError("Each run can only be selected once.")
        rows = []
        for run_id in run_ids:
            detail = self.get_run(run_id)
            rows.append({
                "run_id": detail["id"],
                "name": detail["name"],
                "created_at": detail["created_at"],
                "topology_name": detail["topology_name"],
                "strategy": detail["strategy_label"],
                "events": [f"{e['action']} {e['kind']} {e['target']}" for e in detail["events"]],
                "baseline": detail["metrics"]["baseline"],
                "scenario": detail["metrics"]["scenario"],
            })
        return rows

    def delete_run(self, run_id: int) -> None:
        """Delete one run and (via ON DELETE CASCADE) all of its child rows."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM simulation_runs WHERE id = ?", (run_id,))
            if cur.rowcount == 0:
                raise DatabaseError(f"Run #{run_id} was not found.")
