"""Import/export helpers for NetTwin: JSON, CSV and a standalone HTML report.

Security notes
* Imports use ``json.loads`` only (plain data, never code). Input size is
  capped, NaN/Infinity are rejected, and every value is validated by the
  domain classes before use.
* HTML output escapes every dynamic value and contains no scripts or external
  resources.
* CSV cells that start with ``= + - @`` (or a control character) are prefixed
  with an apostrophe so spreadsheet programs do not treat them as formulas.
  This module works on plain dictionaries (as returned by
  ``Database.get_run``) and does not touch the database.
"""
from __future__ import annotations

import csv
import html
import io
import json
from typing import Any, Mapping

from simulation import Experiment
from topology import Topology

MAX_IMPORT_BYTES = 1_000_000
CSV_TABLES = ("metrics", "routing", "links")


class StorageError(ValueError):
    """Raised when an import/export request is invalid."""


# ------------------------------------------------------------------- JSON
def _reject_constant(name: str) -> Any:
    raise ValueError(f"The value {name} is not allowed in JSON files.")


def _parse_json(source: str | bytes) -> Any:
    if isinstance(source, (bytes, bytearray)):
        if len(source) > MAX_IMPORT_BYTES:
            raise StorageError("File is too large (limit 1 MB).")
        try:
            text = bytes(source).decode("utf-8-sig")  # tolerates a Windows BOM
        except UnicodeDecodeError as exc:
            raise StorageError("File is not valid UTF-8 text.") from exc
    elif isinstance(source, str):
        if len(source.encode("utf-8")) > MAX_IMPORT_BYTES:
            raise StorageError("File is too large (limit 1 MB).")
        text = source.lstrip("\ufeff")
    else:
        raise StorageError("Import source must be text or bytes.")
    if not text.strip():
        raise StorageError("The file is empty.")
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise StorageError("JSON is nested too deeply.") from exc
    except ValueError as exc:
        raise StorageError(f"Invalid JSON: {exc}") from exc


def export_topology_json(topo: Topology) -> str:
    """Topology as pretty-printed JSON."""
    return json.dumps(topo.to_dict(), indent=2)


def import_topology_json(source: str | bytes) -> Topology:
    """Parse and validate an uploaded topology file."""
    data = _parse_json(source)
    try:
        return Topology.from_dict(data)
    except ValueError as exc:  # TopologyError is a ValueError
        raise StorageError(f"Invalid topology file: {exc}") from exc


def export_experiment_json(experiment: Experiment) -> str:
    """Full experiment configuration (topology, demands, events, strategy)."""
    return json.dumps(experiment.to_dict(), indent=2)


def import_experiment_json(source: str | bytes) -> Experiment:
    """Parse and validate an uploaded experiment file."""
    data = _parse_json(source)
    try:
        return Experiment.from_dict(data)
    except ValueError as exc:  # SimulationError / TopologyError
        raise StorageError(f"Invalid experiment file: {exc}") from exc


# -------------------------------------------------------------------- CSV
def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".10g")
    text = str(value)
    if text and text[0] in "=+-@\t\r":
        return "'" + text
    return text


def _change(before: float | None, after: float | None) -> float | None:
    return None if before is None or after is None else after - before


def run_table_csv(detail: Mapping[str, Any], table: str) -> str:
    """One table of a saved run as CSV text: 'metrics', 'routing' or 'links'.

    Encode with ``utf-8-sig`` when offering a download so Excel opens it cleanly.
    """
    if table not in CSV_TABLES:
        raise StorageError(f"Unknown table {table!r}. Choose from {', '.join(CSV_TABLES)}.")
    if table == "metrics":
        header = ["metric", "baseline", "scenario", "change"]
        base, scen = detail["metrics"]["baseline"], detail["metrics"]["scenario"]
        rows = [
            [name, base.get(name), scen.get(name), _change(base.get(name), scen.get(name))]
            for name in sorted(set(base) | set(scen))
        ]
    elif table == "routing":
        header = ["phase", "demand_id", "source", "destination", "rate_mbps", "priority",
                  "status", "path", "hop_count", "latency_ms", "bottleneck_mbps", "delivered_mbps"]
        rows = [
            [r["phase"], r["demand_id"], r["source"], r["destination"], r["rate_mbps"],
             r["priority"], r["status"], ">".join(r["path"]), r["hop_count"], r["latency_ms"],
             r["bottleneck_mbps"], r["delivered_mbps"]]
            for r in detail["routing"]
        ]
    else:
        header = ["phase", "link_id", "operational", "capacity_mbps", "latency_ms",
                  "load_mbps", "utilization_pct"]
        rows = [
            [r["phase"], r["link_id"], r["operational"], r["capacity_mbps"], r["latency_ms"],
             r["load_mbps"], r["utilization"] * 100.0]
            for r in detail["links"]
        ]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    for row in rows:
        writer.writerow([_csv_cell(c) for c in row])
    return buffer.getvalue()


# ------------------------------------------------------------------- HTML
_REPORT_METRICS = [
    ("conn_pair_connectivity", "Node-pair connectivity", "pct"),
    ("served_demands", "Served demands", "int"),
    ("congested_demands", "Congested demands", "int"),
    ("disconnected_demands", "Disconnected demands", "int"),
    ("delivered_mbps", "Delivered traffic (Mbps)", "num"),
    ("delivery_ratio", "Delivery ratio", "pct"),
    ("avg_latency_ms", "Average route latency (ms)", "num"),
    ("max_utilization", "Maximum link utilisation", "pct"),
    ("violation_count", "Capacity-violating links", "int"),
    ("total_excess_mbps", "Excess load (Mbps)", "num"),
]

REPORT_METRICS = _REPORT_METRICS  # public: (key, label, kind) used by reports and the dashboard

_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1f2933}
h1{margin-bottom:.2rem}h2{margin-top:2rem;border-bottom:2px solid #d9e2ec;padding-bottom:.3rem}
table{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.92rem}
th,td{border:1px solid #d9e2ec;padding:.35rem .6rem;text-align:left}th{background:#f0f4f8}
td.n{text-align:right;font-variant-numeric:tabular-nums}.muted{color:#627d98}
.served{color:#0b7a3b;font-weight:600}.congested{color:#b25e00;font-weight:600}
.disconnected{color:#b42318;font-weight:600}.note{background:#fff8e1;border-left:4px solid #f0b429;padding:.6rem .9rem}
@media print{body{margin:0}}
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float | None, kind: str) -> str:
    if value is None:
        return "&mdash;"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "int":
        return f"{value:.0f}"
    return f"{value:.2f}"


def _fmt_change(before: float | None, after: float | None, kind: str) -> str:
    delta = _change(before, after)
    if delta is None:
        return "&mdash;"
    if kind == "pct":
        return f"{delta * 100:+.1f} pp"
    if kind == "int":
        return f"{delta:+.0f}"
    return f"{delta:+.2f}"


def _table(headers: list[str], rows: list[list[str]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f"<td class=\"n\">{c}</td>" if i in numeric else f"<td>{c}</td>" for i, c in enumerate(r)
        ) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _status(status: str) -> str:
    return f"<span class=\"{_e(status)}\">{_e(status)}</span>"


def build_report_html(detail: Mapping[str, Any]) -> str:
    """Standalone HTML report for one saved run (no scripts, no external files)."""
    base, scen = detail["metrics"]["baseline"], detail["metrics"]["scenario"]
    metric_rows = [
        [_e(label), _fmt(base.get(key), kind), _fmt(scen.get(key), kind),
         _fmt_change(base.get(key), scen.get(key), kind)]
        for key, label, kind in _REPORT_METRICS
    ]
    events = detail["events"]
    event_html = (
        "<ol>" + "".join(f"<li>{_e(e['action'])} {_e(e['kind'])} <b>{_e(e['target'])}</b></li>" for e in events) + "</ol>"
        if events else "<p class=\"muted\">No failure or recovery events (baseline only).</p>"
    )
    demand_rows = [
        [_e(d["id"]), _e(d["source"]), _e(d["destination"]), f"{d['rate_mbps']:.2f}", _e(d["priority"])]
        for d in detail["demands"]
    ]
    by_key = {(r["phase"], r["demand_id"]): r for r in detail["routing"]}
    impact_rows = []
    for d in detail["demands"]:
        b, s = by_key.get(("baseline", d["id"])), by_key.get(("scenario", d["id"]))
        if not b or not s:
            continue
        changed = b["path"] != s["path"] or b["status"] != s["status"]
        impact_rows.append([
            _e(d["id"]), _status(b["status"]), _e(" > ".join(b["path"]) or "—"),
            _status(s["status"]), _e(" > ".join(s["path"]) or "—"),
            _fmt(s["latency_ms"], "num") if s["latency_ms"] is not None else "&mdash;",
            "yes" if changed else "no",
        ])
    link_rows = []
    for r in detail["links"]:
        if r["phase"] != "scenario" or not (r["load_mbps"] > 0 or not r["operational"]):
            continue
        state = "failed/unavailable" if not r["operational"] else (
            "OVERLOADED" if r["load_mbps"] > r["capacity_mbps"] + 1e-9 else "ok")
        link_rows.append([
            _e(r["link_id"]), f"{r['load_mbps']:.2f}", f"{r['capacity_mbps']:.2f}",
            f"{r['utilization'] * 100:.1f}%", _e(state),
        ])
    notes = f"<p>{_e(detail['notes'])}</p>" if detail.get("notes") else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>NetTwin report - {_e(detail['name'])}</title><style>{_CSS}</style></head><body>
<h1>NetTwin simulation report</h1>
<p class="muted">Run #{_e(detail['id'])} &middot; {_e(detail['name'])} &middot; saved {_e(detail['created_at'])}</p>
<p><b>Topology:</b> {_e(detail['topology_name'])} &nbsp; <b>Routing strategy:</b> {_e(detail['strategy_label'])}</p>
{notes}
<p class="note">All values are <b>modelled estimates</b> from a configurable simulation. They are not
measurements from real devices or packets. Latency is the sum of configured link latencies (no queueing
delay); traffic is carried on a single path per demand.</p>
<h2>Scenario events</h2>{event_html}
<h2>Traffic demands</h2>
{_table(["ID", "Source", "Destination", "Rate (Mbps)", "Priority"], demand_rows, {3, 4})}
<h2>Baseline vs scenario</h2>
{_table(["Metric", "Baseline", "Scenario", "Change"], metric_rows, {1, 2, 3})}
<h2>Demand impact</h2>
{_table(["Demand", "Baseline status", "Baseline path", "Scenario status", "Scenario path", "Scenario latency (ms)", "Changed"], impact_rows, {5})}
<h2>Link load in the scenario</h2>
{_table(["Link", "Load (Mbps)", "Capacity (Mbps)", "Utilisation", "State"], link_rows, {1, 2, 3})}
</body></html>
"""
