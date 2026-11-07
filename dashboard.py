"""NetTwin Streamlit dashboard.

This module only handles layout, navigation and user input. All calculations
live in the core modules (topology, routing, simulation, metrics, database,
storage) so they can be tested without Streamlit.

Run it through ``python main.py`` (recommended) or directly with
``streamlit run dashboard.py``.
"""
from __future__ import annotations

import os
import platform
import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database, DatabaseError, DuplicateRunError  # noqa: E402
from metrics import (  # noqa: E402
    compare_results,
    compute_metrics,
    connectivity,
    failure_sweep,
    structural_weak_points,
)
from routing import alternative_paths, capacity_aware_path, shortest_path  # noqa: E402
from simulation import (  # noqa: E402
    CONGESTION_THRESHOLD,
    Demand,
    DemandError,
    FailureEvent,
    LinkState,
    SimulationError,
    Strategy,
    apply_event,
    apply_events,
    run_scenario,
    run_simulation,
)
from storage import (  # noqa: E402
    REPORT_METRICS,
    StorageError,
    build_report_html,
    export_experiment_json,
    export_topology_json,
    import_topology_json,
    run_table_csv,
)
from topology import (  # noqa: E402
    NodeType,
    Topology,
    TopologyError,
    sample_topology,
    validate_name,
)
from visualization import (  # noqa: E402
    NODE_STYLES,
    COLOR_FAILED,
    COLOR_OK,
    COLOR_OVER,
    COLOR_ROUTE,
    COLOR_WARN,
    latency_figure,
    layout_from_structure,
    run_comparison_figure,
    sweep_figure,
    topology_figure,
    trend_figure,
    utilization_figure,
)

APP_VERSION = "1.0.0"
SAMPLE_NAME = "Sample Campus Network"
PAGES = [
    "Overview",
    "Network Topology",
    "Routing Lab",
    "Traffic Simulation",
    "Failure & Recovery Lab",
    "Resilience Analysis",
    "Experiment History",
    "Settings & About",
]
ss = st.session_state


# ----------------------------------------------------------------- helpers
THEME_CSS = """
:root{--accent:#00E5FF;--line:#1E2633;--line-soft:#121824;--muted:#8F9BB0;--panel:#07090E;--text:#EEF2F8}
html,body,.stApp,[class*="css"]{font-family:"Segoe UI",Inter,system-ui,sans-serif}
.stApp{background:#000}
header[data-testid="stHeader"]{background:rgba(0,0,0,.65);backdrop-filter:blur(8px)}
#MainMenu,footer{visibility:hidden}
.block-container{padding-top:2.4rem;max-width:1500px}

/* headings */
h1{color:#fff;font-weight:700;letter-spacing:-.02em;padding-bottom:.2rem}
h1::after{content:"";display:block;width:56px;height:3px;margin-top:.45rem;border-radius:2px;
  background:linear-gradient(90deg,#00E5FF,#E040FB)}
h2,h3,h4{color:var(--text);font-weight:600;letter-spacing:-.01em}
[data-testid="stCaptionContainer"]{color:var(--muted)}
hr{border-color:var(--line-soft)}

/* sidebar */
[data-testid="stSidebar"]{border-right:1px solid #161D29}
.nt-brand{display:flex;align-items:center;gap:.75rem;padding:.4rem .2rem 1rem}
.nt-logo{width:40px;height:40px;border-radius:12px;display:flex;align-items:center;justify-content:center;
  font-size:1.2rem;color:#000;font-weight:800;background:linear-gradient(135deg,#00E5FF,#E040FB);
  box-shadow:0 0 18px rgba(0,229,255,.35)}
.nt-name{font-size:1.35rem;font-weight:700;color:#fff;line-height:1.1}
.nt-sub{font-size:.7rem;color:var(--muted);letter-spacing:.04em}
[data-testid="stSidebar"] [data-testid="stRadioGroup"]{gap:.2rem}
[data-testid="stSidebar"] [data-testid="stRadioOption"]{padding:.5rem .7rem;border-radius:10px;
  border:1px solid transparent;width:100%;transition:all .15s ease}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-hovered="true"]{background:#0D131B;border-color:var(--line)}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"],
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked){background:rgba(0,229,255,.10);
  border-color:rgba(0,229,255,.5)}

/* KPI cards */
div[data-testid="stMetric"]{background:linear-gradient(180deg,#0B1017 0%,#06080C 100%);
  border:1px solid var(--line);border-radius:14px;padding:14px 16px;box-shadow:0 8px 22px rgba(0,0,0,.65)}
[data-testid="stMetricLabel"] p{color:var(--muted);text-transform:uppercase;letter-spacing:.07em;
  font-size:.7rem;font-weight:600}
[data-testid="stMetricValue"]{color:#fff;font-weight:700;font-size:1.85rem}

/* buttons */
[data-testid^="stBaseButton"]{border-radius:10px;font-weight:600;transition:all .15s ease}
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-secondaryFormSubmit"]{
  background:#0B1018;border:1px solid #2A3446;color:var(--text)}
[data-testid="stBaseButton-secondary"]:hover:not(:disabled),
[data-testid="stBaseButton-secondaryFormSubmit"]:hover:not(:disabled){
  border-color:var(--accent);color:var(--accent);box-shadow:0 0 14px rgba(0,229,255,.25)}
[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{
  background:var(--accent);border:1px solid var(--accent);color:#000}
[data-testid="stBaseButton-primary"]:hover:not(:disabled){box-shadow:0 0 18px rgba(0,229,255,.55)}
[data-testid^="stBaseButton"]:disabled{opacity:.38}

/* tabs, tables, forms, expanders, charts, alerts */
[data-testid="stTab"]{font-weight:600}
[data-testid="stTab"][data-selected="true"],[data-testid="stTab"][aria-selected="true"]{color:var(--accent)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
[data-testid="stForm"]{border:1px solid var(--line);border-radius:14px;background:var(--panel)}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:var(--panel)}
[data-testid="stPlotlyChart"]{border:1px solid var(--line-soft);border-radius:14px;overflow:hidden;background:#000}
[data-testid="stAlert"]{border-radius:12px}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-track{background:#000}
::-webkit-scrollbar-thumb{background:#243044;border-radius:8px}

/* colour key */
.nt-key{display:flex;flex-wrap:wrap;gap:.45rem;margin:.2rem 0 .8rem}
.nt-chip{display:inline-flex;align-items:center;gap:.5rem;padding:.22rem .7rem;border:1px solid var(--line);
  border-radius:999px;background:#080B10;font-size:.78rem;color:var(--text)}
.nt-dot{width:.72rem;height:.72rem;border-radius:50%;background:var(--c);box-shadow:0 0 9px var(--c)}
.nt-bar{width:1.25rem;height:3px;border-radius:2px;background:var(--c);box-shadow:0 0 8px var(--c)}
.nt-bar.dashed{background:repeating-linear-gradient(90deg,var(--c) 0 4px,transparent 4px 7px);box-shadow:none}
"""


def inject_theme() -> None:
    """Add the custom dark stylesheet (the base theme itself is in .streamlit/config.toml)."""
    css = "\n".join(line for line in THEME_CSS.splitlines() if line.strip())
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def color_key() -> None:
    """Legend explaining the colours used on the topology maps."""
    chips = [f'<span class="nt-chip"><span class="nt-dot" style="--c:{c}"></span>{t.value}</span>'
             for t, (_, c) in NODE_STYLES.items()]
    chips += [
        f'<span class="nt-chip"><span class="nt-bar" style="--c:{COLOR_OK}"></span>Active link</span>',
        f'<span class="nt-chip"><span class="nt-bar" style="--c:{COLOR_ROUTE}"></span>Selected route</span>',
        f'<span class="nt-chip"><span class="nt-bar" style="--c:{COLOR_WARN}"></span>Utilisation ≥ 80%</span>',
        f'<span class="nt-chip"><span class="nt-bar" style="--c:{COLOR_OVER}"></span>Over capacity</span>',
        f'<span class="nt-chip"><span class="nt-bar dashed" style="--c:{COLOR_FAILED}"></span>Failed link / node</span>',
    ]
    st.markdown('<div class="nt-key">' + "".join(chips) + "</div>", unsafe_allow_html=True)


def db_path() -> Path:
    """Database location: $NETTWIN_DB if set, otherwise data/nettwin.db."""
    return Path(os.environ.get("NETTWIN_DB", ROOT / "data" / "nettwin.db"))


@st.cache_resource(show_spinner=False)
def get_database(path: str) -> Database:
    database = Database(path)
    database.initialize()
    return database


@st.cache_data(show_spinner=False)
def _positions(node_ids: tuple, edges: tuple) -> dict:
    return layout_from_structure(node_ids, edges)


def positions_for(topo: Topology) -> dict:
    return _positions(
        tuple(n.node_id for n in topo.nodes), tuple((l.source, l.target) for l in topo.links)
    )


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def num(value: float | None, unit: str = "") -> str:
    return "—" if value is None else f"{value:.2f}{unit}"


def signed(value: float | None, digits: int = 2, unit: str = "") -> str | None:
    return None if value is None else f"{value:+.{digits}f}{unit}"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "nettwin"


def kpi(column, label: str, value: str, delta: str | None = None,
        delta_color: str = "normal", help: str | None = None) -> None:
    """KPI card (styled through the stylesheet in THEME_CSS)."""
    with column:
        st.metric(label, value, delta, delta_color=delta_color, help=help)


def plot(fig, key: str | None = None) -> None:
    """Show a Plotly figure with its own dark styling (not Streamlit's theme overrides)."""
    st.plotly_chart(fig, theme=None, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def flash(kind: str, message: str) -> None:
    ss.flash.append((kind, message))


def show_flash() -> None:
    for kind, message in ss.flash:
        getattr(st, kind)(message)
    ss.flash = []


def fmt_metric(value: float | None, kind: str) -> str:
    if value is None:
        return "—"
    return pct(value) if kind == "pct" else (f"{value:.0f}" if kind == "int" else f"{value:.2f}")


def default_demands() -> list[Demand]:
    """Demonstration traffic for the sample topology."""
    return [
        Demand("web", "H1", "S1", 60),
        Demand("files", "H2", "S1", 80),
        Demand("db", "H3", "S2", 40),
        Demand("backup", "H4", "S2", 50),
    ]


def is_sample_compatible(topo: Topology) -> bool:
    return all(topo.has_node(n) for d in default_demands() for n in (d.source, d.destination))


# ------------------------------------------------------------------- state
def init_state(db: Database) -> None:
    if ss.get("_ready"):
        return
    if not db.list_topologies():
        db.save_topology(sample_topology())
    latest = max(db.list_topologies(), key=lambda t: (t["updated_at"], t["id"]))
    ss.topology = db.load_topology(latest["id"])
    ss.demands = default_demands() if latest["name"] == SAMPLE_NAME and is_sample_compatible(ss.topology) else []
    ss.events = []
    ss.strategy = Strategy.CAPACITY.value
    ss.flash = []
    ss.confirm_reset = False
    ss._ready = True


def sanitize_events() -> int:
    """Drop demands/events that no longer fit the working topology. Returns drops."""
    topo: Topology = ss.topology
    kept_demands = [d for d in ss.demands if topo.has_node(d.source) and topo.has_node(d.destination)]
    dropped = len(ss.demands) - len(kept_demands)
    ss.demands = kept_demands
    probe, kept = topo.copy(), []
    for event in ss.events:
        try:
            apply_event(probe, event)
            kept.append(event)
        except SimulationError:
            dropped += 1
    ss.events = kept
    return dropped


def scenario_topology() -> Topology:
    return apply_events(ss.topology, ss.events)


def adopt_topology(topo: Topology, demands: list[Demand] | None = None) -> None:
    ss.topology = topo
    ss.events = []
    if demands is not None:
        ss.demands = demands
    sanitize_events()


def state_selector(key: str) -> Topology:
    """Let the user pick the baseline network or the network with failures applied."""
    if not ss.events:
        return ss.topology
    choice = st.radio(
        "Network state", ["Baseline", "With failure scenario applied"],
        horizontal=True, key=f"state_{key}",
        help="The scenario is built on the Failure & Recovery Lab page. The baseline is never changed.",
    )
    return ss.topology if choice == "Baseline" else scenario_topology()


def link_states_from_rows(rows: list[dict], phase: str) -> dict[str, LinkState]:
    return {
        r["link_id"]: LinkState(r["link_id"], r["operational"], r["capacity_mbps"],
                                r["latency_ms"], r["load_mbps"], r["utilization"])
        for r in rows if r["phase"] == phase
    }


# ---------------------------------------------------------------- Overview
def page_overview(db: Database) -> None:
    st.title("Network overview")
    topo: Topology = ss.topology
    if not topo.nodes:
        st.info("This topology is empty. Add devices on the **Network Topology** page.")
        return
    result = run_simulation(topo, ss.demands, ss.strategy)
    m = compute_metrics(topo, result)
    cols = st.columns(6)
    kpi(cols[0], "Nodes up", f"{m.connectivity.operational_nodes}/{m.connectivity.total_nodes}")
    kpi(cols[1], "Links", str(len(topo.links)))
    kpi(cols[2], "Traffic demands", str(m.total_demands))
    kpi(cols[3], "Connectivity", pct(m.connectivity.pair_connectivity),
        help="Reachable node pairs / all node pairs (operational network).")
    kpi(cols[4], "Delivery ratio", pct(m.delivery_ratio) if m.total_demands else "—",
        help="Delivered Mbps / offered Mbps (modelled).")
    kpi(cols[5], "Capacity violations", str(m.violation_count),
        help="Links whose offered load exceeds capacity.")
    left, right = st.columns([3, 2])
    with left:
        color_key()
        plot(
            topology_figure(topo, positions_for(topo),
                            utilization={k: v.utilization for k, v in result.link_states.items()}),
            key="overview_map",
        )
    with right:
        st.subheader("Status")
        failed = topo.failed_nodes() + topo.failed_links()
        st.markdown(f"{':green[●]' if not failed else ':red[●]'} **Components** — "
                    + ("all operational" if not failed else f"{len(failed)} marked failed in the baseline"))
        st.markdown(f"{':green[●]' if m.violation_count == 0 else ':red[●]'} **Capacity** — "
                    + ("no link over capacity" if m.violation_count == 0 else f"{m.violation_count} link(s) over capacity"))
        st.markdown(f"{':green[●]' if m.disconnected_demands == 0 else ':red[●]'} **Demands** — "
                    f"{m.served_demands} served, {m.congested_demands} congested, {m.disconnected_demands} disconnected")
        st.markdown(f"{':green[●]' if ss.events == [] else ':orange[●]'} **Scenario** — "
                    + ("no failure events queued" if not ss.events else f"{len(ss.events)} event(s) queued in the lab"))
        st.caption(f"Routing strategy: {Strategy.parse(ss.strategy).label}. All figures are modelled estimates.")
        st.subheader("Recent experiments")
        runs = db.list_runs()[:5]
        if runs:
            st.dataframe(pd.DataFrame([{
                "Run": f"#{r['id']}", "Name": r["name"], "Events": r["event_count"],
                "Delivery after": pct(r["scenario_delivery_ratio"]),
            } for r in runs]), hide_index=True)
        else:
            st.info("No saved experiments yet. Run one in the **Failure & Recovery Lab**.")


# -------------------------------------------------------- Network Topology
def _reset_to_sample_ui() -> None:
    if not ss.confirm_reset:
        if st.button("Reset to sample topology…", key="reset_sample_btn"):
            ss.confirm_reset = True
            st.rerun()
        return
    st.warning("This replaces the working topology, traffic demands and failure events with the sample. "
               "Saved topologies and experiments in the database are not touched.")
    yes, no = st.columns(2)
    if yes.button("Yes, reset to sample", type="primary", key="reset_sample_yes"):
        adopt_topology(sample_topology(), default_demands())
        ss.confirm_reset = False
        flash("success", "Working topology reset to the sample network.")
        st.rerun()
    if no.button("Cancel", key="reset_sample_no"):
        ss.confirm_reset = False
        st.rerun()


def _nodes_tab(topo: Topology) -> None:
    if topo.nodes:
        st.dataframe(pd.DataFrame([{
            "ID": n.node_id, "Name": n.name, "Type": n.node_type.value,
            "Status": "active" if n.active else "FAILED", "Links": len(topo.links_of(n.node_id)),
        } for n in topo.nodes]), hide_index=True)
    else:
        st.info("No nodes yet. Add the first one below.")
    with st.form("add_node_form", clear_on_submit=True):
        st.subheader("Add node")
        c1, c2, c3 = st.columns(3)
        node_id = c1.text_input("Node ID", key="new_node_id", max_chars=32, help="Letters, digits, _ and -")
        name = c2.text_input("Display name", key="new_node_name", max_chars=64)
        node_type = c3.selectbox("Type", [t.value for t in NodeType], key="new_node_type")
        submitted = st.form_submit_button("Add node")
    if submitted:
        try:
            topo.add_node(node_id.strip(), name, node_type)
        except TopologyError as exc:
            st.error(str(exc))
        else:
            flash("success", f"Added node {node_id.strip()}.")
            st.rerun()
    if not topo.nodes:
        return
    st.subheader("Edit or delete a node")
    selected = st.selectbox("Node", [n.node_id for n in topo.nodes], key="edit_node_select")
    node = topo.get_node(selected)
    types = [t.value for t in NodeType]
    with st.form(f"edit_node_form_{selected}"):
        new_name = st.text_input("Display name", node.name, key=f"edit_name_{selected}")
        new_type = st.selectbox("Type", types, index=types.index(node.node_type.value), key=f"edit_type_{selected}")
        active = st.checkbox("Operational (administratively up)", node.active, key=f"edit_active_{selected}")
        save = st.form_submit_button("Save changes")
    if save:
        try:
            topo.update_node(selected, new_name, new_type)
            topo.set_node_active(selected, active)
        except TopologyError as exc:
            st.error(str(exc))
        else:
            sanitize_events()
            flash("success", f"Updated node {selected}.")
            st.rerun()
    confirm = st.checkbox(f"I understand that deleting {selected} also removes its links and any demands using it",
                          key=f"confirm_delete_node_{selected}")
    if st.button("Delete node", disabled=not confirm, key="delete_node_btn"):
        removed = topo.remove_node(selected)
        sanitize_events()
        flash("success", f"Deleted node {selected} and {len(removed)} link(s).")
        st.rerun()


def _links_tab(topo: Topology) -> None:
    if topo.links:
        st.dataframe(pd.DataFrame([{
            "Link": l.link_id, "Latency (ms)": l.latency_ms, "Bandwidth (Mbps)": l.bandwidth_mbps,
            "Admin status": "up" if l.active else "DOWN",
            "Operational": "yes" if topo.is_link_operational(l) else "no",
        } for l in topo.links]), hide_index=True)
    else:
        st.info("No links yet.")
    ids = [n.node_id for n in topo.nodes]
    if len(ids) < 2:
        st.info("Add at least two nodes before creating links.")
        return
    with st.form("add_link_form", clear_on_submit=False):
        st.subheader("Add link")
        c1, c2, c3, c4 = st.columns(4)
        source = c1.selectbox("From", ids, key="new_link_source")
        target = c2.selectbox("To", ids, index=1, key="new_link_target")
        latency = c3.number_input("Latency (ms)", min_value=0.0, value=1.0, step=0.5, key="new_link_latency")
        bandwidth = c4.number_input("Bandwidth (Mbps)", min_value=0.0, value=100.0, step=50.0, key="new_link_bw")
        submitted = st.form_submit_button("Add link")
    if submitted:
        try:
            topo.add_link(source, target, latency, bandwidth)
        except TopologyError as exc:
            st.error(str(exc))
        else:
            flash("success", f"Added link {source}-{target}.")
            st.rerun()
    if not topo.links:
        return
    st.subheader("Edit or delete a link")
    selected = st.selectbox("Link", [l.link_id for l in topo.links], key="edit_link_select")
    link = next(l for l in topo.links if l.link_id == selected)
    with st.form(f"edit_link_form_{selected}"):
        c1, c2 = st.columns(2)
        new_lat = c1.number_input("Latency (ms)", min_value=0.0, value=float(link.latency_ms), step=0.5, key=f"edit_lat_{selected}")
        new_bw = c2.number_input("Bandwidth (Mbps)", min_value=0.0, value=float(link.bandwidth_mbps), step=50.0, key=f"edit_bw_{selected}")
        up = st.checkbox("Administratively up", link.active, key=f"edit_link_active_{selected}")
        save = st.form_submit_button("Save changes")
    if save:
        try:
            topo.update_link(link.source, link.target, new_lat, new_bw)
            topo.set_link_active(link.source, link.target, up)
        except TopologyError as exc:
            st.error(str(exc))
        else:
            sanitize_events()
            flash("success", f"Updated link {selected}.")
            st.rerun()
    confirm = st.checkbox(f"I understand that deleting {selected} cannot be undone", key=f"confirm_delete_link_{selected}")
    if st.button("Delete link", disabled=not confirm, key="delete_link_btn"):
        topo.remove_link(link.source, link.target)
        sanitize_events()
        flash("success", f"Deleted link {selected}.")
        st.rerun()


def _storage_tab(db: Database, topo: Topology) -> None:
    st.subheader("Save to the database")
    new_name = st.text_input("Topology name", topo.name, key="topology_name_input")
    overwrite = st.checkbox("Overwrite a saved topology with this name", key="overwrite_topology")
    if st.button("Save topology", key="save_topology_btn"):
        try:
            candidate = topo.copy()
            candidate.name = validate_name(new_name)
            db.save_topology(candidate, overwrite=overwrite)
        except (TopologyError, DatabaseError) as exc:
            st.error(str(exc))
        else:
            topo.name = candidate.name
            flash("success", f"Saved topology '{topo.name}'.")
            st.rerun()
    st.subheader("Load a saved topology")
    saved = db.list_topologies()
    if saved:
        labels = {f"{t['name']} ({t['node_count']} nodes, {t['link_count']} links)": t["id"] for t in saved}
        choice = st.selectbox("Saved topologies", list(labels), key="load_topology_select")
        c1, c2 = st.columns(2)
        if c1.button("Load into workspace", key="load_topology_btn"):
            try:
                adopt_topology(db.load_topology(labels[choice]))
            except DatabaseError as exc:
                st.error(str(exc))
            else:
                flash("success", "Topology loaded. Failure events were cleared.")
                st.rerun()
        confirm = c2.checkbox("Confirm delete", key="confirm_delete_saved_topology")
        if c2.button("Delete saved topology", disabled=not confirm, key="delete_saved_topology_btn"):
            try:
                db.delete_topology(labels[choice])
            except DatabaseError as exc:
                st.error(str(exc))
            else:
                flash("success", "Saved topology deleted (saved experiments keep their own copy).")
                st.rerun()
    else:
        st.info("Nothing saved yet.")
    st.subheader("JSON import / export")
    st.download_button("Download topology JSON", export_topology_json(topo),
                       file_name=f"{slug(topo.name)}.json", mime="application/json", key="download_topology")
    upload = st.file_uploader("Import a topology JSON file", type=["json"], key="import_uploader")
    if upload is not None and st.button("Import file", key="import_topology_btn"):
        try:
            adopt_topology(import_topology_json(upload.getvalue()))
        except StorageError as exc:
            st.error(str(exc))
        else:
            flash("success", "Topology imported. Review it, then save it if you want to keep it.")
            st.rerun()
    st.subheader("Sample topology")
    _reset_to_sample_ui()


def page_topology(db: Database) -> None:
    st.title("Network topology")
    topo: Topology = ss.topology
    tabs = st.tabs(["Map", "Nodes", "Links", "Save · load · share"])
    with tabs[0]:
        if topo.nodes:
            color_key()
            plot(topology_figure(topo, positions_for(topo)), key="topology_map")
            nodes, bridges = structural_weak_points(topo)
            if nodes or bridges:
                st.caption("Single points of failure (structure only): "
                           f"nodes {', '.join(nodes) or 'none'}; links {', '.join(bridges) or 'none'}.")
        else:
            st.info("The topology is empty.")
    with tabs[1]:
        _nodes_tab(topo)
    with tabs[2]:
        _links_tab(topo)
    with tabs[3]:
        _storage_tab(db, topo)


# ------------------------------------------------------------- Routing Lab
def _render_route(result, title: str) -> None:
    st.markdown(f"#### {title}")
    if not result.found:
        st.error(result.message)
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Hops", result.hop_count)
    c2.metric("Modelled latency", f"{result.latency_ms:.2f} ms")
    c3.metric("Bottleneck capacity", f"{result.bottleneck_mbps:g} Mbps")
    st.markdown("**Path:** " + " → ".join(result.path))
    st.dataframe(pd.DataFrame(
        [{"Link": lid, "Projected utilisation": f"{u * 100:.1f}%", "Over capacity": "YES" if u > 1 else "no"}
         for lid, u in result.link_utilization.items()]
    ), hide_index=True)
    st.markdown("**Why this route**")
    for reason in result.reasons:
        st.markdown(f"- {reason}")


def page_routing(db: Database) -> None:
    st.title("Routing lab")
    topo = state_selector("routing")
    nodes = [n.node_id for n in topo.nodes]
    if len(nodes) < 2:
        st.info("Add at least two nodes on the **Network Topology** page to try routing.")
        return
    c1, c2, c3, c4 = st.columns(4)
    source = c1.selectbox("Source", nodes, key="route_src")
    destination = c2.selectbox("Destination", nodes, index=1, key="route_dst")
    rate = c3.number_input("Offered traffic (Mbps)", min_value=0.0, value=10.0, step=5.0, key="route_rate")
    weight = c4.slider("Congestion weight", 0.0, 5.0, 1.0, 0.1, key="route_weight",
                       help="0 = ignore load (same as latency routing); higher = avoid busy links more.")
    background = st.checkbox("Include the current traffic demands as background load", key="route_background")
    loads = None
    if background and ss.demands:
        loads = {k: v.load_mbps for k, v in run_simulation(topo, ss.demands, ss.strategy).link_states.items()}
    if source == destination:
        st.warning("Choose two different nodes.")
        return
    by_latency = shortest_path(topo, source, destination, rate, loads)
    by_capacity = capacity_aware_path(topo, source, destination, rate, loads, weight)
    left, right = st.columns(2)
    with left:
        _render_route(by_latency, "Shortest path (latency)")
    with right:
        _render_route(by_capacity, "Capacity-aware")
    color_key()
    highlight = st.radio("Highlight on the map", ["Shortest path (latency)", "Capacity-aware"],
                         horizontal=True, key="route_highlight")
    chosen = by_latency if highlight.startswith("Shortest") else by_capacity
    plot(topology_figure(topo, positions_for(topo), route_links=chosen.link_ids,
                                    title=f"{highlight}: {source} → {destination}", height=460),
                    key="routing_map")
    with st.expander("Alternative paths (ordered by latency)"):
        k = st.slider("Number of paths", 1, 6, 3, key="route_k")
        alternatives = alternative_paths(topo, source, destination, k, rate, loads)
        if alternatives:
            st.dataframe(pd.DataFrame([{
                "Rank": i, "Path": " → ".join(a.path), "Hops": a.hop_count,
                "Latency (ms)": round(a.latency_ms, 2), "Bottleneck (Mbps)": a.bottleneck_mbps,
            } for i, a in enumerate(alternatives, 1)]), hide_index=True)
        else:
            st.info("No alternative path exists with the current failures.")
    with st.expander("How routes are chosen"):
        st.markdown(
            "- **Shortest path:** minimises the sum of configured link latencies (Dijkstra).\n"
            "- **Capacity-aware:** minimises the sum over links of `latency / max_latency + weight × u²`, "
            "where `u` = (existing load + this demand) / bandwidth. This is a heuristic trade-off, "
            "not a globally optimal multi-objective solution.\n"
            "- Failed nodes and failed links are never used. Ties are broken deterministically."
        )


# ------------------------------------------------------ Traffic Simulation
def _next_demand_id() -> str:
    used = {d.demand_id for d in ss.demands}
    n = len(used) + 1
    while f"d{n}" in used:
        n += 1
    return f"d{n}"


def _demands_editor(topo: Topology) -> None:
    ids = [n.node_id for n in ss.topology.nodes]
    if ss.demands:
        st.dataframe(pd.DataFrame([{
            "ID": d.demand_id, "Source": d.source, "Destination": d.destination,
            "Rate (Mbps)": d.rate_mbps, "Priority": d.priority,
        } for d in ss.demands]), hide_index=True)
    if len(ids) >= 2:
        with st.form("add_demand_form", clear_on_submit=True):
            st.subheader("Add demand")
            c1, c2, c3, c4, c5 = st.columns(5)
            demand_id = c1.text_input("ID (optional)", key="new_demand_id", max_chars=32)
            source = c2.selectbox("Source", ids, key="new_demand_src")
            destination = c3.selectbox("Destination", ids, index=1, key="new_demand_dst")
            rate = c4.number_input("Rate (Mbps)", min_value=0.0, value=10.0, step=5.0, key="new_demand_rate")
            priority = c5.number_input("Priority (0-10)", min_value=0, max_value=10, value=0, step=1, key="new_demand_prio",
                                       help="Higher priority demands are routed first.")
            submitted = st.form_submit_button("Add demand")
        if submitted:
            try:
                if any(d.demand_id == demand_id.strip() for d in ss.demands):
                    raise DemandError(f"Demand ID {demand_id.strip()!r} already exists.")
                ss.demands.append(Demand(demand_id.strip() or _next_demand_id(), source, destination, rate, int(priority)))
            except DemandError as exc:
                st.error(str(exc))
            else:
                flash("success", "Demand added.")
                st.rerun()
    else:
        st.info("Add at least two nodes before defining traffic.")
    c1, c2 = st.columns(2)
    if ss.demands:
        picked = c1.multiselect("Remove demands", [d.demand_id for d in ss.demands], key="remove_demands_select")
        if c1.button("Remove selected", disabled=not picked, key="remove_demands_btn"):
            ss.demands = [d for d in ss.demands if d.demand_id not in picked]
            flash("success", f"Removed {len(picked)} demand(s).")
            st.rerun()
    if is_sample_compatible(ss.topology) and c2.button("Restore sample demands", key="restore_demands_btn"):
        ss.demands = default_demands()
        flash("success", "Sample demands restored.")
        st.rerun()


def page_traffic(db: Database) -> None:
    st.title("Traffic simulation")
    topo = state_selector("traffic")
    with st.expander("Traffic demands", expanded=not ss.demands):
        _demands_editor(topo)
    if not ss.demands:
        st.info("Define at least one traffic demand to run a simulation.")
        return
    result = run_simulation(topo, ss.demands, ss.strategy)
    m = compute_metrics(topo, result)
    cols = st.columns(6)
    kpi(cols[0], "Offered", f"{m.offered_mbps:.1f} Mbps")
    kpi(cols[1], "Delivered", f"{m.delivered_mbps:.1f} Mbps")
    kpi(cols[2], "Delivery ratio", pct(m.delivery_ratio))
    kpi(cols[3], "Avg route latency", num(m.avg_latency_ms, " ms"), help="Configured latency only; no queueing delay.")
    kpi(cols[4], "Congested links", str(m.congested_link_count), help=f"Utilisation ≥ {CONGESTION_THRESHOLD:.0%}")
    kpi(cols[5], "Capacity violations", str(m.violation_count), help="Offered load above capacity.")
    for state in result.capacity_violations:
        st.error(f"**{state.link_id}** is overloaded: {state.load_mbps:.1f} Mbps offered on "
                 f"{state.capacity_mbps:g} Mbps ({state.utilization_pct:.0f}%).")
    for state in result.congested_links():
        if not state.is_violation:
            st.warning(f"**{state.link_id}** is congested at {state.utilization_pct:.0f}% utilisation.")
    for item in result.demand_results:
        if item.status == "disconnected":
            st.error(f"Demand **{item.demand.demand_id}** ({item.demand.source} → {item.demand.destination}) "
                     f"is disconnected: {item.message}")
    left, right = st.columns(2)
    with left:
        plot(utilization_figure([(Strategy.parse(ss.strategy).label, result.link_states)]),
                        key="traffic_util")
    with right:
        plot(topology_figure(topo, positions_for(topo), height=380,
                                        utilization={k: v.utilization for k, v in result.link_states.items()}),
                        key="traffic_map")
    st.subheader("Demand results")
    st.dataframe(pd.DataFrame([{
        "Demand": r.demand.demand_id, "From": r.demand.source, "To": r.demand.destination,
        "Offered (Mbps)": r.demand.rate_mbps, "Delivered (Mbps)": round(r.delivered_mbps, 2),
        "Status": r.status, "Path": " → ".join(r.route.path) or "—",
        "Latency (ms)": None if r.latency_ms is None else round(r.latency_ms, 2),
    } for r in result.demand_results]), hide_index=True)
    st.subheader("Link load")
    st.dataframe(pd.DataFrame(sorted(
        [{"Link": s.link_id, "Load (Mbps)": round(s.load_mbps, 2), "Capacity (Mbps)": s.capacity_mbps,
          "Utilisation": f"{s.utilization_pct:.1f}%", "Operational": "yes" if s.operational else "NO"}
         for s in result.link_states.values()],
        key=lambda r: -float(r["Utilisation"].rstrip("%")))), hide_index=True)
    with st.expander("Model assumptions"):
        st.markdown(
            "- Each demand uses **one path**; demands are routed in order of priority, then rate, then ID.\n"
            "- Link load is the sum of offered rates of demands crossing it; utilisation = load / capacity.\n"
            "- On an overloaded link all flows get the same share (capacity ÷ load). This is a "
            "single-bottleneck approximation.\n"
            "- Latency is the sum of configured link latencies. **These are modelled estimates, not "
            "packet-level measurements.**"
        )


# ------------------------------------------------- Failure & Recovery Lab
def _candidates(topo: Topology, kind: str, action: str) -> list[str]:
    want_active = action == "fail"
    if kind == "node":
        return [n.node_id for n in topo.nodes if n.active == want_active]
    return [l.link_id for l in topo.links if l.active == want_active]


def _event_builder() -> None:
    current = scenario_topology()
    c1, c2, c3 = st.columns([1, 1, 2])
    action = c1.radio("Action", ["fail", "recover"], horizontal=True, key="fl_action",
                      format_func=str.capitalize)
    kind = c2.radio("Component", ["node", "link"], horizontal=True, key="fl_kind", format_func=str.capitalize)
    options = _candidates(current, kind, action)
    if not options:
        c3.info(f"No {kind} can be {'failed' if action == 'fail' else 'recovered'} right now.")
    target = c3.selectbox(f"{kind.capitalize()}", options, key=f"fl_target_{kind}_{action}") if options else None
    b1, b2, b3 = st.columns(3)
    if b1.button("Add event", disabled=target is None, type="primary", key="fl_add"):
        try:
            event = FailureEvent(kind, target, action)
            apply_event(current, event)  # validate against the current scenario state
        except SimulationError as exc:
            st.error(str(exc))
        else:
            ss.events.append(event)
            st.rerun()
    if b2.button("Undo last event", disabled=not ss.events, key="fl_undo"):
        ss.events.pop()
        st.rerun()
    if b3.button("Clear all events", disabled=not ss.events, key="fl_clear"):
        ss.events = []
        st.rerun()


def _save_run_ui(db: Database, run) -> None:
    st.subheader("Save this experiment")
    default_name = f"{ss.topology.name} - {len(ss.events)} event(s)"
    name = st.text_input("Run name", default_name, key="run_name", max_chars=100)
    notes = st.text_area("Notes (optional)", key="run_notes", max_chars=2000)
    allow_duplicate = st.checkbox("Save even if an identical experiment already exists", key="run_allow_duplicate")
    if st.button("Save experiment to history", key="save_run_btn"):
        try:
            known = {t["name"]: t["id"] for t in db.list_topologies()}
            run_id = db.save_run(run, name, topology_id=known.get(ss.topology.name), notes=notes,
                                 allow_duplicate=allow_duplicate)
        except DuplicateRunError as exc:
            st.info(f"{exc} Open it in **Experiment History**, or tick the box above to save a copy.")
        except DatabaseError as exc:
            st.error(str(exc))
        else:
            flash("success", f"Saved as run #{run_id}.")
            st.rerun()


def page_failure_lab(db: Database) -> None:
    st.title("Failure & recovery lab")
    st.caption("Failures are applied to a copy. Your baseline topology is never modified, so every "
               "comparison starts from the same reference.")
    if len(ss.topology.nodes) < 2:
        st.info("Build a topology with at least two nodes first.")
        return
    _event_builder()
    if ss.events:
        st.markdown("**Event sequence**")
        for i, event in enumerate(ss.events, 1):
            st.markdown(f"{i}. {event.describe()}")
    else:
        st.info("No events yet. Add a failure above to compare the network before and after.")
    if not ss.demands:
        st.warning("There are no traffic demands, so only connectivity can be compared. Add demands on the "
                   "**Traffic Simulation** page for a fuller analysis.")
    run = run_scenario(ss.topology, ss.demands, ss.events, ss.strategy)
    cmp = compare_results(run.baseline_topology, run.baseline_result, run.scenario_topology, run.scenario_result)
    b, a, d = cmp.before, cmp.after, cmp.deltas
    cols = st.columns(5)
    kpi(cols[0], "Connectivity", pct(a.connectivity.pair_connectivity), signed(d["pair_connectivity"] * 100, 1, " pp"))
    kpi(cols[1], "Delivery ratio", pct(a.delivery_ratio) if a.total_demands else "—",
        signed(d["delivery_ratio"] * 100, 1, " pp") if a.total_demands else None)
    kpi(cols[2], "Disconnected demands", str(a.disconnected_demands), signed(d["disconnected_demands"], 0), "inverse")
    kpi(cols[3], "Avg route latency", num(a.avg_latency_ms, " ms"), signed(d["avg_latency_ms"], 2, " ms"), "inverse")
    kpi(cols[4], "Capacity violations", str(a.violation_count), signed(d["violation_count"], 0), "inverse")
    if cmp.newly_failed_nodes or cmp.newly_failed_links:
        st.markdown(":red[●] **Failed in scenario:** " + ", ".join(cmp.newly_failed_nodes + cmp.newly_failed_links))
    if cmp.recovered_nodes or cmp.recovered_links:
        st.markdown(":green[●] **Recovered in scenario:** " + ", ".join(cmp.recovered_nodes + cmp.recovered_links))
    tabs = st.tabs(["Topology", "Demand impact", "Link utilisation", "Latency"])
    positions = positions_for(ss.topology)
    with tabs[0]:
        color_key()
        left, right = st.columns(2)
        with left:
            plot(topology_figure(run.baseline_topology, positions, title="Baseline", height=430,
                                            utilization={k: v.utilization for k, v in run.baseline_result.link_states.items()}),
                            key="lab_map_before")
        with right:
            plot(topology_figure(run.scenario_topology, positions, title="Scenario", height=430,
                                            utilization={k: v.utilization for k, v in run.scenario_result.link_states.items()}),
                            key="lab_map_after")
    with tabs[1]:
        if not cmp.impacts:
            st.info("No demands to compare.")
        else:
            st.dataframe(pd.DataFrame([{
                "Demand": i.demand_id, "Change": i.change.replace("_", " "),
                "Before": i.before_status, "After": i.after_status,
                "Path before": " → ".join(i.before_path) or "—", "Path after": " → ".join(i.after_path) or "—",
                "Latency change (ms)": None if i.latency_delta_ms is None else round(i.latency_delta_ms, 2),
            } for i in cmp.impacts]), hide_index=True)
            st.caption(f"{len(cmp.affected_demands)} of {len(cmp.impacts)} demand(s) affected.")
    with tabs[2]:
        plot(utilization_figure([("Baseline", run.baseline_result.link_states),
                                            ("Scenario", run.scenario_result.link_states)]), key="lab_util")
    with tabs[3]:
        rows = [{"demand_id": i.demand_id,
                 "before_ms": run.baseline_result.result_for(i.demand_id).latency_ms,
                 "after_ms": run.scenario_result.result_for(i.demand_id).latency_ms} for i in cmp.impacts]
        if rows:
            plot(latency_figure(rows), key="lab_latency")
        else:
            st.info("No demands to compare.")
    _save_run_ui(db, run)


# ------------------------------------------------------ Resilience Analysis
SWEEP_LIMIT = 150


def page_resilience(db: Database) -> None:
    st.title("Resilience analysis")
    topo = state_selector("resilience")
    if len(topo.nodes) < 2:
        st.info("Build a topology with at least two nodes to analyse its resilience.")
        return
    conn = connectivity(topo)
    cols = st.columns(4)
    kpi(cols[0], "Node-pair connectivity", pct(conn.pair_connectivity),
        help="Reachable node pairs / all node pairs. Failed nodes count as unreachable.")
    kpi(cols[1], "Reachable pairs", f"{conn.reachable_pairs}/{conn.total_pairs}")
    kpi(cols[2], "Separate network parts", str(conn.components))
    kpi(cols[3], "Largest part", f"{conn.largest_component} nodes")
    st.subheader("Structural weak points")
    nodes, bridges = structural_weak_points(topo)
    if nodes or bridges:
        if nodes:
            st.warning("**Single-point-of-failure nodes** (losing one splits the network): " + ", ".join(nodes))
        if bridges:
            st.warning("**Single-point-of-failure links** (no alternative path): " + ", ".join(bridges))
        st.caption("Device-facing links to hosts are normally expected here; look for weak points in the core.")
    else:
        st.success("No single node or link failure can split the operational network.")
    st.subheader("Single-failure sweep")
    st.caption("Each operational node and link is failed on its own and the traffic is re-simulated.")
    total = sum(1 for n in topo.nodes if n.active) + sum(1 for l in topo.links if topo.is_link_operational(l))
    if total > SWEEP_LIMIT and not st.checkbox(f"This network has {total} components; run the sweep anyway",
                                               key="sweep_large"):
        return
    with st.spinner("Simulating single failures…"):
        rows = failure_sweep(topo, ss.demands, ss.strategy)
    if not rows:
        st.info("There are no operational components to fail.")
        return
    worst = rows[0]
    st.markdown(f"**Most harmful single failure:** {worst.kind} `{worst.target}` — delivery ratio "
                f"{worst.delivery_ratio_change * 100:+.1f} pp, {worst.disconnected_demands} demand(s) disconnected, "
                f"{worst.affected_demands} demand(s) affected.")
    if not ss.demands:
        st.info("No traffic demands are defined, so only connectivity effects are shown.")
    plot(sweep_figure(rows), key="sweep_chart")
    st.dataframe(pd.DataFrame([{
        "Failed component": f"{r.kind} {r.target}",
        "Connectivity": pct(r.pair_connectivity), "Δ connectivity (pp)": round(r.connectivity_change * 100, 1),
        "Delivery ratio": pct(r.delivery_ratio), "Δ delivery (pp)": round(r.delivery_ratio_change * 100, 1),
        "Disconnected demands": r.disconnected_demands, "Congested demands": r.congested_demands,
        "Demands affected": r.affected_demands, "Capacity violations": r.violation_count,
    } for r in rows]), hide_index=True)
    with st.expander("Metric definitions and limitations"):
        st.markdown(
            "- **Connectivity** = reachable node pairs ÷ all node pairs (operational graph).\n"
            "- **Delivery ratio** = delivered Mbps ÷ offered Mbps under the single-bottleneck model.\n"
            "- **Affected demands** = demands whose path or status changed versus the unfailed network.\n"
            "- The sweep tests one failure at a time; it does not model simultaneous or cascading failures. "
            "Use the Failure & Recovery Lab for combined scenarios.\n"
            "- No single 'resilience score' is computed: a score would hide trade-offs between metrics."
        )


# ------------------------------------------------------ Experiment History
def _load_run_callback(run_id: int) -> None:
    """Runs before the page re-renders, so it may set the widget-bound strategy key."""
    try:
        exp = get_database(str(db_path())).load_experiment(run_id)
    except DatabaseError as exc:
        flash("error", str(exc))
        return
    ss.topology = exp.topology
    ss.demands = list(exp.demands)
    ss.events = list(exp.events)
    ss.strategy = exp.strategy.value
    flash("success", f"Run #{run_id} loaded: topology, demands, events and strategy are now in the workspace.")


def _history_detail(db: Database, run_id: int) -> None:
    detail = db.get_run(run_id)
    st.markdown(f"### #{detail['id']} · {detail['name']}")
    st.caption(f"Saved {detail['created_at']} · topology **{detail['topology_name']}** · "
               f"strategy **{detail['strategy_label']}**")
    if detail["notes"]:
        st.write(detail["notes"])
    if detail["events"]:
        st.markdown("**Events:** " + "; ".join(f"{e['action']} {e['kind']} {e['target']}" for e in detail["events"]))
    else:
        st.markdown("**Events:** none (baseline only)")
    base, scen = detail["metrics"]["baseline"], detail["metrics"]["scenario"]
    st.dataframe(pd.DataFrame([{
        "Metric": label, "Baseline": fmt_metric(base.get(key), kind), "Scenario": fmt_metric(scen.get(key), kind),
    } for key, label, kind in REPORT_METRICS]), hide_index=True)
    plot(utilization_figure([
        ("Baseline", link_states_from_rows(detail["links"], "baseline")),
        ("Scenario", link_states_from_rows(detail["links"], "scenario")),
    ]), key=f"history_util_{run_id}")
    st.dataframe(pd.DataFrame([{
        "Phase": r["phase"], "Demand": r["demand_id"], "From": r["source"], "To": r["destination"],
        "Offered (Mbps)": r["rate_mbps"], "Delivered (Mbps)": round(r["delivered_mbps"], 2),
        "Status": r["status"], "Path": " → ".join(r["path"]) or "—",
    } for r in detail["routing"]]), hide_index=True)
    st.subheader("Export")
    c = st.columns(5)
    for column, table in zip(c[:3], ("metrics", "routing", "links")):
        column.download_button(f"CSV: {table}", run_table_csv(detail, table).encode("utf-8-sig"),
                               file_name=f"run{run_id}_{table}.csv", mime="text/csv", key=f"dl_csv_{table}_{run_id}")
    c[3].download_button("HTML report", build_report_html(detail), file_name=f"run{run_id}_report.html",
                         mime="text/html", key=f"dl_html_{run_id}")
    c[4].download_button("Experiment JSON", export_experiment_json(db.load_experiment(run_id)),
                         file_name=f"run{run_id}_experiment.json", mime="application/json", key=f"dl_json_{run_id}")
    st.button("Load this experiment into the workspace", on_click=_load_run_callback, args=(run_id,),
              key=f"load_run_{run_id}")
    confirm = st.checkbox("Confirm: permanently delete this run", key=f"confirm_delete_run_{run_id}")
    if st.button("Delete run", disabled=not confirm, key=f"delete_run_{run_id}"):
        db.delete_run(run_id)
        flash("success", f"Run #{run_id} deleted.")
        st.rerun()


def page_history(db: Database) -> None:
    st.title("Experiment history")
    runs = db.list_runs()
    if not runs:
        st.info("No experiments saved yet. Build a scenario in the **Failure & Recovery Lab** and save it.")
        return
    st.dataframe(pd.DataFrame([{
        "Run": f"#{r['id']}", "Name": r["name"], "Saved": r["created_at"].replace("T", " ")[:16],
        "Topology": r["topology_name"], "Strategy": Strategy.parse(r["strategy"]).label,
        "Demands": r["demand_count"], "Events": r["event_count"],
        "Delivery before": pct(r["baseline_delivery_ratio"]), "Delivery after": pct(r["scenario_delivery_ratio"]),
        "Disconnected after": int(r["scenario_disconnected"] or 0),
    } for r in runs]), hide_index=True)
    if len(runs) >= 2:
        plot(trend_figure(runs), key="history_trend")
    labels = {f"#{r['id']} · {r['name']} · {r['created_at'][:16].replace('T', ' ')}": r["id"] for r in runs}
    tab_detail, tab_compare = st.tabs(["Run details", "Compare runs"])
    with tab_detail:
        choice = st.selectbox("Run", list(labels), key="history_select")
        _history_detail(db, labels[choice])
    with tab_compare:
        picked = st.multiselect("Select two or more runs", list(labels), key="history_compare")
        if len(picked) < 2:
            st.info("Pick at least two runs to compare them.")
        else:
            rows = db.compare_runs([labels[p] for p in picked])
            st.dataframe(pd.DataFrame([{
                "Run": f"#{r['run_id']} {r['name']}", "Strategy": r["strategy"],
                "Events": "; ".join(r["events"]) or "none",
                "Delivery (baseline → scenario)": f"{pct(r['baseline'].get('delivery_ratio'))} → {pct(r['scenario'].get('delivery_ratio'))}",
                "Connectivity (scenario)": pct(r["scenario"].get("conn_pair_connectivity")),
                "Disconnected": fmt_metric(r["scenario"].get("disconnected_demands"), "int"),
                "Violations": fmt_metric(r["scenario"].get("violation_count"), "int"),
                "Avg latency (ms)": fmt_metric(r["scenario"].get("avg_latency_ms"), "num"),
            } for r in rows]), hide_index=True)
            plot(run_comparison_figure(rows), key="history_compare_chart")


# -------------------------------------------------------- Settings & About
def page_about(db: Database) -> None:
    st.title("Settings & about")
    path = db_path()
    size_kb = path.stat().st_size / 1024 if path.exists() else 0
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Saved topologies", str(len(db.list_topologies())))
    kpi(c2, "Saved experiments", str(len(db.list_runs())))
    kpi(c3, "Database size", f"{size_kb:.0f} KB")
    st.markdown(f"**Database file:** `{path}`  \n**Schema version:** 1  \n"
                "Set the environment variable `NETTWIN_DB` to use a different file.")
    st.subheader("About NetTwin")
    st.markdown(
        "NetTwin is a **graph-based network digital-twin prototype**: a configurable simulation, not a live "
        "twin of real devices. Latency and traffic figures are **modelled estimates**, not measurements of "
        "packets. Nothing here touches your real network.\n\n"
        "**Model summary**\n"
        "- *Shortest path:* minimum total configured latency.\n"
        "- *Capacity-aware:* minimum of Σ (latency ÷ max latency + weight × utilisation²) per link.\n"
        "- *Traffic:* one path per demand; overloaded links share capacity proportionally.\n"
        "- *Failures:* applied to a copy of the baseline; recovery restores the exact prior state.\n"
        "- *Not modelled:* queueing delay, packet loss processes, protocols (OSPF/BGP convergence), "
        "traffic splitting, directional bandwidth."
    )
    st.subheader("Environment")
    import streamlit  # local import: only needed for the version string

    st.dataframe(pd.DataFrame([
        {"Component": "NetTwin", "Version": APP_VERSION},
        {"Component": "Python", "Version": platform.python_version()},
        {"Component": "Streamlit", "Version": streamlit.__version__},
        {"Component": "Plotly", "Version": plotly.__version__},
        {"Component": "NetworkX", "Version": nx.__version__},
        {"Component": "pandas", "Version": pd.__version__},
    ]), hide_index=True)


PAGE_FUNCS = {
    "Overview": page_overview,
    "Network Topology": page_topology,
    "Routing Lab": page_routing,
    "Traffic Simulation": page_traffic,
    "Failure & Recovery Lab": page_failure_lab,
    "Resilience Analysis": page_resilience,
    "Experiment History": page_history,
    "Settings & About": page_about,
}


def main() -> None:
    st.set_page_config(page_title="NetTwin", page_icon="🛰️", layout="wide")
    inject_theme()
    try:
        db = get_database(str(db_path()))
        init_state(db)
    except (DatabaseError, TopologyError) as exc:
        st.error(f"NetTwin could not start: {exc}")
        st.stop()
    with st.sidebar:
        st.markdown('<div class="nt-brand"><div class="nt-logo">N</div><div>'
                    '<div class="nt-name">NetTwin</div>'
                    '<div class="nt-sub">DIGITAL TWIN · RESILIENCE SIMULATOR</div></div></div>',
                    unsafe_allow_html=True)
        page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        st.selectbox("Routing strategy", [s.value for s in Strategy],
                     format_func=lambda v: Strategy.parse(v).label, key="strategy",
                     help="Used by the Overview, Traffic, Failure Lab and Resilience pages.")
        st.caption(f"Topology: **{ss.topology.name}**  \n{len(ss.topology.nodes)} nodes · "
                   f"{len(ss.topology.links)} links · {len(ss.demands)} demands · {len(ss.events)} event(s)")
    dropped = sanitize_events()
    if dropped:
        st.warning(f"{dropped} demand(s) or event(s) no longer fit the topology and were removed.")
    show_flash()
    try:
        PAGE_FUNCS[page](db)
    except DatabaseError as exc:
        st.error(f"Database problem: {exc}")


if __name__ == "__main__":
    main()
