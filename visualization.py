"""Plotly figure builders for NetTwin (no Streamlit dependency).

Dark "black canvas" design: every colour below was chosen to be clearly visible
on pure black (contrast ratio >= 4.5:1) and distinguishable from the others
(this is verified by tests/test_visualization.py).
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import networkx as nx
import plotly.graph_objects as go

from simulation import CONGESTION_THRESHOLD, LinkState
from topology import NodeType, Topology

# ------------------------------------------------------------------ palette
BG = "#000000"            # stark black canvas
TEXT = "#EEF2F8"
MUTED = "#8F9BB0"
GRID = "#171E2B"
GRID_STRONG = "#2A3446"
FONT = "Segoe UI, Inter, system-ui, sans-serif"

# One distinct hue per device type (symbol shape differs too).
NODE_STYLES: dict[NodeType, tuple[str, str]] = {
    NodeType.HOST: ("circle", "#00E676"),     # neon green
    NodeType.SWITCH: ("square", "#448AFF"),   # electric blue
    NodeType.ROUTER: ("diamond", "#FF7A00"),  # vivid orange
    NodeType.SERVER: ("hexagon", "#E040FB"),  # magenta-violet
}
COLOR_OK = "#A9B6CC"       # healthy link: light slate, never lost on black
COLOR_ROUTE = "#00E5FF"    # selected route: electric cyan
COLOR_WARN = "#FFEA00"     # utilisation >= 80 %: yellow
COLOR_OVER = "#FF1744"     # over capacity: red
COLOR_FAILED = "#FF5252"   # failed link / failed node: coral red (links dashed)
COLOR_BEFORE = "#40C4FF"   # "baseline" series
COLOR_AFTER = "#FF9100"    # "scenario" series
COLOR_GOOD = "#00E676"

PALETTE: dict[str, str] = {
    **{t.value: c for t, (_, c) in NODE_STYLES.items()},
    "Active link": COLOR_OK,
    "Selected route": COLOR_ROUTE,
    "Utilisation >= 80%": COLOR_WARN,
    "Over capacity": COLOR_OVER,
    "Failed": COLOR_FAILED,
}

Positions = dict[str, tuple[float, float]]


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _style(fig: go.Figure, title: str | None = None, height: int | None = None,
           margin: dict | None = None, **layout) -> go.Figure:
    """Apply the shared black theme to a figure."""
    legend = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=12))
    legend.update(layout.pop("legend", {}))
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.01, font=dict(size=16, color=TEXT)) if title else None,
        height=height,
        margin=margin or dict(l=10, r=10, t=56 if title else 10, b=10),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, color=TEXT, size=13),
        hoverlabel=dict(bgcolor="#0B1018", bordercolor=GRID_STRONG, font=dict(family=FONT, color=TEXT)),
        legend=legend,
        **layout,
    )
    axis = dict(gridcolor=GRID, zerolinecolor=GRID_STRONG, linecolor=GRID_STRONG,
                tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


# ------------------------------------------------------------------- layout
def layout_from_structure(
    node_ids: Sequence[str], edges: Iterable[tuple[str, str]], seed: int = 7
) -> Positions:
    """Deterministic spring layout from plain node IDs and (a, b) edges."""
    graph = nx.Graph()
    graph.add_nodes_from(sorted(node_ids))
    graph.add_edges_from(sorted(tuple(sorted(e)) for e in edges))
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_nodes() == 1:
        return {next(iter(graph.nodes)): (0.0, 0.0)}
    raw = nx.spring_layout(graph, seed=seed, iterations=200, weight=None)
    return {node: (float(x), float(y)) for node, (x, y) in raw.items()}


def layout_positions(topo: Topology, seed: int = 7) -> Positions:
    """Layout of ALL nodes (failed ones keep their place, so views stay comparable)."""
    return layout_from_structure(
        [n.node_id for n in topo.nodes], [(l.source, l.target) for l in topo.links], seed
    )


def _segments(links, positions: Positions) -> tuple[list, list]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for link in links:
        (x0, y0), (x1, y1) = positions[link.source], positions[link.target]
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    return xs, ys


# ----------------------------------------------------------------- topology
def topology_figure(
    topo: Topology,
    positions: Positions | None = None,
    route_links: Iterable[str] = (),
    utilization: Mapping[str, float] | None = None,
    title: str | None = None,
    height: int = 520,
) -> go.Figure:
    """Interactive topology on black.

    Node colour and shape = device type (green circle host, blue square switch,
    orange diamond router, violet hexagon server); red X = failed node; red
    dashed = failed link; cyan = selected route; yellow/red links = utilisation
    >= 80 % / over capacity (when ``utilization`` is given).
    """
    positions = positions or layout_positions(topo)
    route_set = set(route_links)
    util = utilization or {}
    groups: dict[str, list] = {"ok": [], "route": [], "warn": [], "over": [], "failed": []}
    for link in topo.links:
        if not topo.is_link_operational(link):
            groups["failed"].append(link)
        elif link.link_id in route_set:
            groups["route"].append(link)
        elif util.get(link.link_id, 0.0) > 1.0 + 1e-9:
            groups["over"].append(link)
        elif util.get(link.link_id, 0.0) >= CONGESTION_THRESHOLD:
            groups["warn"].append(link)
        else:
            groups["ok"].append(link)
    styles = {  # key: (legend label, colour, width, dash, glow width)
        "ok": ("Active link", COLOR_OK, 2.5, "solid", 0),
        "route": ("Selected route", COLOR_ROUTE, 5, "solid", 14),
        "warn": ("Utilisation >= 80%", COLOR_WARN, 4, "solid", 10),
        "over": ("Over capacity", COLOR_OVER, 5, "solid", 14),
        "failed": ("Failed link", COLOR_FAILED, 2.5, "dash", 0),
    }
    fig = go.Figure()
    for key, (label, color, width, dash, glow) in styles.items():
        if not groups[key]:
            continue
        xs, ys = _segments(groups[key], positions)
        if glow:  # soft halo under emphasised links (no legend entry)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color=_rgba(color, 0.22), width=glow)))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label, hoverinfo="skip",
                                 line=dict(color=color, width=width, dash=dash)))
    if topo.links:  # invisible markers at link midpoints carry the hover text
        mid_x, mid_y, text = [], [], []
        for link in topo.links:
            (x0, y0), (x1, y1) = positions[link.source], positions[link.target]
            mid_x.append((x0 + x1) / 2)
            mid_y.append((y0 + y1) / 2)
            status = "operational" if topo.is_link_operational(link) else "FAILED"
            line = f"<b>{link.link_id}</b><br>{link.latency_ms:g} ms, {link.bandwidth_mbps:g} Mbps<br>{status}"
            if link.link_id in util:
                line += f"<br>utilisation {util[link.link_id] * 100:.1f}%"
            text.append(line)
        fig.add_trace(go.Scatter(
            x=mid_x, y=mid_y, mode="markers", hovertext=text, hoverinfo="text",
            marker=dict(size=16, opacity=0), showlegend=False,
        ))
    for node_type, (symbol, color) in NODE_STYLES.items():
        nodes = [n for n in topo.nodes if n.node_type is node_type and n.active]
        if not nodes:
            continue
        xs = [positions[n.node_id][0] for n in nodes]
        ys = [positions[n.node_id][1] for n in nodes]
        fig.add_trace(go.Scatter(  # glow halo
            x=xs, y=ys, mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(symbol="circle", size=46, color=_rgba(color, 0.16)),
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", name=node_type.value,
            text=[n.node_id for n in nodes], textposition="top center",
            textfont=dict(color=TEXT, size=13),
            hovertext=[f"<b>{n.node_id}</b> - {n.name}<br>{n.node_type.value}<br>active" for n in nodes],
            hoverinfo="text",
            marker=dict(symbol=symbol, size=26, color=color, line=dict(color="#FFFFFF", width=2)),
        ))
    failed = [n for n in topo.nodes if not n.active]
    if failed:
        xs = [positions[n.node_id][0] for n in failed]
        ys = [positions[n.node_id][1] for n in failed]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(symbol="circle", size=46, color=_rgba(COLOR_FAILED, 0.20)),
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", name="Failed node",
            text=[n.node_id for n in failed], textposition="top center",
            textfont=dict(color=COLOR_FAILED, size=12),
            hovertext=[f"<b>{n.node_id}</b> - {n.name}<br>{n.node_type.value}<br>FAILED" for n in failed],
            hoverinfo="text",
            marker=dict(symbol="x", size=28, color=COLOR_FAILED, line=dict(color=COLOR_FAILED, width=3)),
        ))
    return _style(
        fig, title=title, height=height, hovermode="closest",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )


# ------------------------------------------------------------------- charts
def _pct(value: float) -> float:
    return value * 100.0


def _util_color(utilization: float) -> str:
    if utilization > 1.0 + 1e-9:
        return COLOR_OVER
    return COLOR_WARN if utilization >= CONGESTION_THRESHOLD else COLOR_BEFORE


def utilization_figure(
    series: Sequence[tuple[str, Mapping[str, LinkState]]],
    title: str = "Link utilisation",
) -> go.Figure:
    """Bars of utilisation (%) per operational link with 80 % / 100 % guides.

    A single series is coloured by severity (blue / yellow / red); several
    series are shown side by side in distinct colours.
    """
    fig = go.Figure()
    palette = [COLOR_BEFORE, COLOR_AFTER, COLOR_GOOD, "#E040FB"]
    link_ids = sorted({lid for _, states in series for lid, s in states.items() if s.operational})
    for index, (label, states) in enumerate(series):
        values = [states[l].utilization if l in states and states[l].operational else None for l in link_ids]
        color = ([_util_color(v) if v is not None else COLOR_BEFORE for v in values]
                 if len(series) == 1 else palette[index % len(palette)])
        fig.add_trace(go.Bar(
            name=label, x=link_ids, y=[None if v is None else _pct(v) for v in values],
            marker=dict(color=color, line=dict(width=0)),
            hovertemplate="%{x}<br>%{y:.1f}%<extra>" + label + "</extra>",
        ))
    fig.add_hline(y=100, line_dash="dash", line_color=COLOR_OVER, annotation_text="capacity",
                  annotation_font_color=COLOR_OVER)
    fig.add_hline(y=_pct(CONGESTION_THRESHOLD), line_dash="dot", line_color=COLOR_WARN)
    return _style(
        fig, title=title, height=380, barmode="group", bargap=0.25,
        yaxis_title="Utilisation (%)", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def latency_figure(rows: Sequence[Mapping[str, object]], title: str = "Modelled route latency") -> go.Figure:
    """Grouped bars per demand. Each row: {demand_id, before_ms, after_ms} (None = no route)."""
    fig = go.Figure()
    ids = [str(r["demand_id"]) for r in rows]
    fig.add_trace(go.Bar(name="Baseline", x=ids, y=[r["before_ms"] for r in rows], marker_color=COLOR_BEFORE))
    fig.add_trace(go.Bar(name="Scenario", x=ids, y=[r["after_ms"] for r in rows], marker_color=COLOR_AFTER))
    return _style(
        fig, title=title, height=340, barmode="group", bargap=0.25,
        yaxis_title="Latency (ms, configured)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def sweep_figure(rows: Sequence, top: int = 12) -> go.Figure:
    """Horizontal bars: delivery-ratio drop caused by failing each single component."""
    shown = list(rows)[:top][::-1]
    labels = [f"{r.kind} {r.target}" for r in shown]
    fig = go.Figure(go.Bar(
        x=[-_pct(r.delivery_ratio_change) for r in shown], y=labels, orientation="h",
        marker_color=[COLOR_OVER if r.disconnected_demands else COLOR_WARN for r in shown],
        hovertemplate="%{y}<br>delivery drop %{x:.1f} pp<extra></extra>",
    ))
    return _style(
        fig, title="Delivery-ratio drop when one component fails", height=max(260, 34 * len(shown) + 110),
        xaxis_title="Drop (percentage points)",
    )


def trend_figure(runs: Sequence[Mapping[str, object]]) -> go.Figure:
    """Scenario delivery ratio and connectivity across saved runs (oldest first)."""
    ordered = sorted(runs, key=lambda r: r["id"])
    labels = [f"#{r['id']} {r['name']}" for r in ordered]
    fig = go.Figure()
    for key, name, color in (
        ("scenario_delivery_ratio", "Delivery ratio", COLOR_ROUTE),
        ("scenario_connectivity", "Node-pair connectivity", COLOR_GOOD),
    ):
        fig.add_trace(go.Scatter(
            x=labels, y=[None if r[key] is None else _pct(r[key]) for r in ordered],
            mode="lines+markers", name=name, line=dict(color=color, width=3),
            marker=dict(size=9, color=color, line=dict(color="#000000", width=1)),
        ))
    return _style(
        fig, title="Scenario results across saved runs", height=340, yaxis_title="%",
        yaxis_range=[0, 105], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def run_comparison_figure(rows: Sequence[Mapping[str, object]]) -> go.Figure:
    """Grouped bars comparing scenario outcomes of selected runs."""
    names = [f"#{r['run_id']} {r['name']}" for r in rows]
    fig = go.Figure()
    for key, label, color in (
        ("delivery_ratio", "Delivery ratio (%)", COLOR_ROUTE),
        ("conn_pair_connectivity", "Connectivity (%)", COLOR_GOOD),
    ):
        fig.add_trace(go.Bar(
            name=label, x=names,
            y=[None if r["scenario"].get(key) is None else _pct(r["scenario"][key]) for r in rows],
            marker_color=color,
        ))
    return _style(
        fig, title="Scenario outcome by run", height=340, barmode="group", bargap=0.25,
        yaxis_range=[0, 105], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
