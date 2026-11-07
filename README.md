# NetTwin — Network Digital Twin & Resilience Simulator

A graph-based network digital-twin **prototype** (BCA project). Build a virtual network, simulate traffic,
fail and recover components, compare routing strategies and save experiments. Everything is simulated
locally; latency and traffic values are **modelled estimates**, not packet measurements.

## Run on Windows 10/11
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
`python main.py` checks dependencies, creates `data\nettwin.db` automatically and opens the dashboard
(default http://localhost:8501). Options: `--port 8600`, `--no-browser`, `--check`.

## Look and feel
The dashboard uses a stark black theme (`.streamlit/config.toml` plus a stylesheet in `dashboard.py`). Device types have fixed high-contrast colours and shapes (host green circle, switch blue square, router orange diamond, server violet hexagon); cyan marks the selected route, yellow/red mark busy/overloaded links and red dashed lines or an X mark failures. A colour key is shown above each map.

## Run the tests
```
python -m pytest -q
```
Tests use temporary databases and never touch `data\nettwin.db`.

## Layout
`topology.py` model · `routing.py` algorithms · `simulation.py` traffic and failures · `metrics.py`
resilience metrics · `database.py` SQLite · `storage.py` JSON/CSV/HTML · `visualization.py` Plotly charts ·
`dashboard.py` Streamlit UI · `main.py` launcher.
