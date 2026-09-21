"""
DESTINASI ? Root Entrypoint for Streamlit & Cloud Deployment
Author: DESTINASI Core Engineering Team
Target: DOSM Datathon 2026
"""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dashboard_app = ROOT / "dashboard" / "app.py"
runpy.run_path(str(dashboard_app), run_name="__main__")
