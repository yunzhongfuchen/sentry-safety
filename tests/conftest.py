import os
import sys

# Add project root and backend directory to sys.path so tests can import
# both `backend.*` modules and the unqualified `inference_engine` module
# used internally by backend/safety_detection/detector_core.py.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
for path in (ROOT, BACKEND):
    if path not in sys.path:
        sys.path.insert(0, path)
