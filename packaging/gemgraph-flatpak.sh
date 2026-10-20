#!/bin/sh
set -e
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
export PYTHONPATH="/app/lib/python${PYVER}/site-packages:/app/lib64/python${PYVER}/site-packages:$PYTHONPATH"
exec python3 -m src.app "$@"
