#!/bin/sh
# Container entry point: write secret files from environment variables, start the API privately, then the UI.
set -e

python -m deploy.bootstrap

# If the host mounts a persistent disk at /data, keep the audit log there so it survives restarts.
if [ -d /data ] && [ -w /data ]; then
    export AUDIT_DB=/data/audit.db
fi

# The API listens on localhost only; the UI reaches it inside the container.
uvicorn app.api.main:app --host 127.0.0.1 --port 8000 &

exec streamlit run ui/streamlit_app.py \
    --server.port 7860 --server.address 0.0.0.0 --server.headless true \
    --server.fileWatcherType none --client.toolbarMode viewer --browser.gatherUsageStats false
