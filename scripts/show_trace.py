"""Print the latest traced chat turn from Langfuse as a tree, to check what was recorded.

Run from the project root:  python -m scripts.show_trace
Or a specific trace:        python -m scripts.show_trace <trace_id>
"""

import json
import sys
import time

from app.config import get_settings
from app.observability import init_tracing


def short(value, limit=160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def main() -> None:
    settings = get_settings()
    client = init_tracing(settings)
    api = client.api.observations
    if len(sys.argv) > 1:
        trace_id = sys.argv[1]
    else:
        # Ingestion takes a few seconds; wait for the newest root observation to show up.
        for _ in range(10):
            roots = api.get_many(name="answer-question", is_root_observation=True, limit=1,
                                 environment=settings.langfuse_tracing_environment, fields="core").data
            if roots:
                break
            time.sleep(3)
        else:
            sys.exit("No traces found yet.")
        trace_id = roots[0].trace_id

    observations = api.get_many(trace_id=trace_id, limit=100, fields="core,basic,io,model,usage,metadata").data
    children: dict = {}
    for o in observations:
        children.setdefault(o.parent_observation_id, []).append(o)

    def show(o, depth):
        usage = getattr(o, "usage_details", None) or {}
        model = getattr(o, "provided_model_name", None) or getattr(o, "model", None)
        print(f"{'  ' * depth}- [{o.type}] {o.name}"
              + (f"  model={model}" if model else "")
              + (f"  usage={usage}" if usage else "")
              + (f"  level={o.level}" if o.level and o.level != "DEFAULT" else ""))
        print(f"{'  ' * depth}    in:  {short(o.input)}")
        print(f"{'  ' * depth}    out: {short(o.output)}")
        if o.metadata:
            print(f"{'  ' * depth}    meta: {short(o.metadata, 220)}")
        for child in sorted(children.get(o.id, []), key=lambda c: c.start_time):
            show(child, depth + 1)

    root = next(o for o in observations if not o.parent_observation_id)
    print(f"trace {trace_id}  session={getattr(root, 'session_id', None)}  environment={root.environment}")
    show(root, 0)
    print(f"\n{settings.langfuse_base_url}/trace/{trace_id}")


if __name__ == "__main__":
    main()
