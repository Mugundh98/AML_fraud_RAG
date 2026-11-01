"""Evaluate the agent on the aml-investigations-v0 dataset with Langfuse experiments.

Run from the project root:
    python -m scripts.evaluate show              # print the dataset items (review expected outputs here)
    python -m scripts.evaluate upload            # create/update the dataset in Langfuse (idempotent)
    python -m scripts.evaluate run               # run an experiment; scores land on each item's trace
    python -m scripts.evaluate run --limit 3     # a quick partial run

Each item runs the full agent against the aml-data MCP server only (no web, calendar or Slack), signed in as
an analyst, one item at a time to stay inside the LLM free tiers. A run is ~40-60 model calls.
"""

import argparse
import json
from datetime import datetime

from app.agent.loop import Agent
from app.agent.mcp_hub import MCPHub, default_servers
from app.auth import User
from app.config import PROJECT_ROOT, get_settings
from app.evaluation import DATASET_NAME, ITEM_EVALUATORS, averages, build_items, parse_verdict
from app.llm import build_router
from app.observability import init_tracing

EVAL_USER = User("eval", "Evaluation run", "analyst")


def show(items: list[dict]) -> None:
    for item in items:
        expected = item["expected_output"]
        print(f"\n{item['id']}  [{item['metadata']['kind']}]")
        print(f"  input:    {item['input']['question']}")
        print(f"  expected: verdict={expected['verdict']}  typology={expected['typology']}  "
              f"must_mention={expected['must_mention']}  tools={expected['expected_tools']}")
        print(f"  metadata: {json.dumps({k: v for k, v in item['metadata'].items() if k != 'source'})}")


def upload(langfuse, items: list[dict]) -> None:
    langfuse.create_dataset(
        name=DATASET_NAME,
        description="Planted AML scenarios (one per typology), ordinary customers, a false positive, and policy "
                    "facts. Expected outputs come from the synthetic data generator's answer key.",
        metadata={"source": "data/synthetic/scenarios.csv", "items": len(items)},
    )
    for item in items:  # fixed IDs, so uploading again updates items instead of duplicating them
        langfuse.create_dataset_item(dataset_name=DATASET_NAME, id=item["id"], input=item["input"],
                                     expected_output=item["expected_output"], metadata=item["metadata"])
    langfuse.flush()
    print(f"Uploaded {len(items)} items to dataset {DATASET_NAME}.")


def run(langfuse, settings, limit: int | None, name: str) -> None:
    router = build_router(settings)

    async def task(*, item, **kwargs):
        servers = {"aml-data": default_servers(settings, PROJECT_ROOT)["aml-data"]}
        async with MCPHub(servers) as hub:
            agent = Agent(hub, router, pseudonymize=settings.pii_pseudonymization)
            result = await agent.answer(item.input["question"], [], user=EVAL_USER)
        return {"answer": result.answer, "tools": [s.tool for s in result.tool_calls],
                "model": f"{result.provider}/{result.model}", "fallbacks": len(result.fallback_reasons)}

    dataset = langfuse.get_dataset(DATASET_NAME)
    items = dataset.items[:limit] if limit else dataset.items
    result = langfuse.run_experiment(
        name=name, run_name=f"{name} {datetime.now():%Y-%m-%d %H:%M}",
        description="Full agent, aml-data tools only, analyst role",
        data=items, task=task, evaluators=ITEM_EVALUATORS, run_evaluators=[averages],
        max_concurrency=1,  # free-tier rate limits
        metadata={"models": ", ".join(router.configured()), "pii": str(settings.pii_pseudonymization)},
    )
    langfuse.flush()

    print(f"\n{'item':42} {'expected':15} {'got':15} {'tools'}")
    for r in result.item_results:
        expected = r.item.expected_output
        got = parse_verdict(r.output["answer"]) if r.output else "error"
        scores = {e.name: e.value for e in r.evaluations}
        print(f"{r.item.id.removeprefix(DATASET_NAME + '-'):42} {str(expected.get('verdict')):15} {str(got):15} "
              f"{r.output['tools'] if r.output else ''}  {scores}")
    print()
    for evaluation in result.run_evaluations:
        print(f"{evaluation.name:22} {evaluation.value:.2f}  ({evaluation.comment})")
    if result.experiment_url:
        print(f"\n{result.experiment_url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["show", "upload", "run"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--name", default="agent-baseline")
    args = parser.parse_args()

    settings = get_settings()
    items = build_items(settings.synthetic_dir)
    if args.command == "show":
        show(items)
        return
    langfuse = init_tracing(settings)
    if args.command == "upload":
        upload(langfuse, items)
    else:
        run(langfuse, settings, args.limit, args.name)


if __name__ == "__main__":
    main()
