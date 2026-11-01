"""Generate the synthetic bank dataset as CSVs in data/synthetic/.

Run from the project root:  python -m scripts.generate_data --transactions 20000
"""

import argparse

from app.config import get_settings
from app.graph.generator import SyntheticBank


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.transactions > settings.max_transactions:
        parser.error(f"--transactions is capped at MAX_TRANSACTIONS={settings.max_transactions} (see .env)")

    bank = SyntheticBank(seed=args.seed, n_transactions=args.transactions).generate()
    bank.write_csv(settings.synthetic_dir)
    print(
        f"Wrote {len(bank.transfers)} transfers, {len(bank.customers)} customers, {len(bank.accounts)} accounts, "
        f"{len(bank.alerts)} alerts and {len(bank.scenarios)} planted scenarios to {settings.synthetic_dir}"
    )


if __name__ == "__main__":
    main()
