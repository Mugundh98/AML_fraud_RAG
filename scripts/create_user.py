"""Create an analyst account and print their API key (shown once; only its hash is stored).

Run from the project root:  python -m scripts.create_user --name "Priya Shah" --role senior_analyst

Roles: analyst (Level 1), senior_analyst (Level 2, may escalate to Slack), lead (may also read the audit log).
Creating the first user turns on sign-in for the API: restart it afterwards.
"""

import argparse

from app.auth import ROLES, UserDirectory
from app.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", required=True, choices=ROLES)
    args = parser.parse_args()

    directory = UserDirectory(get_settings().users_file)
    first = directory.open_mode
    user, key = directory.add(args.name, args.role)
    print(f"Created {user.name} ({user.role}), id {user.user_id}.")
    print(f"API key (copy it now, it isn't stored): {key}")
    if first:
        print("This is the first user, so the API now requires sign-in. Restart it, and create a lead too.")


if __name__ == "__main__":
    main()
