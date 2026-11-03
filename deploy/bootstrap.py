"""Prepare a hosted container before the app starts (run by deploy/start.sh).

Hosting platforms hand secrets over as environment variables, but three of ours are files: the users file
(API-key hashes) and the Google client secret and token. Each `*_JSON` variable below is written to the path
the app reads, readable only by the app's user.

A hosted copy never runs in open mode: without USERS_JSON anyone with the link would act as a lead (post to
Slack, book meetings, use up the AI quota), so the container refuses to start unless ALLOW_OPEN_MODE=true.
"""

import json
import os
import sys

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    files = {
        "USERS_JSON": settings.users_file,
        "GOOGLE_CLIENT_SECRET_JSON": settings.google_client_secret_file,
        "GOOGLE_TOKEN_JSON": settings.google_token_file,
    }
    for name, path in files.items():
        value = os.environ.get(name)
        if not value:
            continue
        try:
            json.loads(value)
        except json.JSONDecodeError:
            sys.exit(f"{name} isn't valid JSON. Paste the whole file's content as the secret.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
        print(f"Wrote {name} to {path}")

    if not settings.users_file.exists() and os.environ.get("ALLOW_OPEN_MODE", "").lower() != "true":
        sys.exit("No USERS_JSON secret, so anyone with the link would be a lead. Create users with "
                 "`python -m scripts.create_user` and add secrets/users.json as USERS_JSON "
                 "(or set ALLOW_OPEN_MODE=true for a throwaway demo).")
    if settings.google_client_secret_file.exists() and not settings.google_token_file.exists():
        print("No GOOGLE_TOKEN_JSON: Google Calendar is listed but will ask to sign in.")


if __name__ == "__main__":
    main()
