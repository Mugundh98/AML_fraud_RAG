"""Sign in to Google once so the google-calendar MCP server can use your calendar.

Run from the project root:  python -m scripts.google_auth

A browser window opens on Google's consent screen. The token is saved to GOOGLE_TOKEN_FILE (in secrets/, which
git ignores) and refreshes itself. While the OAuth app is in Google's "Testing" mode, sign-ins expire after
7 days: run this again when the assistant says so.
"""

import sys

from app.config import get_settings
from mcp_servers.google_calendar.auth import SCOPES, authorize


def main() -> None:
    settings = get_settings()
    if not settings.google_client_secret_file.exists():
        sys.exit(f"OAuth client file not found: {settings.google_client_secret_file}\n"
                 "Download it from Google Cloud Console (Credentials > OAuth client, type Desktop app).")
    print(f"Opening the browser to grant: {', '.join(SCOPES)}")
    authorize(settings.google_client_secret_file, settings.google_token_file)
    print(f"Signed in. Token saved to {settings.google_token_file}")


if __name__ == "__main__":
    main()
