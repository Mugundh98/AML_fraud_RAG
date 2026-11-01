"""Google OAuth for the Calendar server: a one-time browser sign-in, then a saved token that refreshes itself."""

from pathlib import Path

# Read and write events only: no access to calendar settings, sharing or other Google data.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

NOT_AUTHORIZED = "Google Calendar isn't connected yet. Run `python -m scripts.google_auth` and sign in."


def authorize(client_secret_file: Path, token_file: Path):
    """Open the browser for Google's consent screen and save the resulting token."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), SCOPES)
    credentials = flow.run_local_server(port=0, open_browser=True)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def load_credentials(token_file: Path):
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    if not token_file.exists():
        raise RuntimeError(NOT_AUTHORIZED)
    credentials = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not credentials.valid:
        if not credentials.refresh_token:
            raise RuntimeError(NOT_AUTHORIZED)
        try:
            credentials.refresh(Request())
        except RefreshError as exc:
            # Apps in Google's "Testing" mode get refresh tokens that expire after 7 days.
            raise RuntimeError(f"Google sign-in expired ({exc}). {NOT_AUTHORIZED}") from exc
        token_file.write_text(credentials.to_json(), encoding="utf-8")
    return credentials
