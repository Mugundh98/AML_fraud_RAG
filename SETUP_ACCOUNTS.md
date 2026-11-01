# Accounts to create

All of these have free tiers. You only need the first two for phase 1.

## Phase 1

### 1. Google Gemini API key (primary LLM)
1. Go to https://aistudio.google.com and sign in with a Google account.
2. Click **Get API key** → **Create API key**.
3. Put it in `.env` as `GEMINI_API_KEY`.
4. Check which Flash models your key can use in AI Studio. `gemini-flash-latest` is an alias for the current one; set `GEMINI_MODEL` to a specific ID if you prefer.

Note: on the free tier, Google may use prompts to improve its models. Fine for this project's public and synthetic data.

### 2. Groq API key (fallback LLM)
1. Go to https://console.groq.com and sign up.
2. **API Keys** → **Create API Key**.
3. Put it in `.env` as `GROQ_API_KEY`.
4. Check https://console.groq.com/docs/models for current production models. Groq retired its Llama 3.x models in 2026, so the default here is `openai/gpt-oss-120b`.

Run `python -m scripts.check_llm` to confirm both work.

### 3. Qdrant Cloud (recommended from phase 2)
Embedded Qdrant on disk can only be opened by one process, so the API and the MCP server can't both use it. To move to the cloud:
1. Sign up at https://cloud.qdrant.io and create a **Free** cluster.
2. Copy the cluster URL and create an API key.
3. Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`, then re-index with `python -m scripts.ingest_docs`.

## Phase 2

### 4. Neo4j AuraDB Free
1. Sign up at https://neo4j.com/cloud/aura-free and create a **Free** instance.
2. Save the generated password immediately; it's shown only once.
3. Set `NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`. On newer instances the username **and** the database name are both the instance ID, so also set `NEO4J_DATABASE=<instance-id>`.
4. Free instances have node and relationship limits and pause after a few days without use. Resume them from the Aura console.
5. Load the data: `python -m scripts.generate_data`, then `python -m scripts.load_graph`.

Alternative for local testing: if Docker Desktop is running,

```bash
docker run -d --name neo4j-aml -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/localpassword -e NEO4J_server_memory_heap_max__size=512m neo4j:5
```

then set `NEO4J_URI=bolt://localhost:7687` and `NEO4J_PASSWORD=localpassword`.

## Phase 4

Each of these is optional: the API only starts the MCP servers whose credentials are set.

### 5. Tavily (web search)
Sign up at https://tavily.com. The free plan has 1,000 credits a month, no card needed. Put the key in `.env` as `TAVILY_API_KEY`.

### 6. Google Calendar API
1. Use any project at https://console.cloud.google.com (the one AI Studio created for your Gemini key works).
2. Enable the **Google Calendar API** (APIs & Services > Library).
3. Configure the OAuth consent screen as **External**, leave it in **Testing**, and under **Audience** add your own Google account as a test user.
4. Create an **OAuth client ID** of type **Desktop app**, download the JSON and save it as `secrets/google_client_secret.json` (git ignores `secrets/`).
5. Sign in once:
   ```bash
   python -m scripts.google_auth
   ```
   The browser shows "Google hasn't verified this app": it's your own app, so continue. The token is saved in `secrets/`. While the app is in **Testing**, Google expires the sign-in after 7 days; the assistant tells you when to run this again.

### 7. Slack
1. Use a workspace where you can install apps (a free one is fine).
2. At https://api.slack.com/apps choose **Create New App > From a manifest**, pick the workspace and paste:
   ```yaml
   display_information:
     name: AML Assistant
   features:
     bot_user:
       display_name: AML Assistant
       always_online: false
   oauth_config:
     scopes:
       bot:
         - chat:write
         - chat:write.public
         - channels:read
   settings:
     org_deploy_enabled: false
     socket_mode_enabled: false
     token_rotation_enabled: false
   ```
3. **Install to Workspace**, then copy the **Bot User OAuth Token** (`xoxb-...`) into `.env` as `SLACK_BOT_TOKEN`.
4. Set `SLACK_DEFAULT_CHANNEL` to the escalation channel's ID: the `C...` part of its URL (`https://app.slack.com/client/T.../C...`). With `chat:write.public` the bot can post to public channels without being invited; for a private channel, `/invite @AML Assistant` first.

## Phase 5

### 8. Langfuse (tracing)
Sign up at https://cloud.langfuse.com for the free Hobby plan.
