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

### 3. Qdrant Cloud (optional for now)
The project works with embedded Qdrant on disk. To move to the cloud:
1. Sign up at https://cloud.qdrant.io and create a **Free** cluster.
2. Copy the cluster URL and create an API key.
3. Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`, then re-index.

## Phase 2

### 4. Neo4j AuraDB Free
1. Sign up at https://neo4j.com/cloud/aura-free and create a **Free** instance.
2. Save the generated password immediately; it's shown only once.
3. Set `NEO4J_URI` (starts with `neo4j+s://`), `NEO4J_USERNAME`, `NEO4J_PASSWORD`.
4. Free instances have node and relationship limits and pause after a few days without use. Resume them from the Aura console.

## Phase 4

### 5. Tavily (web search)
Sign up at https://tavily.com. The free plan includes monthly search credits, no card needed.

### 6. Google Calendar API
1. Create a project at https://console.cloud.google.com.
2. Enable the **Google Calendar API**.
3. Configure the OAuth consent screen as **External**, leave it in **Testing**, and add your own Google account as a test user.
4. Create an **OAuth client ID** of type **Desktop app** and download the JSON.

### 7. Slack
1. Create a free workspace for testing.
2. Create an app at https://api.slack.com/apps, add bot scopes (`chat:write`, `channels:read`), and install it to the workspace.
3. Create an `#fcu-escalations` channel and invite the bot.

## Phase 5

### 8. Langfuse (tracing)
Sign up at https://cloud.langfuse.com for the free Hobby plan.
