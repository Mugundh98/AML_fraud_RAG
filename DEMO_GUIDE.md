# Running and demonstrating the AML & Fraud Assistant

Everything runs on your laptop. The databases and AI models are online services, so you need internet.

## Before the demo (10 minutes, ideally an hour before)

1. **Open the project in VS Code** (`E:\VS code projects\aml_fraud_RAG`) and open a terminal (Terminal > New Terminal).

2. **Check the AI models answer** (Gemini and Groq):

   ```bash
   .venv\Scripts\python -m scripts.check_llm
   ```

   If Gemini says its quota is used up, that's fine: the app falls back to Groq. Free Gemini quotas reset daily, so a morning demo is safest.

3. **Check Google Calendar is still signed in.** The sign-in expires every 7 days. If the booking demo fails, or to be safe, sign in again (a browser window opens):

   ```bash
   .venv\Scripts\python -m scripts.google_auth
   ```

4. **Have your sign-in key ready.** It's in `secrets\lead_api_key.txt` (or wherever you saved it).

5. **Optional: an analyst account** to show that roles work. Run this and save the key it prints:

   ```bash
   .venv\Scripts\python -m scripts.create_user --name "Demo Analyst" --role analyst
   ```

## Start the app (2 terminals)

**Terminal 1, the backend.** Wait until it prints `Application startup complete` (about 20 seconds):

```bash
.venv\Scripts\python -m uvicorn app.api.main:app --port 8000
```

**Terminal 2, the web app** (click the + in the terminal panel for a second one). It opens your browser:

```bash
.venv\Scripts\python -m streamlit run ui/streamlit_app.py
```

If the browser doesn't open, go to http://localhost:8501.

**Sign in:** paste your key into **API key** in the left sidebar and click **Sign in**. The sidebar should then show:
- **Backend connected**
- MCP servers `aml-data`, `tavily`, `google-calendar` and `slack`, each with a tool count
- Your name and role (lead)

If a server shows an error or "Backend not reachable", see Troubleshooting below.

## The demo (about 10 minutes)

Click the example buttons on the page, or type the questions. After each answer, open **Tool calls** to show what the assistant did.

| # | Ask | What to point out |
|---|---|---|
| 1 | "How long does a bank have to file a SAR?" | Answer comes from the bank's policy documents, with a citation [1] |
| 2 | "Who owns account A001520? Is their cash activity suspicious?" | It finds the owner, runs the structuring check and finds cash deposits just under $10,000 at different branches. Open **Tool calls** |
| 3 | "Review customer C00001. Is there suspicious activity?" | An ordinary customer: it does not accuse innocent people |
| 4 | Click "Are there accounts structuring cash deposits that have no alert yet?" | It finds cases the bank's alert system missed |
| 5 | Click "Any recent FinCEN enforcement actions involving structuring? Search the web." | Live web search with links |
| 6 | Click "Escalate alert AL-0002 to the compliance Slack channel with a short summary." | An **Approve / Reject** card appears: nothing is sent until a person approves. Click **Reject** (Approve really posts to Slack) |
| 7 | Click "Book a 30-minute case review for alert AL-0002 tomorrow afternoon." | Same approval card. Approve only if you want the meeting in your calendar |
| 8 | Click the **trace link** under any answer | Langfuse shows every step, and customer names appear as `<NAME_1>`: the AI company never sees them |
| 9 | Turn on **Show audit log** in the sidebar | Every question and decision is recorded; the check shows the log hasn't been tampered with |

**Optional, roles (2 minutes):** click **Sign out**, sign in with the Demo Analyst key, and ask "Escalate alert AL-0002 to Slack". The assistant says only a senior analyst or lead can escalate, and offers a draft instead.

## What to say

**In one sentence:** "An AI assistant for anti-money-laundering analysts that gathers the evidence, runs the expert checks and finds the rules in seconds, while a person makes every decision and approves every action."

The full explanation, likely questions and answers, and the architecture diagram are in the project report (the doc "AML & Fraud Assistant — Project Report").

## After the demo

Press **Ctrl+C** in both terminals to stop the app.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Backend not reachable" | Terminal 1 isn't running or hasn't finished starting. Start it and wait for `Application startup complete` |
| "That key isn't valid." | Copy the key again from `secrets\lead_api_key.txt`, with no spaces. Lost it? Create another lead with `.venv\Scripts\python -m scripts.create_user --name "Your Name" --role lead` |
| Investigation questions fail, or `aml-data` shows an error | The Neo4j or Qdrant free database may be paused after days without use. Log in to console.neo4j.io or cloud.qdrant.io and resume it |
| Answers are slow, or say a fallback happened | Gemini's free quota is busy or used up; Groq answers instead. It still works, just a little slower |
| Calendar booking fails with "sign-in expired" | Run `.venv\Scripts\python -m scripts.google_auth`, then restart Terminal 1 |
| "Address already in use" on port 8000 or 8501 | The app is already running in another terminal: use that one, or close it first |
