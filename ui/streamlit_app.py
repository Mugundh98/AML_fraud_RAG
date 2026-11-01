import json
import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AML & Fraud Assistant", page_icon="🛡️", layout="wide")


def api(method: str, path: str, **kwargs) -> dict:
    # LLM calls and re-indexing can take a while on a laptop CPU.
    headers = {"X-API-Key": st.session_state["api_key"]} if st.session_state.get("api_key") else {}
    response = httpx.request(method, f"{API_URL}{path}", timeout=300, headers=headers, **kwargs)
    response.raise_for_status()
    return response.json()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            where = " › ".join(part for part in (s["title"], s["section"]) if part)
            if s["page"]:
                where += f" · page {s['page']}"
            st.markdown(f"**[{s['number']}] {where}**  \n`{s['source']}`")
            st.caption(s["text"][:600] + ("…" if len(s["text"]) > 600 else ""))


def render_tool_calls(steps: list[dict]) -> None:
    if not steps:
        return
    rounds = max(s["round"] for s in steps)
    with st.expander(f"Tool calls ({len(steps)} in {rounds} round{'s' if rounds > 1 else ''})"):
        for s in steps:
            args = ", ".join(f"{k}={v!r}" for k, v in s["arguments"].items())
            icon = "⏸️" if s.get("needs_approval") else "✅" if s["ok"] else "⚠️"
            st.markdown(f"{icon} **{s['round']}.** `{s['tool']}({args})` · {s['duration_ms']} ms")
            st.caption(s["preview"])


ACTION_LABELS = {"post_message": "Post to Slack", "create_event": "Create a calendar event"}
STATUS_TEXT = {
    "done": "✅ Approved and done",
    "failed": "⚠️ Approved, but it failed",
    "rejected": "❌ Rejected, nothing was done",
    "expired": "⌛ Expired or already decided, nothing was done",
    "denied": "🔒 Your role can't approve this; it's still waiting for someone who can",
}


def render_actions(message: dict) -> None:
    """Approve/reject cards for actions the agent proposed. Nothing runs until the analyst clicks Approve."""
    outcomes = message.setdefault("outcomes", {})
    for action in message.get("pending_actions", []):
        label = ACTION_LABELS.get(action["tool"], action["tool"])
        with st.container(border=True):
            st.markdown(f"**{label}** · needs your approval · `{action['server']}`")
            for key, value in action["arguments"].items():
                if isinstance(value, str) and ("\n" in value or len(value) > 80):
                    st.markdown(f"**{key}:**")
                    st.text(value)
                else:
                    st.markdown(f"**{key}:** {', '.join(value) if isinstance(value, list) else value}")
            outcome = outcomes.get(action["id"])
            if outcome and outcome["status"] == "denied":
                st.markdown(STATUS_TEXT["denied"])
                continue
            if outcome:
                st.markdown(STATUS_TEXT[outcome["status"]])
                link = _link(outcome["result"])
                if link:
                    st.markdown(f"[Open it]({link})")
                elif outcome["status"] == "failed":
                    st.caption(outcome["result"])
                continue
            approve, reject = st.columns(2)
            for column, approved, text in ((approve, True, "Approve"), (reject, False, "Reject")):
                if column.button(text, key=f"{action['id']}-{text}", type="primary" if approved else "secondary",
                                 width="stretch"):
                    outcome = api("POST", f"/actions/{action['id']}",
                                  json={"approve": approved, "session_id": st.session_state.session_id})
                    outcomes[action["id"]] = outcome
                    # Recorded in the conversation, so the agent knows on the next question what happened.
                    message["content"] += f"\n\n_{label}: {STATUS_TEXT[outcome['status']].split(' ', 1)[1].lower()}._"
                    st.rerun()


def _link(result: str) -> str | None:
    try:
        return json.loads(result).get("link")
    except (ValueError, AttributeError):
        return None


def render_meta(message: dict) -> None:
    if message.get("provider") and message["provider"] != "none":
        st.caption(f"Answered by {message['provider']} · {message['model']}")
    if message.get("fallback_reasons"):
        reasons = [r if len(r) <= 90 else r[:90] + "…" for r in message["fallback_reasons"]]
        st.caption("Fallbacks and retries: " + " · ".join(reasons))
    if message.get("trace_url"):
        st.caption(f"[Trace in Langfuse]({message['trace_url']})")


def sign_in(auth_mode: str) -> dict | None:
    """Sidebar sign-in. Returns the signed-in user, or None (and shows the form) if there isn't one."""
    if st.session_state.get("user"):
        return st.session_state.user
    if auth_mode == "open":
        st.session_state.user = api("GET", "/me")
        return st.session_state.user
    key = st.text_input("API key", type="password", help="Ask your lead, or run `python -m scripts.create_user`.")
    if st.button("Sign in", width="stretch") and key:
        st.session_state.api_key = key
        try:
            st.session_state.user = api("GET", "/me")
            st.rerun()
        except httpx.HTTPStatusError:
            st.session_state.api_key = None
            st.error("That key isn't valid.")
    return None


def render_audit() -> None:
    st.subheader("Audit log")
    check = api("GET", "/audit/verify")
    if check["ok"]:
        st.success(f"Hash chain intact: {check['entries']} entries. Latest hash `{(check['latest_hash'] or '')[:16]}…`")
    else:
        st.error(f"Hash chain broken at entry {check['broken_at']} ({check['reason']}).")
    rows = api("GET", "/audit", params={"limit": 100})
    st.dataframe(
        [{"id": r["id"], "time (UTC)": r["ts"][:19].replace("T", " "), "who": f"{r['actor_id']} ({r['actor_role']})",
          "event": r["event"], "details": json.dumps(r["details"])[:200]} for r in rows],
        width="stretch", hide_index=True,
    )


with st.sidebar:
    st.header("System")
    try:
        health = api("GET", "/health")
        st.success("Backend connected")
        st.write("**LLM providers:**", ", ".join(health["llm_providers"]) or "none configured")
        st.write("**Vector store:**", health["vector_store"])
        if health["indexed_chunks"] is not None:
            st.write("**Indexed chunks:**", health["indexed_chunks"])
        st.write("**Tracing:**", "Langfuse" if health["tracing"] else "off")
        st.write("**PII:**", "pseudonymized for the LLM and traces" if health["pii_pseudonymization"] else "sent as is")
        for name, server in health["mcp_servers"].items():
            if server["error"]:
                st.error(f"MCP `{name}`: {server['error']}")
            else:
                approval = f", {len(server['approval_tools'])} need approval" if server["approval_tools"] else ""
                st.write(f"**MCP `{name}`:** {len(server['tools'])} tools{approval}")
    except httpx.HTTPError:
        st.error(f"Backend not reachable at {API_URL}. Start it with `uvicorn app.api.main:app`.")
        st.stop()

    st.header("You")
    user = sign_in(health["auth"])
    if user is None:
        st.stop()
    st.write(f"**{user['name']}** · {user['role'].replace('_', ' ')}")
    if user["approvable_tools"]:
        st.caption("You can approve: " + ", ".join(user["approvable_tools"]))
    if health["auth"] == "open":
        st.caption("Open mode: no users file, so everyone is a local lead.")
    elif st.button("Sign out", width="stretch"):
        st.session_state.clear()
        st.rerun()
    if user["can_read_audit"]:
        st.toggle("Show audit log", key="show_audit")

    if st.button("Re-index documents", width="stretch"):
        try:
            with st.spinner("Embedding documents…"):
                stats = api("POST", "/ingest")
            st.success(f"Indexed {stats['chunks']} chunks from {stats['files']} files.")
        except httpx.HTTPStatusError as exc:
            st.error(exc.response.json().get("detail", str(exc)))

    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())  # a new conversation is a new Langfuse session
        st.rerun()

if st.session_state.get("show_audit"):
    render_audit()
    st.stop()

st.title("AML & Fraud Assistant")
st.caption("An agent that uses MCP tools: policy documents, the transaction graph, the web, your calendar and "
           "Slack. Anything that changes something waits for your approval. Bank data is synthetic.")

EXAMPLES = [
    "Which open alerts are High priority?",
    "Are there accounts structuring cash deposits that have no alert yet?",
    "Investigate the circular flow alert. Does our policy require escalation?",
    "Any recent FinCEN enforcement actions involving structuring? Search the web.",
    "Escalate alert AL-0002 to the compliance Slack channel with a short summary.",
    "Book a 30-minute case review for alert AL-0002 tomorrow afternoon.",
]

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_meta(message)
            render_tool_calls(message.get("tool_calls", []))
            render_sources(message.get("sources", []))
            render_actions(message)

if not st.session_state.messages:
    st.markdown("**Try:**")
    for example in EXAMPLES:
        if st.button(example):
            st.session_state.pending = example
            st.rerun()

question = st.chat_input("Ask about alerts, customers, transactions, typologies, SAR rules…")
question = question or st.session_state.pop("pending", None)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Working: the agent may call several tools…"):
            try:
                result = api("POST", "/chat", json={"message": question, "history": history,
                                                    "session_id": st.session_state.session_id})
            except httpx.HTTPError as exc:
                result = {"answer": f"Request failed: {exc}", "provider": "none", "model": "", "sources": []}
    st.session_state.messages.append({**result, "role": "assistant", "content": result["answer"]})
    st.rerun()  # the history loop above renders the answer, including any approval buttons
