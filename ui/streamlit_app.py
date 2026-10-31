import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AML & Fraud Assistant", page_icon="🛡️", layout="wide")


def api(method: str, path: str, **kwargs) -> dict:
    # LLM calls and re-indexing can take a while on a laptop CPU.
    response = httpx.request(method, f"{API_URL}{path}", timeout=180, **kwargs)
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


def render_meta(message: dict) -> None:
    if message.get("provider") and message["provider"] != "none":
        st.caption(f"Answered by {message['provider']} · {message['model']}")
    if message.get("fallback_reasons"):
        reasons = [r if len(r) <= 90 else r[:90] + "…" for r in message["fallback_reasons"]]
        st.caption("Skipped: " + " · ".join(reasons))


with st.sidebar:
    st.header("System")
    try:
        health = api("GET", "/health")
        st.success("Backend connected")
        st.write("**LLM providers:**", ", ".join(health["llm_providers"]) or "none configured")
        st.write("**Vector store:**", health["vector_store"])
        st.write("**Indexed chunks:**", health["indexed_chunks"])
    except httpx.HTTPError:
        st.error(f"Backend not reachable at {API_URL}. Start it with `uvicorn app.api.main:app`.")
        st.stop()

    if st.button("Re-index documents", width="stretch"):
        with st.spinner("Embedding documents…"):
            stats = api("POST", "/ingest")
        st.success(f"Indexed {stats['chunks']} chunks from {stats['files']} files.")
        st.rerun()

    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

st.title("AML & Fraud Assistant")
st.caption("Answers come from the indexed knowledge base, with numbered citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_meta(message)
            render_sources(message.get("sources", []))

if question := st.chat_input("Ask about AML typologies, SAR rules, fraud red flags…"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking…"):
            try:
                result = api("POST", "/chat", json={"message": question, "history": history})
            except httpx.HTTPError as exc:
                result = {"answer": f"Request failed: {exc}", "provider": "none", "model": "", "sources": []}
        message = {"role": "assistant", "content": result["answer"], **result}
        st.markdown(message["content"])
        render_meta(message)
        render_sources(message.get("sources", []))
    st.session_state.messages.append(message)
