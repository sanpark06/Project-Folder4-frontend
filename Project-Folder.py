# ui/app.py
# Person 4 deliverable: Streamlit UI that integrates with Person 1 (core), Person 2 (retrieval), Person 3 (memory)
# Run: streamlit run ui/app.py

import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st


# -----------------------------
# Integration Contract (IMPORTANT)
# -----------------------------
# This UI expects Person 1 to expose ONE callable:
#   from core.agent_controller import run_agent
#   result = run_agent(query: str, mode: str, session_id: str, user_id: str, preferences: dict)
#
# And it expects the returned result dict to look like:
# {
#   "answer": "...final markdown...",
#   "mode": "quick" | "deep",
#   "clarifying_question": null | "...",
#   "citations": [
#       {"title": "...", "url": "...", "snippet": "...", "score": 0.82}
#   ],
#   "memory": {
#       "used": ["..."],
#       "saved": ["..."]
#   },
#   "metrics": {
#       "latency_s": 12.4,
#       "tokens_in": 1200,
#       "tokens_out": 640,
#       "estimated_cost_usd": 0.012
#   },
#   "debug": {"...": "..."}
# }
#
# Person 2 provides citations list.
# Person 3 provides memory used/saved.
# Person 1 orchestrates all of it.


# -----------------------------
# Safe Import: If core isn't ready yet
# -----------------------------

def _try_import_agent():
    try:
        from core.agent_controller import run_agent  # type: ignore
        return run_agent
    except Exception:
        return None


run_agent = _try_import_agent()


# -----------------------------
# Helpers
# -----------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _pretty_money(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"${x:.4f}"
    except Exception:
        return str(x)


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _render_citations(citations: List[Dict[str, Any]]):
    if not citations:
        st.info("No citations returned.")
        return

    # Sort by score if present
    citations_sorted = sorted(
        citations,
        key=lambda c: float(c.get("score", 0.0) or 0.0),
        reverse=True,
    )

    for i, c in enumerate(citations_sorted, start=1):
        title = c.get("title") or f"Source {i}"
        url = c.get("url") or ""
        snippet = c.get("snippet") or ""
        score = c.get("score", None)

        with st.expander(f"[{i}] {title}"):
            if url:
                st.write(url)
            if score is not None:
                st.caption(f"Relevance score: {score}")
            if snippet:
                st.write(snippet)


def _render_metrics(metrics: Dict[str, Any]):
    latency = metrics.get("latency_s")
    tin = metrics.get("tokens_in")
    tout = metrics.get("tokens_out")
    cost = metrics.get("estimated_cost_usd")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latency", f"{latency:.2f}s" if isinstance(latency, (int, float)) else "—")
    col2.metric("Tokens In", str(tin) if tin is not None else "—")
    col3.metric("Tokens Out", str(tout) if tout is not None else "—")
    col4.metric("Est. Cost", _pretty_money(cost))


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(
    page_title="HackGeek – Technical Deep Research Agent",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 HackGeek – Technical & Coding Deep Research Agent")
st.caption("Quick Mode (<2 min) • Deep Mode (<10 min) • Memory • Citations • Cost & Latency")


# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{_now_ms()}"

if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_user"

if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {role, content, meta}

if "preferences" not in st.session_state:
    st.session_state.preferences = {
        "prefer_code_examples": True,
        "answer_style": "structured",
        "citation_level": "high",
    }


# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")

    mode = st.radio(
        "Research Mode",
        ["quick", "deep"],
        index=0,
        help="Quick = fast high-signal. Deep = multi-source synthesis.",
    )

    st.subheader("🧩 Preferences (Memory)")
    st.session_state.preferences["prefer_code_examples"] = st.checkbox(
        "Prefer code examples",
        value=st.session_state.preferences.get("prefer_code_examples", True),
    )

    st.session_state.preferences["answer_style"] = st.selectbox(
        "Answer style",
        ["structured", "bullet", "concise", "detailed"],
        index=["structured", "bullet", "concise", "detailed"].index(
            st.session_state.preferences.get("answer_style", "structured")
        ),
    )

    st.session_state.preferences["citation_level"] = st.selectbox(
        "Citation density",
        ["low", "medium", "high"],
        index=["low", "medium", "high"].index(
            st.session_state.preferences.get("citation_level", "high")
        ),
    )

    st.divider()
    st.subheader("🧪 Debug")
    show_debug = st.checkbox("Show debug payload", value=False)

    if st.button("🧹 Clear chat"):
        st.session_state.chat = []
        st.session_state.session_id = f"sess_{_now_ms()}"
        st.rerun()


# Main layout
left, right = st.columns([1.35, 1])


with left:
    st.subheader("💬 Chat")

    # Render chat
    for msg in st.session_state.chat:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        meta = msg.get("meta", {})

        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)

            # Optional: show metrics under each assistant message
            metrics = meta.get("metrics") or {}
            if metrics:
                with st.container():
                    _render_metrics(metrics)


with right:
    st.subheader("🧾 Sources & Memory")

    # Last assistant payload
    last_assistant = None
    for msg in reversed(st.session_state.chat):
        if msg.get("role") == "assistant" and msg.get("meta"):
            last_assistant = msg
            break

    if last_assistant:
        citations = _safe_get(last_assistant, ["meta", "citations"], []) or []
        memory_used = _safe_get(last_assistant, ["meta", "memory", "used"], []) or []
        memory_saved = _safe_get(last_assistant, ["meta", "memory", "saved"], []) or []

        st.markdown("### 🔗 Citations")
        _render_citations(citations)

        st.markdown("### 🧠 Memory")
        if memory_used:
            st.success("Memory used in this answer:")
            st.write(memory_used)
        else:
            st.info("No memory used.")

        if memory_saved:
            st.success("New memory saved:")
            st.write(memory_saved)

        if show_debug:
            st.markdown("### 🧪 Debug")
            st.code(json.dumps(last_assistant.get("meta", {}), indent=2), language="json")

    else:
        st.info("Ask a question to see citations + memory here.")


# Input
query = st.chat_input("Ask a technical question… (e.g., 'Deep dive on RAG chunking strategies')")

if query:
    # Save user message
    st.session_state.chat.append({"role": "user", "content": query, "meta": {}})

    # If core agent isn't ready, provide a placeholder response
    if run_agent is None:
        placeholder = (
            "⚠️ Core agent not wired yet.\n\n"
            "Person 1 must expose: `core/agent_controller.py` with `run_agent(...)`.\n\n"
            "For now, this is a UI scaffold that will integrate cleanly once core is merged."
        )
        st.session_state.chat.append(
            {
                "role": "assistant",
                "content": placeholder,
                "meta": {
                    "citations": [],
                    "memory": {"used": [], "saved": []},
                    "metrics": {"latency_s": 0.0, "tokens_in": 0, "tokens_out": 0, "estimated_cost_usd": 0.0},
                },
            }
        )
        st.rerun()

    # Call the agent
    start = time.time()

    with st.spinner("Researching…"):
        try:
            result = run_agent(
                query=query,
                mode=mode,
                session_id=st.session_state.session_id,
                user_id=st.session_state.user_id,
                preferences=st.session_state.preferences,
            )
        except Exception as e:
            result = {
                "answer": f"❌ Agent error: {e}",
                "mode": mode,
                "clarifying_question": None,
                "citations": [],
                "memory": {"used": [], "saved": []},
                "metrics": {
                    "latency_s": round(time.time() - start, 2),
                    "tokens_in": None,
                    "tokens_out": None,
                    "estimated_cost_usd": None,
                },
                "debug": {"exception": str(e)},
            }

    # Clarifying question UX
    clarifying_q = result.get("clarifying_question")
    answer = result.get("answer") or ""

    if clarifying_q and not answer.strip():
        answer = f"🤔 I need one clarification before I proceed:\n\n**{clarifying_q}**"

    # Ensure metrics includes UI-measured latency if missing
    metrics = result.get("metrics") or {}
    if metrics.get("latency_s") is None:
        metrics["latency_s"] = round(time.time() - start, 2)
    result["metrics"] = metrics

    # Save assistant message
    st.session_state.chat.append(
        {
            "role": "assistant",
            "content": answer,
            "meta": {
                "citations": result.get("citations", []),
                "memory": result.get("memory", {}),
                "metrics": result.get("metrics", {}),
                "debug": result.get("debug", {}),
            },
        }
    )

    st.rerun()