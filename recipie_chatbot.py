import time
import uuid
from typing import Literal, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

APP_TITLE = "MasterChef Chat Bot"
TITLE_TYPING_DELAY = 0.03
DEFAULT_ASSISTANT_MESSAGE = (
    "I couldn't generate a recipe response just now. Please try again."
)


class RecipeState(TypedDict):
    user_input: str
    is_recipe_query: bool
    recipe_response: str
    error_message: str


def validate_recipe_query(state: RecipeState) -> RecipeState:
    """
    Check if the user input is related to a recipe query.
    """
    user_input = state["user_input"]

    system_message = """You are a recipe validator. Your task is to determine if the user's query is asking for a recipe or cooking instructions.

    A recipe query typically:
    - Asks how to make/cook/prepare a specific dish
    - Requests ingredients or steps for cooking
    - Mentions cooking methods, kitchen tools, or food preparation

    Respond with ONLY 'True' if it's a recipe query, or 'False' if it's not.

    Examples:
    "How to make Beef Karahi?" -> True
    "What's the weather today?" -> False
    "Tell me a joke" -> False
    "Ingredients for chicken biryani" -> True
    """

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_input),
    ]

    try:
        response = model.invoke(messages)
        state["is_recipe_query"] = response.content.strip().lower() == "true"
        state["error_message"] = ""
    except Exception as exc:
        state["is_recipe_query"] = False
        state["error_message"] = f"Error validating query: {exc}"

    return state


def generate_recipe_response(state: RecipeState) -> RecipeState:
    """
    Generate a recipe response for valid recipe queries.
    """
    user_input = state["user_input"]

    system_message = """You are a MasterChef expert assistant. Provide detailed, professional recipe responses including:

    1. Brief introduction about the dish
    2. List of ingredients with measurements
    3. Step-by-step cooking instructions
    4. Tips and tricks for best results
    5. Serving suggestions

    Make the response engaging and helpful, as if you're teaching someone to cook."""

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_input),
    ]

    try:
        response = model.invoke(messages)
        state["recipe_response"] = response.content
        state["error_message"] = ""
    except Exception as exc:
        state["recipe_response"] = ""
        state["error_message"] = f"Error generating recipe: {exc}"

    return state


def handle_non_recipe_query(state: RecipeState) -> RecipeState:
    """
    Handle queries that are not recipe-related.
    """
    state["recipe_response"] = (
        "Sorry, I can only provide information about recipes and cooking. "
        "Please ask me about how to make a specific dish!"
    )
    state["error_message"] = ""
    return state


def route_query(state: RecipeState) -> Literal["generate_recipe", "non_recipe"]:
    """
    Conditional routing function to determine the next step.
    """
    if state["is_recipe_query"]:
        return "generate_recipe"
    return "non_recipe"


checkpointer = InMemorySaver()

workflow = StateGraph(RecipeState)
workflow.add_node("validate_query", validate_recipe_query)
workflow.add_node("generate_recipe", generate_recipe_response)
workflow.add_node("non_recipe", handle_non_recipe_query)
workflow.set_entry_point("validate_query")
workflow.add_conditional_edges(
    "validate_query",
    route_query,
    {
        "generate_recipe": "generate_recipe",
        "non_recipe": "non_recipe",
    },
)
workflow.add_edge("generate_recipe", END)
workflow.add_edge("non_recipe", END)
chatbot = workflow.compile(checkpointer=checkpointer)


# ─── Session helpers ───────────────────────────────────────────────


def _init_state() -> None:
    defaults = {
        "chat_sessions": {},
        "session_order": [],
        "active_session_id": None,
        "title_done": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if st.session_state["active_session_id"] is None:
        _new_session()


def _new_session() -> str:
    sid = f"chat-{uuid.uuid4().hex[:6]}"
    st.session_state["chat_sessions"][sid] = []
    st.session_state["session_order"].insert(0, sid)
    st.session_state["active_session_id"] = sid
    return sid


def _switch_session(sid: str) -> None:
    st.session_state["active_session_id"] = sid


def _messages() -> list[dict]:
    sid = st.session_state["active_session_id"]
    return st.session_state["chat_sessions"].setdefault(sid, [])


def _add_msg(role: str, content: str) -> None:
    _messages().append({"role": role, "content": content})


def _stream_text(text: str):
    """Yield small word-chunks for st.write_stream."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.03)


# ─── Backend call ──────────────────────────────────────────────────


def _ask_chef(user_input: str) -> str:
    config = {"configurable": {"thread_id": st.session_state["active_session_id"]}}
    try:
        result = chatbot.invoke(
            {
                "user_input": user_input,
                "is_recipe_query": False,
                "recipe_response": "",
                "error_message": "",
            },
            config=config,
        )
    except Exception as exc:
        return f"Something went wrong: {exc}"
    return (
        result.get("recipe_response")
        or result.get("error_message")
        or DEFAULT_ASSISTANT_MESSAGE
    ).strip()


# ─── Page chrome ───────────────────────────────────────────────────


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Page ── */
        .stApp { background: #212121; }
        .block-container { max-width: 820px; padding-top: 1rem; }

        /* ── Header ── */
        .mc-header {
            text-align: center;
            padding: 2rem 1rem 1.2rem;
        }
        .mc-header h1 {
            margin: 0;
            font-size: 2.4rem;
            font-weight: 700;
            color: #ececec;
        }
        .mc-header p {
            margin: 0.4rem 0 0;
            color: #9e9e9e;
            font-size: 0.95rem;
        }
        .mc-cursor {
            display: inline-block;
            width: 2px;
            background: #ececec;
            margin-left: 2px;
            animation: mc-blink 0.8s step-end infinite;
        }
        @keyframes mc-blink { 50% { opacity: 0; } }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: #171717;
        }
        [data-testid="stSidebar"] * {
            color: #d1d1d1 !important;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] > button {
            background: #2a2a2a;
            color: #d1d1d1 !important;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            text-align: left;
            padding: 0.55rem 0.75rem;
            font-size: 0.85rem;
            transition: background 0.15s;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
            background: #353535;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
            background: #3b3b3b;
            border-color: #555;
        }
        .sb-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #777 !important;
            margin: 1rem 0 0.4rem;
        }

        /* ── Chat messages ── */
        [data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            padding: 0.4rem 0;
        }
        [data-testid="stChatMessageContent"] {
            background: #2f2f2f !important;
            border: none !important;
            border-radius: 16px !important;
            padding: 0.9rem 1.1rem !important;
            box-shadow: none !important;
        }
        [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessageContent"] li,
        [data-testid="stChatMessageContent"] h1,
        [data-testid="stChatMessageContent"] h2,
        [data-testid="stChatMessageContent"] h3,
        [data-testid="stChatMessageContent"] h4,
        [data-testid="stChatMessageContent"] strong,
        [data-testid="stChatMessageContent"] span {
            color: #e3e3e3 !important;
            line-height: 1.7;
        }
        [data-testid="stChatMessageContent"] ul,
        [data-testid="stChatMessageContent"] ol {
            color: #e3e3e3 !important;
            padding-left: 1.3rem;
        }
        [data-testid="stChatMessageContent"] code {
            background: #3a3a3a !important;
            color: #f0f0f0 !important;
            padding: 0.15rem 0.35rem;
            border-radius: 4px;
        }

        /* ── Chat input ── */
        [data-testid="stChatInput"] {
            background: transparent !important;
            padding-top: 0.5rem;
        }
        [data-testid="stChatInput"] textarea {
            background: #2f2f2f !important;
            color: #e3e3e3 !important;
            border: 1px solid #444 !important;
            border-radius: 12px !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #888 !important;
        }

        /* ── Misc ── */
        .stSpinner > div { color: #aaa !important; }
        .empty-hint {
            text-align: center;
            color: #777;
            padding: 3rem 1rem;
            font-size: 0.95rem;
            line-height: 1.8;
        }
        .empty-hint strong { color: #bbb; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    ph = st.empty()
    if st.session_state["title_done"]:
        ph.markdown(
            '<div class="mc-header"><h1>🍳 MasterChef Chat Bot</h1>'
            "<p>Your personal recipe assistant — ask anything about cooking.</p></div>",
            unsafe_allow_html=True,
        )
        return

    partial = ""
    for ch in APP_TITLE:
        partial += ch
        ph.markdown(
            f'<div class="mc-header"><h1>🍳 {partial}'
            f'<span class="mc-cursor">&nbsp;</span></h1>'
            "<p>Your personal recipe assistant — ask anything about cooking.</p></div>",
            unsafe_allow_html=True,
        )
        time.sleep(TITLE_TYPING_DELAY)

    ph.markdown(
        '<div class="mc-header"><h1>🍳 MasterChef Chat Bot</h1>'
        "<p>Your personal recipe assistant — ask anything about cooking.</p></div>",
        unsafe_allow_html=True,
    )
    st.session_state["title_done"] = True


# ─── Sidebar ───────────────────────────────────────────────────────


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🍳 MasterChef")
        st.button("＋  New Chat", on_click=_new_session, use_container_width=True)
        st.markdown('<p class="sb-label">History</p>', unsafe_allow_html=True)

        for sid in st.session_state["session_order"]:
            msgs = st.session_state["chat_sessions"].get(sid, [])
            preview = msgs[0]["content"].replace("\n", " ")[:30] if msgs else "New chat"
            is_active = sid == st.session_state["active_session_id"]
            st.button(
                f"{'● ' if is_active else ''}{preview}",
                key=f"sb-{sid}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                on_click=_switch_session,
                args=(sid,),
            )


# ─── Chat area ─────────────────────────────────────────────────────


def _render_history() -> None:
    msgs = _messages()
    if not msgs:
        st.markdown(
            "<div class='empty-hint'>"
            "<strong>No messages yet.</strong><br>"
            "Try: <em>How do I make butter chicken?</em><br>"
            "or: <em>Give me a quick pasta recipe</em>"
            "</div>",
            unsafe_allow_html=True,
        )
        return
    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def _handle_input(user_input: str) -> None:
    _add_msg("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Cooking up a response..."):
            reply = _ask_chef(user_input)
        full_text = st.write_stream(_stream_text(reply))

    _add_msg("assistant", full_text if full_text else reply)


# ─── Main ──────────────────────────────────────────────────────────


def run_app() -> None:
    st.set_page_config(
        page_title="MasterChef Chat Bot",
        page_icon="🍳",
        layout="centered",
    )
    _init_state()
    _inject_css()
    _render_sidebar()
    _render_header()
    _render_history()

    if prompt := st.chat_input("Ask me for a recipe..."):
        _handle_input(prompt)


run_app()
