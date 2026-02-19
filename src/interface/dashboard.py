"""
CheckPrioritizer: Research Dashboard
A specialized interface for analyzing claim detection and check-worthiness.
This module connects the user to the RAG engine for real-time triage.
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="CheckPrioritizer", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="ViewTransitions"] {
        font-size: 20px;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-size: 32px !important;
        margin-bottom: 20px !important;
    }

    [data-testid="stSidebar"] .stAlert p {
        font-size: 22px !important;
    }

    [data-testid="stSidebar"] .stWidgetLabel p {
        font-size: 24px !important;
        font-weight: bold !important;
    }

    [data-testid="stSidebar"] .stCaptionContainer p {
        font-size: 18px !important;
        font-style: italic;
    }

    [data-testid="stNotification"] p {
        font-size: 28px !important;
        line-height: 1.4 !important;
        font-weight: 500 !important;
    }

    [data-testid="stChatMessageContent"] p {
        font-size: 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.info("Dataset: CLEF CheckThat! (2022)")

    st.divider()
    st.subheader("Search Controls")
    source_count = st.slider("Retrieval Depth (Sources)", 1, 5, 3)

    st.divider()
    st.caption("Powered by: Llama 3 & ChromaDB")

st.title("🛡️ CheckPrioritizer: Fact-Checking Assistant")

st.info(
    "💡 **Challenge the Engine:** Paste a claim below to determine if it warrants \
    a full investigation based on our research database."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Paste a claim or ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing claim priority..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"query": prompt, "count": source_count},
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                    if data.get("sources"):
                        with st.expander("🔍 Deep Dive: Evidence"):
                            cols = st.columns(len(data["sources"]))
                            for i, source in enumerate(data["sources"]):
                                with cols[i]:
                                    st.markdown(f"**Source {i + 1}**")
                                    label = source["metadata"].get(
                                        "class_label", "Claim"
                                    )
                                    st.success(f"Tag: {label}")
                                    st.caption(source["text"][:200] + "...")
                else:
                    st.error("Engine Error: The API is unreachable.")

            except requests.exceptions.ConnectionError:
                st.error("Offline: Ensure the FastAPI server is running on port 8000.")
