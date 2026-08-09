import streamlit as st

from foundry_local_sdk import Configuration, FoundryLocalManager

from src.chat import get_chat_model, answer_query
from src.embeddings import get_embedding_model


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background: #f8fafc;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 6rem;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.8rem;
}


/* Hero */

.hero-card {
    background:
        linear-gradient(
            135deg,
            rgba(79, 70, 229, 0.09),
            rgba(14, 165, 233, 0.06)
        );
    border: 1px solid #dbeafe;
    border-radius: 24px;
    padding: 42px;
    margin-bottom: 28px;
    box-shadow:
        0 12px 40px rgba(15, 23, 42, 0.06);
}

.hero-badge {
    display: inline-block;
    background: #ffffff;
    border: 1px solid #dbeafe;
    color: #4f46e5;
    padding: 6px 11px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 17px;
}

.hero-title {
    color: #0f172a;
    font-size: 2.6rem;
    font-weight: 780;
    letter-spacing: -0.045em;
    margin-bottom: 10px;
    line-height: 1.1;
}

.hero-subtitle {
    color: #64748b;
    font-size: 1rem;
    line-height: 1.7;
    max-width: 720px;
}


/* Feature cards */

.feature-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px;
    min-height: 155px;
    box-shadow:
        0 5px 18px rgba(15, 23, 42, 0.035);
}

.feature-icon {
    font-size: 1.4rem;
    margin-bottom: 14px;
}

.feature-title {
    color: #0f172a;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.feature-description {
    color: #64748b;
    font-size: 0.87rem;
    line-height: 1.55;
}


/* Chat */

div[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 15px 18px;
    margin-bottom: 13px;
    box-shadow:
        0 4px 18px rgba(15, 23, 42, 0.04);
}

div[data-testid="stChatInput"] textarea {
    border-radius: 16px !important;
    font-size: 0.95rem !important;
}


/* Expander */

details {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 5px 10px;
}


/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid #cbd5e1;
    background: #ffffff;
    color: #0f172a;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #6366f1;
    color: #4f46e5;
    background: #eef2ff;
}


/* Footer */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.76rem;
    margin-top: 35px;
    padding-top: 20px;
    border-top: 1px solid #e2e8f0;
}


/* Hide Streamlit extras */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# INITIALIZE RAG
# =========================================================

@st.cache_resource
def initialize_rag():

    manager = FoundryLocalManager.instance

    if manager is None:
        config = Configuration(
            app_name="foundry_local_rag"
        )

        FoundryLocalManager.initialize(config)

        manager = FoundryLocalManager.instance

    if manager is None:
        raise RuntimeError(
            "FoundryLocalManager could not be initialized."
        )

    embedding_model = get_embedding_model(
        manager
    )

    embedding_client = (
        embedding_model.get_embedding_client()
    )

    chat_model = get_chat_model(
        manager
    )

    chat_client = (
        chat_model.get_chat_client()
    )

    return (
        embedding_model,
        embedding_client,
        chat_model,
        chat_client,
    )


# =========================================================
# LOAD MODELS
# =========================================================

with st.spinner("Preparing Local RAG..."):

    (
        embedding_model,
        embedding_client,
        chat_model,
        chat_client,
    ) = initialize_rag()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ◉ Local RAG")

    st.caption(
        "Local document intelligence powered by "
        "Microsoft Foundry Local."
    )

    st.divider()

    st.caption("EMBEDDING MODEL")
    st.markdown(
        "**qwen3-embedding-0.6b**"
    )

    st.markdown("")

    st.caption("CHAT MODEL")
    st.markdown(
        "**qwen2.5-0.5b**"
    )

    st.markdown("")

    st.caption("SIMILARITY THRESHOLD")
    st.markdown(
        "**0.40**"
    )

    st.markdown("")

    st.success(
        "● Local models ready"
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="hero-card">

<div class="hero-badge">
LOCAL AI • RAG
</div>

<div class="hero-title">
Local RAG Assistant
</div>

<div class="hero-subtitle">
Ask questions about your local knowledge base.
The system retrieves relevant document chunks using
semantic search and generates grounded answers with
a locally running language model.
</div>

</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# EMPTY STATE
# =========================================================

if len(st.session_state.messages) == 0:

    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )

    with col1:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">
📄
</div>

<div class="feature-title">
Document Grounded
</div>

<div class="feature-description">
Answers are generated using information retrieved
from your indexed local documents.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">
🔎
</div>

<div class="feature-title">
Semantic Retrieval
</div>

<div class="feature-description">
Embedding similarity is used to identify the most
relevant document chunks for each question.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">
🔒
</div>

<div class="feature-title">
Local Inference
</div>

<div class="feature-description">
Embedding and chat models run locally using
Microsoft Foundry Local.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        sources = message.get(
            "sources",
            [],
        )

        if sources:

            with st.expander(
                f"View sources ({len(sources)})"
            ):

                for index, source in enumerate(
                    sources,
                    start=1,
                ):

                    source_name = source.get(
                        "source_name",
                        "Unknown source",
                    )

                    chunk_index = source.get(
                        "chunk_index",
                        "N/A",
                    )

                    score = source.get(
                        "score",
                        0,
                    )

                    st.markdown(
                        f"""
**{index}. {source_name}**

Chunk: `{chunk_index}`  
Similarity: `{float(score):.4f}`
"""
                    )

                    if index != len(sources):
                        st.divider()


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(
        user_message
    )

    with st.chat_message("user"):
        st.markdown(question)


    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents..."
        ):

            try:

                answer, sources = answer_query(
                    question,
                    embedding_client,
                    chat_client,
                )

                st.markdown(answer)


                if sources:

                    with st.expander(
                        f"View sources ({len(sources)})"
                    ):

                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):

                            source_name = source.get(
                                "source_name",
                                "Unknown source",
                            )

                            chunk_index = source.get(
                                "chunk_index",
                                "N/A",
                            )

                            score = source.get(
                                "score",
                                0,
                            )

                            st.markdown(
                                f"""
**{index}. {source_name}**

Chunk: `{chunk_index}`  
Similarity: `{float(score):.4f}`
"""
                            )

                            if index != len(sources):
                                st.divider()


            except Exception as error:

                answer = (
                    "I couldn't generate a response. "
                    f"Error: {error}"
                )

                sources = []

                st.error(answer)


    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
    }

    st.session_state.messages.append(
        assistant_message
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
Microsoft Foundry Local
&nbsp;•&nbsp;
SQLite
&nbsp;•&nbsp;
Semantic Retrieval
&nbsp;•&nbsp;
Local RAG
</div>
""",
    unsafe_allow_html=True,
)