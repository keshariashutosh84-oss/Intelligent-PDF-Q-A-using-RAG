import streamlit as st
from dotenv import load_dotenv
from rag import process_pdf, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄",
    layout="wide"
)

# session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "pages" not in st.session_state:
    st.session_state.pages = 0

if "chunks" not in st.session_state:
    st.session_state.chunks = 0


# sidebar
with st.sidebar:
    st.title("📄 AI PDF Assistant")
    st.caption("Upload a document and start asking questions.")

    st.divider()
    st.subheader("Upload Document")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        st.write("Selected document")
        st.code(uploaded_file.name)

        if st.button("Process Document", type="primary", use_container_width=True):
            try:
                with st.spinner("Processing your document..."):
                    vectorstore, pages, chunks = process_pdf(uploaded_file)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.pages = pages
                    st.session_state.chunks = chunks
                    st.session_state.messages = []

                st.success("Document processed!")

            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.current_file:
        st.divider()
        st.subheader("Document Info")
        st.info(f"📄 {st.session_state.current_file}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Pages", st.session_state.pages)

        with col2:
            st.metric("Chunks", st.session_state.chunks)

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Powered by")
    st.write("LangChain • ChromaDB • Mistral AI")


# main page
st.title("📄 AI PDF Assistant")
st.markdown("#### Chat with your documents using Retrieval-Augmented Generation")
st.caption(
    "Upload a PDF, ask questions, and get answers grounded in your document."
)

st.divider()


# shown before a PDF is processed
if st.session_state.vectorstore is None:
    with st.container(border=True):
        st.subheader("👋 Welcome")
        st.write("Turn your PDF into an interactive knowledge assistant.")
        st.write("Upload a document from the sidebar to get started.")

    st.write("")
    st.subheader("How it works")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.subheader("📤 Upload")
            st.write("Upload any PDF document from your computer.")

    with col2:
        with st.container(border=True):
            st.subheader("⚙️ Process")
            st.write("The document is split and converted into vector embeddings.")

    with col3:
        with st.container(border=True):
            st.subheader("💬 Chat")
            st.write("Ask questions and receive answers based on your PDF.")

    st.write("")

    with st.container(border=True):
        st.subheader("🔎 RAG Powered")
        st.write(
            "Relevant sections of your document are retrieved using semantic "
            "search before the AI generates an answer."
        )

else:
    col1, col2 = st.columns([4, 1])

    with col1:
        st.success(f"📄 Chatting with **{st.session_state.current_file}**")

    with col2:
        st.metric("Pages", st.session_state.pages)


# chat
if st.session_state.vectorstore is not None:

    if not st.session_state.messages:
        with st.container(border=True):
            st.subheader("💬 Start a conversation")
            st.write(
                "Your document is ready. Try asking a question about its content."
            )
            st.caption(
                "Example: Summarize the main topics discussed in this document."
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


query = st.chat_input("Ask something about your document...")

if query:

    if st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF first.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching your document..."):
            try:
                answer, docs = ask_question(
                    st.session_state.vectorstore,
                    query
                )

                st.markdown(answer)

                if docs:
                    with st.expander("📚 View Retrieved Sources"):

                        for i, doc in enumerate(docs, start=1):
                            page = doc.metadata.get("page", "Unknown")

                            if isinstance(page, int):
                                page += 1

                            with st.container(border=True):
                                st.markdown(f"**Source {i} • Page {page}**")
                                st.write(doc.page_content[:500])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error generating response: {e}")