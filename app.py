import os
import tempfile
import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

st.title("📄 RAG-Powered Document Q&A System")
st.markdown("Upload a PDF and ask questions about its content. Powered by LangChain, ChromaDB, and Llama 3 via Ollama.")

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document... this may take a minute."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                pipeline = RAGPipeline()
                num_chunks = pipeline.process_document(tmp_path)

                st.session_state.rag_pipeline = pipeline
                st.session_state.document_processed = True
                st.session_state.chat_history = []

                os.unlink(tmp_path)

            st.success(f"Document processed into {num_chunks} chunks. Ask away!")

    st.divider()
    st.markdown("**Tech stack:** LangChain · ChromaDB · HuggingFace Embeddings · Llama 3 (Ollama) · Streamlit")

if st.session_state.document_processed:
    st.subheader("Ask a question about your document")
    question = st.text_input("Your question:", placeholder="What is this document about?")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            result = st.session_state.rag_pipeline.query(question)

        st.session_state.chat_history.append(
            {"question": question, "answer": result["answer"], "sources": result["sources"]}
        )

    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {entry['question']}**")
        st.write(entry["answer"])
        with st.expander("View source chunks"):
            for i, doc in enumerate(entry["sources"]):
                st.markdown(f"**Source {i+1}** (page {doc.metadata.get('page', 'N/A')})")
                st.text(doc.page_content[:500] + "...")
        st.divider()
else:
    st.info("Upload and process a PDF from the sidebar to get started.")