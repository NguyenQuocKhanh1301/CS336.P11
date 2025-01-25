import streamlit as st
from io import BytesIO
from typing import List
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from brain import *

# Function to extract text from a PDF
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answered" not in st.session_state:
    st.session_state.answered = False

st.title("RAG with LangChain and Streamlit")
st.write("Upload a PDF file and ask questions based on its content.")

# File uploader for PDF
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file is not None and not st.session_state.get("pdf_processed", False):
    # Extract text from the uploaded PDF
    with st.spinner("Extracting text from PDF..."):
        st.session_state.pdf_text = extract_text_from_pdf(pdf_file)
        # st.session_state.db = FAISS.load_local("./faiss", creat_embedding(), allow_dangerous_deserialization=True)
        st.session_state.db = create_retriever_with_chunks(st.session_state.pdf_text)
        st.session_state.retriever = st.session_state.db.as_retriever()
        st.session_state.pdf_processed = True
        st.success("Vector database created successfully!")

if st.session_state.get("pdf_processed", False):
    # Choose the method
    method = st.radio("Choose a method for answering questions:", ("RAG chunking", "RAG LLM"))
    st.write("Type 'bye' to stop asking questions.")

    # Get user input
    question = st.text_input("Ask a question about the PDF content:")

    # If a question is asked and not "bye"
    if question and question.lower() != "bye":
        if method == "RAG chunking":
            with st.spinner("Generating answer..."):
                results = st.session_state.retriever.invoke(question, similarity_top_k=3)
                answer = " ".join([doc.page_content for doc in results])
                st.session_state.questions.append((question, answer))
        elif method == "RAG LLM":
            with st.spinner("Generating answer..."):
                answer = use_llm_for_answering(question, st.session_state.db)
                st.session_state.questions.append((question, answer))

    # Show the conversation history
    if st.session_state.questions:
        st.write("### Conversation History")
        for q, a in st.session_state.questions:
            st.write(f"**You:** {q}")
            st.write(f"**Bot:** {a}")

    # If "bye" is typed
    if question.lower() == "bye":
        st.write("Chúc bạn một ngày tốt lành!")
