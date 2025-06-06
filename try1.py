import streamlit as st
import pandas as pd
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        return df
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

def dataframe_to_docs(df):
    documents = []
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(Document(page_content=content))
    return documents

def create_qa_chain(df):
    documents = dataframe_to_docs(df)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key="AqoChT9cmkAzALwMLdWT3BlbkFJcNHsH5Z5LN2lxPcDAop")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_store.as_retriever()

    llm = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
        api_key= "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"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

st.title("📊 RAG Chatbot on Uploaded Data")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = load_file(uploaded_file)
    if df is not None:
        st.write("✅ File uploaded successfully.")
        st.dataframe(df.head())

        with st.spinner("Indexing and loading RAG model..."):
            qa_chain = create_qa_chain(df)

        st.success("Ask your questions below 👇")

        query = st.text_input("🔍 Ask a question about your data:")

        if query:
            with st.spinner("Generating answer..."):
                result = qa_chain(query)
                st.markdown("### 🧠 Answer")
                st.write(result["result"])

                st.markdown("### 📚 Retrieved Context (Chunks)")
                for doc in result["source_documents"]:
                    st.code(doc.page_content)

