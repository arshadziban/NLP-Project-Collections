import os
import streamlit as st
import pickle
import time

from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

#  Set Page Config
st.set_page_config(page_title="News Research Tool", layout="wide")

#  Page Title
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📰 News Research Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste a news article URL, ask a question, and get intelligent answers with source references.</p>", unsafe_allow_html=True)
st.markdown("---")

#  Sidebar - URL Input
st.sidebar.header("🔗 Input URL")
url = st.sidebar.text_input("Enter a News Article URL")
urls = [url] if url else []
process_url_clicked = st.sidebar.button("🚀 Process Article")

#  File Path to Save FAISS
file_path = r"D:\Skills\NLP-Project-Collections\Gen Ai Project\News Research Tool\faiss_store_openai.pkl"

#  Load LLM
llm = OpenAI(temperature=0.9, max_tokens=500)

#  Main Section Placeholder
main_placeholder = st.empty()

#  Process Article
if process_url_clicked:
    if not url:
        st.sidebar.warning("⚠️ Please enter a URL before clicking Process.")
    else:
        with st.spinner("🔄 Loading and processing the article..."):
            try:
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()

                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                docs = text_splitter.split_documents(data)

                # Create embeddings
                embeddings = OpenAIEmbeddings()
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                time.sleep(1)

                # Save FAISS index
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_openai, f)

                st.success("✅ Article processed and embedded successfully!")
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")

# Question Input
st.markdown("### ❓ Ask a Question")
query = st.text_input("Type your question here...")

# Answer Section
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            with st.spinner("🤖 Thinking..."):
                try:
                    result = chain({"question": query}, return_only_outputs=True)

                    st.markdown("## 🧠 Answer")
                    st.success(result["answer"])

                    sources = result.get("sources", "")
                    if sources:
                        with st.expander("🔗 View Sources"):
                            for source in sources.split("\n"):
                                st.markdown(f"- {source}")
                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")
    else:
        st.warning("⚠️ No article processed yet. Please input and process a URL first.")
