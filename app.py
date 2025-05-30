import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# ------------------- Load Environment Variables -------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("Website Chatbot")

# ------------------- Helper: Get All Internal Links -------------------
def get_all_links(start_url, max_pages=30):
    visited = set()
    to_visit = [start_url]
    domain = urlparse(start_url).netloc

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        try:
            response = requests.get(current_url, timeout=5)
            soup = BeautifulSoup(response.content, "html.parser")
            visited.add(current_url)

            for link_tag in soup.find_all("a", href=True):
                href = link_tag["href"]
                full_url = urljoin(start_url, href)
                if urlparse(full_url).netloc == domain and full_url not in visited:
                    to_visit.append(full_url)

        except Exception as e:
            print(f"Failed to fetch {current_url}: {e}")
            continue

    return list(visited)

# ------------------- Streamlit UI -------------------
website_url = st.text_input("🔗 Enter a website URL:", placeholder="https://example.com")

@st.cache_resource(show_spinner="🔍 Crawling website and indexing content...")
def create_qa_chain(url):
    all_links = get_all_links(url, max_pages=30)  # Still crawling subpages but no message shown
    
    all_docs = []
    for link in all_links:
        try:
            loader = WebBaseLoader(link)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            # Optional: you can log or show warnings here if you want
            continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    texts = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)

    llm = ChatGroq(
        temperature=0.1,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    return qa_chain

if website_url:
    qa_chain = create_qa_chain(website_url)

    question = st.text_input("❓ Ask a question about the website:")
    if question:
        with st.spinner("Thinking..."):
            result = qa_chain(question)
            st.markdown("### 💬 Answer")
            st.write(result["result"])

            st.markdown("---")
            st.markdown("### 📄 Source(s):")
            for doc in result["source_documents"]:
                st.markdown(f"- {doc.metadata.get('source', 'Unknown')}")
