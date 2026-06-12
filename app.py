__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import json
import re
from pypdf import PdfReader
from docx import Document
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- App Configuration ---
st.set_page_config(page_title="AI Career Pilot", page_icon="🚀", layout="wide")

def extract_text_from_file(file):
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        return " ".join([page.extract_text() for page in pdf.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    return file.getvalue().decode("utf-8")

def clean_json_response(text):
    # Extract only the content between [ ] or { }
    match = re.search(r'(\[.*?\]|\{.*?\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return [text]

# --- Sidebar: Setup ---
with st.sidebar:
    st.title("🛠️ Configuration")
    hf_token = st.text_input("Hugging Face Token", type="password", value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))
    st.info("This app uses Qwen2.5-7B on the HF Free Inference API.")

# --- Main UI ---
st.title("🚀 AI Career Pilot: End-to-End RAG")
st.markdown("Tailor your resume and cover letter to any job description using Retrieval-Augmented Generation.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Your Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    resume_text = ""
    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)
        st.success("Resume Loaded!")

with col2:
    st.subheader("💼 Job Description")
    jd_text = st.text_area("Paste the JD here...", height=200)

if st.button("✨ Generate Tailored Application", type="primary"):
    if not hf_token or not resume_text or not jd_text:
        st.error("Please provide the API token, your resume, and the job description.")
    else:
        with st.spinner("🤖 AI is analyzing and tailoring..."):
            llm_base = HuggingFaceEndpoint(
                repo_id="Qwen/Qwen2.5-7B-Instruct",
                huggingfacehub_api_token=hf_token,
                temperature=0.1,
                max_new_tokens=1000,
                task="conversational"
            )
            llm = ChatHuggingFace(llm=llm_base)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(resume_text)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # Requirement Extraction with improved parsing
            req_prompt = ChatPromptTemplate.from_template("List top 5 technical requirements from this JD as a JSON list. Output ONLY the JSON. JD: {jd}")
            req_chain = req_prompt | llm | StrOutputParser()
            raw_requirements = req_chain.invoke({"jd": jd_text})
            requirements = clean_json_response(raw_requirements)

            docs = retriever.invoke(jd_text)
            context = "\n".join([d.page_content for d in docs])

            res_prompt = ChatPromptTemplate.from_template("Write 4 tailored resume bullet points based on this context and JD. Context: {context}. JD: {jd}")
            cv_prompt = ChatPromptTemplate.from_template("Write a professional cover letter based on this context and JD. Context: {context}. JD: {jd}")

            res_output = (res_prompt | llm | StrOutputParser()).invoke({"context": context, "jd": jd_text})
            cv_output = (cv_prompt | llm | StrOutputParser()).invoke({"context": context, "jd": jd_text})

            st.success("Done!")
            tab1, tab2, tab3 = st.tabs(["🎯 Tailored Resume", "✉️ Cover Letter", "🔍 RAG Insight"])

            with tab1:
                st.write(res_output)
            with tab2:
                st.write(cv_output)
            with tab3:
                st.write("**Extracted Job Requirements:**")
                st.json(requirements)
                st.write("**Retrieved Resume Chunks Used:**")
                st.info(context)
