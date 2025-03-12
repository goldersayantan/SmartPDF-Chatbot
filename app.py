import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    st.error("Google API key not found. Please set it in the .env file.")
else:
    genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""  # Handle None case
            text += page_text
    return text

def get_docx_text(docx_files):
    text = ""
    for docx in docx_files:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save FAISS index
    return vector_store

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    if not docs:
        st.write("No relevant documents found.")
        return
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("SMART CHATBOT")
    st.header("SMART CHATBOT 🤖 | Chat with PDFs & DOCX!")

    user_question = st.text_input("Ask a Question from the Documents")

    if user_question and 'vector_store' in st.session_state:
        user_input(user_question, st.session_state.vector_store)
    
    with st.sidebar:
        st.title("Menu:")
        file_type = st.radio("Choose file type to upload", ("PDF", "DOCX"))
        
        if file_type == "PDF":
            pdf_docs = st.file_uploader("Upload your PDF Files", type=["pdf"], accept_multiple_files=True)
            if st.button("Submit PDF & Process"):
                if pdf_docs:
                    with st.spinner("Processing PDF..."):
                        raw_text = get_pdf_text(pdf_docs)
                        process_text(raw_text)
                else:
                    st.error("Please upload at least one PDF file.")
        
        elif file_type == "DOCX":
            docx_docs = st.file_uploader("Upload your DOCX Files", type=["docx"], accept_multiple_files=True)
            if st.button("Submit DOCX & Process"):
                if docx_docs:
                    with st.spinner("Processing DOCX..."):
                        raw_text = get_docx_text(docx_docs)
                        process_text(raw_text)
                else:
                    st.error("Please upload at least one DOCX file.")
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = load_vector_store()

def process_text(raw_text):
    if not raw_text.strip():
        st.error("No text extracted. Please check the document.")
        return
    
    text_chunks = get_text_chunks(raw_text)
    if not text_chunks:
        st.error("No text chunks created. Please check the document content.")
        return
    
    try:
        st.session_state.vector_store = create_vector_store(text_chunks)
        st.success("File Uploaded Successfully, Now Ask Questions.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

if __name__ == "__main__":
    main()
