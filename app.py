import streamlit as st
from PyPDF2 import PdfReader
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

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save FAISS index
    return vector_store


# def load_vector_store():
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     if os.path.exists("faiss_index"):
#         return FAISS.load_local("faiss_index", embeddings)
#     return None

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists("faiss_index") or not os.listdir("faiss_index"):
        st.warning("FAISS index not found. Please upload a PDF and process it first.")
        return None
    return FAISS.load_local("faiss_index", embeddings)


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
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
    st.set_page_config("SMARTPDF CHATBOT")
    st.header("SMARTPDF-CHATBOT 🤖 | Chat with PDFs!")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and 'vector_store' in st.session_state:
        user_input(user_question, st.session_state.vector_store)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text extracted from the PDF files. Please check the files.")
                        return
                    
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("No text chunks created. Please check the PDF content.")
                        return
                    
                    try:
                        st.session_state.vector_store = create_vector_store(text_chunks)
                        st.success("PDF Uploaded Successfully, Now Ask Questions.")
                    except Exception as e:
                        st.error(f"Error creating vector store: {e}")
            else:
                st.error("Please upload at least one PDF file.")
    
    # Load FAISS index if available
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = load_vector_store()

if __name__ == "__main__":
    main()
