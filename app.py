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

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None case
    return text

# Extract text from DOCX
def get_docx_text(docx_files):
    text = ""
    for docx in docx_files:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# Split text into chunks for embedding
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Create FAISS vector store
def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Load vector store if exists
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

# Chat model configuration
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handles user input and updates chat history
def user_input(user_question, vector_store):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize chat history

    # Display the chat history before adding the new question
    display_chat()

    # Immediately show the user's question
    with st.chat_message("user"):
        st.markdown(f"**User:** {user_question}")

    # Add the user question to session state chat history
    st.session_state.chat_history.append(("User", user_question))

    # Show a loader while generating the response
    with st.spinner("Generating response..."):
        docs = vector_store.similarity_search(user_question)
        if not docs:
            bot_response = "No relevant documents found."
        else:
            chain = get_conversational_chain()
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            bot_response = response["output_text"]

    # Add bot response to chat history
    st.session_state.chat_history.append(("Bot", bot_response))

    # Show bot's response
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {bot_response}")

# Display chat history (excluding the latest user input)
def display_chat():
    # st.subheader("Chat History")

    for sender, message in st.session_state.chat_history:
        with st.chat_message("user" if sender == "User" else "assistant"):
            st.markdown(f"**{sender}:** {message}")

# Process uploaded text and update vector store
def process_text(raw_text):
    if not raw_text.strip():
        st.error("No text extracted. Please check the document.")
        return

    text_chunks = get_text_chunks(raw_text)
    if not text_chunks:
        st.error("No text chunks created. Please check the document content.")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load existing vector store if available
        if "vector_store" in st.session_state and st.session_state.vector_store:
            existing_vector_store = st.session_state.vector_store
            new_vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            existing_vector_store.merge_from(new_vector_store)  # Merge new embeddings
        else:
            existing_vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

        # Save updated vector store to session state
        st.session_state.vector_store = existing_vector_store
        st.success("Files processed successfully! You can now ask questions.")

    except Exception as e:
        st.error(f"Error creating/updating vector store: {e}")


# Main function
def main():
    st.set_page_config("SMART CHATBOT")
    st.header("SMART CHATBOT 🤖 | Chat with PDFs & DOCX!")

    st.markdown("""
        <style>
        .stChatMessage {background-color: #1e1e1e; color: white; padding: 10px; border-radius: 10px;}
        .stSidebar {background-color: #282828; color: white;}
        .stButton > button {width: 100%;}
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar file upload
    with st.sidebar:
        st.title("Upload Documents:")
        uploaded_files = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)

        if uploaded_files:
            if "processed_files" not in st.session_state:
                st.session_state.processed_files = []

            new_files = [file for file in uploaded_files if file.name not in st.session_state.processed_files]

            if new_files:
                with st.spinner("Processing files..."):
                    raw_text = ""
                    pdf_files = [f for f in new_files if f.type == "application/pdf"]
                    docx_files = [f for f in new_files if f.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]

                    if pdf_files:
                        raw_text += get_pdf_text(pdf_files)
                    if docx_files:
                        raw_text += get_docx_text(docx_files)

                    process_text(raw_text)
                    st.session_state.processed_files.extend([file.name for file in new_files])  # Mark as processed

    # Load existing vector store if not already loaded
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = load_vector_store()

    # User input at bottom
    st.markdown("---")
    user_question = st.chat_input("Ask a question about your uploaded documents...")

    if user_question and 'vector_store' in st.session_state and st.session_state.vector_store:
        user_input(user_question, st.session_state.vector_store)

if __name__ == "__main__":
    main()

