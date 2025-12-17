import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="RAG Knowledge Base", layout="wide")
st.title("📚 Verifiable RAG Assistant")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    
    # ENABLE MULTIPLE FILES
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# --- STATE MANAGEMENT ---
# We need to track the set of file IDs to detect changes
if "processed_file_ids" not in st.session_state:
    st.session_state.processed_file_ids = set()

# --- MAIN LOGIC ---
if api_key and uploaded_files:
    os.environ["GROQ_API_KEY"] = api_key
    
    # Check if the uploaded file list has changed
    current_file_ids = {f.file_id for f in uploaded_files}
    
    # If the set of files is different, reload everything
    if current_file_ids != st.session_state.processed_file_ids:
        st.session_state.vector_db = None
        st.session_state.messages = []
        st.session_state.processed_file_ids = current_file_ids
        st.write("🔄 Document set changed. Rebuilding Knowledge Base...")

    # Initialize Vector DB (Process ALL files together)
    if st.session_state.vector_db is None:
        all_splits = []
        
        with st.status("Processing Documents...", expanded=True) as status:
            for uploaded_file in uploaded_files:
                st.write(f"📄 Reading {uploaded_file.name}...")
                
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Load & Split
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # Add source metadata so we know which file the answer came from
                for doc in docs:
                    doc.metadata["source_file"] = uploaded_file.name
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                all_splits.extend(splits)
                
            status.update(label="Generating Embeddings (Local CPU)...", state="running")
            
            # Embed all chunks at once
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
            
            st.session_state.vector_db = vectorstore
            status.update(label=f"✅ Knowledge Base Ready! ({len(all_splits)} chunks from {len(uploaded_files)} files)", state="complete")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question across your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.vector_db is not None:
                # Retrieve (k=3 to get more context from multiple docs)
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
                
                system_prompt = (
                    "You are a researcher. Answer ONLY based on the context provided. "
                    "If the documents don't agree, mention the conflict. "
                    "Always cite the filename along with the page number."
                    "\n\nContext:\n{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_template))
                response = chain.invoke({"input": prompt})
                
                answer = response["answer"]
                
                # Format Sources with Filenames
                sources_text = ""
                if response['context']:
                    unique_sources = {}
                    for doc in response['context']:
                        filename = doc.metadata.get('source_file', 'Unknown')
                        page = doc.metadata.get('page', '?')
                        if filename not in unique_sources:
                            unique_sources[filename] = set()
                        unique_sources[filename].add(str(page + 1))
                    
                    sources_text = "\n\n**Sources:**\n"
                    for fname, pages in unique_sources.items():
                        sources_text += f"- *{fname}*: Pages {', '.join(sorted(list(pages)))}\n"

                final_output = answer + sources_text
                st.markdown(final_output)
                st.session_state.messages.append({"role": "assistant", "content": final_output})

elif not api_key:
    st.info("Please enter your Groq API Key.")
elif not uploaded_files:
    st.info("Please upload one or more PDF documents.")