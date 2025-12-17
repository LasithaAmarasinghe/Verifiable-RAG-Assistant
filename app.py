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
from langchain_community.tools import DuckDuckGoSearchRun # <--- NEW TOOL

# --- CONFIGURATION ---
st.set_page_config(page_title="RAG Knowledge Base", layout="wide")
st.title("📚 Verifiable RAG Assistant")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# --- STATE MANAGEMENT ---
if "processed_file_ids" not in st.session_state:
    st.session_state.processed_file_ids = set()

# --- MAIN LOGIC ---
if api_key and uploaded_files:
    os.environ["GROQ_API_KEY"] = api_key
    
    # Check for file changes
    current_file_ids = {f.file_id for f in uploaded_files}
    if current_file_ids != st.session_state.processed_file_ids:
        st.session_state.vector_db = None
        st.session_state.messages = []
        st.session_state.processed_file_ids = current_file_ids
        st.write("🔄 Document set changed. Rebuilding Knowledge Base...")

    # Initialize Vector DB
    if st.session_state.vector_db is None:
        all_splits = []
        with st.status("Processing Documents...", expanded=True) as status:
            for uploaded_file in uploaded_files:
                st.write(f"📄 Reading {uploaded_file.name}...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = uploaded_file.name
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                all_splits.extend(splits)
                
            status.update(label="Generating Embeddings (Local CPU)...", state="running")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
            st.session_state.vector_db = vectorstore
            status.update(label="✅ Knowledge Base Ready!", state="complete")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.vector_db is not None:
                llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
                
                # --- STEP 1: TRY PDF SEARCH ---
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                
                # We use a specific "trigger phrase" in the prompt to detect failure
                pdf_system_prompt = (
                    "You are a researcher. Answer ONLY based on the context provided. "
                    "If the specific answer is NOT in the context, strictly reply with exactly: 'SEARCH_WEB'."
                    "Do NOT try to answer from your own knowledge."
                    "\n\nContext:\n{context}"
                )
                
                pdf_prompt_template = ChatPromptTemplate.from_messages([
                    ("system", pdf_system_prompt),
                    ("human", "{input}"),
                ])
                
                pdf_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, pdf_prompt_template))
                response = pdf_chain.invoke({"input": prompt})
                answer = response["answer"]

                # --- STEP 2: CHECK IF WE NEED INTERNET ---
                if "SEARCH_WEB" in answer:
                    with st.spinner("Not found in PDF. Searching the Internet... 🌍"):
                        try:
                            search_tool = DuckDuckGoSearchRun()
                            search_results = search_tool.run(prompt)
                            
                            # Summarize the web results using the LLM
                            web_system_prompt = (
                                "You are a helpful assistant. The user asked a question that was not in their local documents. "
                                "Answer using the following web search results. "
                                "Explicitly state that this information is from the internet."
                                "\n\nWeb Results:\n{context}"
                            )
                            
                            web_prompt = ChatPromptTemplate.from_messages([
                                ("system", web_system_prompt),
                                ("human", "{input}"),
                            ])
                            
                            # We reuse the LLM but without retrieval, just summarization
                            web_chain = web_prompt | llm
                            web_response = web_chain.invoke({"input": prompt, "context": search_results})
                            
                            final_output = f"⚠️ **Note:** This information was not found in your PDFs.\n\n{web_response.content}\n\n**Source:** [Internet Search]"
                        except Exception as e:
                            final_output = f"I couldn't find that in the PDF, and the web search failed. Error: {e}"
                
                else:
                    # Found in PDF - Format Citations
                    sources_text = ""
                    if response['context']:
                        unique_sources = {}
                        for doc in response['context']:
                            filename = doc.metadata.get('source_file', 'Unknown')
                            page = doc.metadata.get('page', '?')
                            if filename not in unique_sources:
                                unique_sources[filename] = set()
                            unique_sources[filename].add(str(page + 1))
                        
                        sources_text = "\n\n**PDF Sources:**\n"
                        for fname, pages in unique_sources.items():
                            sources_text += f"- *{fname}*: Pages {', '.join(sorted(list(pages)))}\n"
                    
                    final_output = answer + sources_text

                st.markdown(final_output)
                st.session_state.messages.append({"role": "assistant", "content": final_output})

elif not api_key:
    st.info("Please enter your Groq API Key.")
elif not uploaded_files:
    st.info("Please upload PDFs.")