import sys
import os
import warnings
import streamlit as st
import tempfile
import warnings
warnings.filterwarnings("ignore")
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# --- 1. SETUP & CONFIG ---
warnings.filterwarnings("ignore")
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

st.set_page_config(page_title="RAG Knowledge Base", layout="wide")
st.title("📚 Verifiable RAG Assistant")

# --- SIDEBAR ---
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
        st.write("🔄 Documents changed. Rebuilding index...")

    # Initialize Vector DB (With Optimized Chunk Sizes)
    if st.session_state.vector_db is None:
        all_splits = []
        with st.status("Reading & Indexing...", expanded=True) as status:
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = uploaded_file.name
                
                # OPTIMIZED: Smaller chunks to prevent Rate Limit Errors
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)
                all_splits.extend(splits)
                
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_db = Chroma.from_documents(documents=all_splits, embedding=embeddings)
            status.update(label="✅ Ready to Search!", state="complete")

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
                
                # --- PHASE 1: PDF RETRIEVAL ---
                # Retrieve top 2 chunks (Lower count to save tokens)
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 2})
                
                pdf_system_prompt = (
                    "You are a senior researcher. Answer ONLY based on the provided context. "
                    "If the user asks for external information (like stock prices, current events, or authors' companies) "
                    "that is NOT in the context, you must output exactly: 'SEARCH_WEB'. "
                    "Do NOT try to guess. "
                    "\n\nContext:\n{context}"
                )
                
                pdf_prompt = ChatPromptTemplate.from_messages([
                    ("system", pdf_system_prompt),
                    ("human", "{input}"),
                ])
                
                chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, pdf_prompt))
                
                # Run the chain
                with st.spinner("Analyzing Documents..."):
                    response = chain.invoke({"input": prompt})
                    answer = response["answer"]
                
                final_output = answer

                # --- PHASE 2: WEB SEARCH FALLBACK (Smart Query Refinement) ---
                if "SEARCH_WEB" in answer:
                    with st.spinner("🔍 Refining search query..."):
                        # Step 1: Extract the Entity from PDF Context (The "Smart" Step)
                        # We use the retrieve context to find "Microsoft" first
                        pdf_context = ""
                        if response['context']:
                            for doc in response['context']:
                                pdf_context += f"{doc.page_content}\n"
                        
                        # Ask LLM to generate a better search query
                        query_generation_prompt = (
                            "You are a search query optimizer. "
                            "The user asked: '{input}' "
                            "Based on the following PDF context, identify the specific entity (company, person, etc.) "
                            "and write a simple web search query to find the answer. "
                            "Output ONLY the search query. "
                            "\n\nPDF Context:\n{context}"
                        )
                        
                        query_gen_chain = ChatPromptTemplate.from_template(query_generation_prompt) | llm
                        optimized_query = query_gen_chain.invoke({
                            "input": prompt, 
                            "context": pdf_context
                        }).content.strip()
                        
                        st.write(f"**Searching Web for:** *{optimized_query}*") # Debug info for Demo

                    with st.spinner(f"Searching: {optimized_query} 🌍"):
                        # Step 2: Search with the OPTIMIZED query (e.g., "Microsoft stock price")
                        search = DuckDuckGoSearchRun()
                        web_results = search.run(optimized_query)
                        
                        # Step 3: Synthesize
                        synthesis_prompt = (
                            "You are a smart research assistant. "
                            "Combine the internal PDF knowledge and external Web search results to answer the user. "
                            "\n\n1. Internal PDF Context:\n{pdf_context}"
                            "\n\n2. External Web Search Results:\n{web_context}"
                            "\n\nAnswer the user's question explicitly citing sources."
                        )
                        
                        final_prompt = ChatPromptTemplate.from_messages([
                            ("system", synthesis_prompt),
                            ("human", "{input}"),
                        ])
                        
                        synthesis_chain = final_prompt | llm
                        final_response = synthesis_chain.invoke({
                            "input": prompt, 
                            "pdf_context": pdf_context, 
                            "web_context": web_results
                        })
                        
                        final_output = final_response.content + "\n\n**[Sources: PDF + Web Search]**"

                else:
                    # Pure PDF Answer - Add Citations
                    if response['context']:
                        sources = {}
                        for doc in response['context']:
                            fname = doc.metadata.get('source_file', 'Doc')
                            page = str(doc.metadata.get('page', 0) + 1)
                            if fname not in sources: sources[fname] = []
                            sources[fname].append(page)
                        
                        citation_text = "\n\n**Sources:**\n"
                        for f, p in sources.items():
                            citation_text += f"- *{f}*: Pages {', '.join(p)}\n"
                        final_output += citation_text

                st.markdown(final_output)
                st.session_state.messages.append({"role": "assistant", "content": final_output})

elif not api_key:
    st.info("Please enter your Groq API Key.")
elif not uploaded_files:
    st.info("Please upload PDFs.")