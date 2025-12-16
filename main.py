import os
import getpass
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. SETUP KEYS ---
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY is None or GROQ_API_KEY.startswith("PASTE"):
    print("Please set your Groq API Key in the environment variable 'GROQ_API_KEY'!")
    exit()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def build_hybrid_rag(pdf_path):
    print("--- 1. Ingesting Data ---")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    # --- 2. EMBEDDINGS (Local & Free) ---
    print("--- 2. Generating Embeddings (Local CPU) ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # --- 3. LLM (Cloud - Groq) ---
    # UPDATED: Using Llama 3.1 (The new standard)
    print("--- 3. Initializing Llama 3.1 on Groq ---")
    llm = ChatGroq(
        model="llama-3.1-8b-instant", # <--- FIXED HERE
        temperature=0
    )

    system_prompt = (
        "You are a precise research assistant representing a university. "
        "Your task is to answer the user's question ONLY based on the provided context snippets. "
        "Do NOT just repeat or regurgitate the context text. Extract the specific answer and present it clearly. "
        "If the exact answer is not present in the context paragraphs below, state 'I do not know based on this document'."
        "\n\n"
        "--- Context Start ---\n{context}\n--- Context End ---"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    import argparse
    
    # Create a parser so we can read from the command line
    parser = argparse.ArgumentParser(description="Run RAG on any PDF")
    parser.add_argument("filename", type=str, help="The path to the PDF file you want to analyze")
    args = parser.parse_args()
    
    pdf_path = args.filename # Now it takes whatever you type!
    
    if not os.path.exists(pdf_path):
        print(f" ERROR: The file '{pdf_path}' does not exist.")
    else:
        try:
            chain = build_hybrid_rag(pdf_path)
            print("\n System Ready! (Powered by Groq Llama 3.1 + HuggingFace)")
            
            while True:
                query = input("\nAsk Question (or 'q' to quit): ")
                if query.lower() == 'q': break
                
                print("Thinking...")
                response = chain.invoke({"input": query})
                print(f"\nAI: {response['answer']}")
                
                if response['context']:
                    page_num = response['context'][0].metadata.get('page', 'Unknown')
                    print(f"[Source: Page {page_num}]")
                    
        except Exception as e:
            print(f"\n Error: {e}")