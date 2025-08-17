import os
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq


load_dotenv()



os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or ""
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY") or ""
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") or ""
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT") or ""
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT") or ""


st.title("Equity and News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Sidebar input for URLs
urls = []
for i in range(5):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")

# File path for storing FAISS index
file_path = "vector_index"

# Placeholder for progress updates
main_placeholder = st.empty()

# Initialize LLM  default settings here
# llm = OpenAI(temperature=0.9, max_tokens=500)

llm = ChatGroq(
    temperature=0.9,
    model="openai/gpt-oss-120b",   # or llama-3.3-70b-versatile,openai/gpt-oss-120b etc.
    max_tokens=500
)

# Initialize session state for embeddings and vectorstore
if "embeddings" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings()

if "vectorstore" not in st.session_state and os.path.exists(file_path):
    st.session_state.vectorstore = FAISS.load_local(
        file_path, st.session_state.embeddings, allow_dangerous_deserialization=True
    )

if process_url_clicked and urls:
    try:
        # Load data 
        loader = UnstructuredURLLoader(urls=urls, continue_on_failure=True)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=500,
            chunk_overlap=100,
            )
        main_placeholder.text("Splitting text into chunks...✅✅✅")
        docs = text_splitter.split_documents(data)

        # Create FAISS index
        vectorstore = FAISS.from_documents(docs, st.session_state.embeddings)
        vectorstore.save_local(file_path)

        # Store the vectorstore in session state
        st.session_state.vectorstore = vectorstore
        main_placeholder.text("FAISS index created and saved successfully.")
        st.success("Processing complete! You can now ask questions.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Question input and retrieval
query = st.text_input("Ask a question:")
if query:
    try:
        # Ensure vectorstore is loaded
        if "vectorstore" in st.session_state:
            vectorstore = st.session_state.vectorstore
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

            # Query the chain
            result = chain({"question": query}, return_only_outputs=True)

            # Display the answer
            st.header("Answer")
            st.write(result["answer"])

            # Display sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split sources by newline
                for source in sources_list:
                    st.write(source)
        else:
            st.error("FAISS index is not available. Please process URLs first.")
    except Exception as e:
        st.error(f"An error occurred while querying: {str(e)}")
