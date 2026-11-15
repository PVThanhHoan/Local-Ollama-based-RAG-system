from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings,  OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from IPython.display import display, Markdown
import streamlit as st
import tempfile
import os
  # Folder that contains .txt files.
CHUNK_SIZE = 1000 # Maximum characters per text chunk.
CHUNK_OVERLAP = 200 # Number of characters shared between nearby chunks.
EMBED_MODEL = "nomic-embed-text" # Ollama embedding model used to convert text to vectors (Remember to pull the embedding model by running the command: ollama pull nomic-embed-text)
LLM_MODEL = "qwen2.5:0.5b" # Ollama model name for generating answers (Remember to pull the LLM model by running the command: ollama pull qwen2.5:0.5b).
TEMPERATURE = 0.7 # Controls how random or creative the model response is (the higher, the more creative; 0.0: deterministic).
OLLAMA_URL = "http://127.0.0.1:11434" # URL of local (127.0.0.1) Ollama server. Its default port is 11434

st.title("Ollama based Local Retrieval Argument System")
st.write("***:red[WARNING]***")
st.write('All large language models like the one this Local Rag system uses are known to be inaccurate at times')  
st.write('This programm was designed only to advise, not make decision for you /n ')
st.write('This programm nor its maker will be held responsible for any decision you make based on its output')
 
uploaded_file = st.file_uploader("upload a PDF",accept_multiple_files=False, type=["pdf"])

if uploaded_file is not None: 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:     # save the pdf later
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
    loader = PyPDFLoader(temp_path)   # loading and extracting the texts content from the uploaded pdf
    data = loader.load()

    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(data)

    vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL),
    collection_name="local-rag"
    )
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)

    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate 2
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
    )
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )


    a='x'
    c=0
    while a=='x':
        a='o'
        st.write('*****************************************************************************')
        st.write("Enter your question")
        question = st.text_input(label='',key=c)
        if question:
            response = chain.invoke(question)
            st.write('Answer: ')
            st.write(response)
            c+=1
            a='x'





