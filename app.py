# Importing Necessary Dependencies
import os
import time
from langchain_core.prompts import prompt
from langchain_core.tools import retriever
from sqlalchemy.sql.ddl import exc
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langchain.prompts.chat import ChatPromptTemplate

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Initializing Groq env varaible
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

st.title('Groq-Llama3-RAG-Based-Project')

class InitializeSessions:
    FILE_PATH = os.path.join(os.getcwd(), 'Files')
    
    @staticmethod
    def initializeVectorDB():
        if 'vectordb' not in st.session_state:
            try:
                st.session_state.loader = PyPDFDirectoryLoader(InitializeSessions.FILE_PATH)
                st.session_state.documents = st.session_state.loader.load()

                if not st.session_state.documents:
                    st.warning("No valid PDFs found in the directory.")
                    return

                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)

                st.session_state.embeddings = OllamaEmbeddings(model='nomic-embed-text')
                st.session_state.vectordb = Chroma.from_documents(documents=st.session_state.final_documents, embedding=st.session_state.embeddings, persist_directory='./chroma_db')
        

            except Exception as e:
                print(e)

    @staticmethod
    def initializeRetrievalQAChain():
        try:
            if 'retrieval_chain' not in st.session_state:
                st.session_state.llm = ChatGroq(
                    groq_api_key=os.getenv('GROQ_API_KEY'),
                    model_name="Llama3-8b-8192"
                )

                st.session_state.template = """
                    Answer the questions based on the provided context only.
                    Please provide the most accurate response based on the question
                    <context>
                    {context}
                    <context>

                    Questions:{input}
                """

                st.session_state.prompt = ChatPromptTemplate.from_messages(
                    [
                        ('system', 'You are an helpful assistant'),
                        ('user', st.session_state.template)
                    ]
                )

                st.session_state.document_chain = create_stuff_documents_chain(st.session_state.llm, st.session_state.prompt)

                st.session_state.retriever = st.session_state.vectordb.as_retriever()

                st.session_state.retrieval_chain = create_retrieval_chain(st.session_state.retriever, st.session_state.document_chain)

                # st.session_state.retrievalqa_chain = RetrievalQA.from_chain_type(
                #     llm = st.session_state.llm,
                #     chain_type = 'stuff',
                #     retriever = st.session_state.retriever,
                #     return_source_documents = True,
                #     chain_type_kwargs = {'prompt' : st.session_state.prompt}
                # )

        except Exception as e:
            print(e)
            
class ChatBot:
    @staticmethod
    def initializeChat(query: str):
        try:
            start = time.process_time()

            if "retrieval_chain" not in st.session_state:
                print("Error: retrieval_chain  is not initialized.")
                return None, None  # Ensure it returns a tuple

            response = st.session_state.retrieval_chain.invoke({'input': query})

            end_time = time.process_time() - start

            if response is None:
                print("Error: Response is None.")
                return None, None  # Ensure it returns a tuple

            return response, end_time

        except Exception as e:
            print(f"Error in initializeChat: {e}")
            return None, None  # Ensure it returns a tuple

# Initializing Sessions
InitializeSessions.initializeVectorDB()
InitializeSessions.initializeRetrievalQAChain()

user_query = st.chat_input('What you want to ask...')

if user_query:

    response, response_time = ChatBot.initializeChat(user_query)

    if response is None:
        st.error("An error occurred. Please try again.")

    else:
        st.write(f"Response time: {response_time:.2f} seconds")

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(response.get('answer', "No answer found."))

        # Show document similarity search results
        with st.expander("Document Similarity Search"):
            if "context" in response:
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            else:
                st.write("No similar documents found.")
