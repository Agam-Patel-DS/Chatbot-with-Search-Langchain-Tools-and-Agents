# Importing Important Library
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
import streamlit as st
from dotenv import load_dotenv

# Loading Environvent Variables (API Keys)
load_dotenv()
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

# Loading Embeddings (from Hugging Face API)
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Wikipedia API Wrapper
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# Arxiv API Wrapper
api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# Duck Duck Go Search Tool
search=DuckDuckGoSearchRun(name="Search")

# Streamlit App
st.title("Langchain - Chat with Search")
"""
In this example, we are using 'StreamlitCallbackHandler' to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more Langchain + Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent].
"""

# Sidebar
st.sidebar.title("Info")
groq_api_key=st.sidebar.text_input("Enter your Groq API Key", type="password")

if "messages" not in st.session_state:
  st.session_state["messages"]=[
    {"role":"assistance", "content":"Hi, I am an assisatance who can search on web. How can I help you?"},
  ]

for msg in st.session_state.messages:
  st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
  st.session_state.messages.append({"role":"user", "content":prompt})
  st.chat_message("user").write(prompt)

  # Loading Large Language Models (from Groq API)
  llm=ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", streaming=True)
  tools=[search, arxiv, wiki]

  search_agent=initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

  with st.chat_message("assistance"):
    st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    response=search_agent.run(st.session_state.messages, callbacks=[st_cb])
    st.session_state.messages.append({"role":"assistance", "content":response})
    st.write(response)

















